import logging
import os
import tempfile
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from langchain_core.messages import HumanMessage

from agent import create_crypto_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "deepseek-v4-flash": {"name": "deepseek-v4-flash", "provider": "deepseek", "label": "DeepSeek V4 Flash"},
    "deepseek-v4-pro": {"name": "deepseek-v4-pro", "provider": "deepseek", "label": "DeepSeek V4 Pro"},
    "qwen3.6-max-preview": {"name": "qwen3.6-max-preview", "provider": "openai", "label": "Qwen 3.6 Max"},
    "kimi-k2.6": {"name": "kimi-k2.6", "provider": "kimi", "label": "Kimi K2.6"},
    "mimo-v2.5-pro": {"name": "mimo-v2.5-pro", "provider": "mimo", "label": "Mimo V2.5 Pro"},
}

agents: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application started")
    yield
    logger.info("Shutting down application...")


app = FastAPI(title="CryptoExpert Pro", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatMessage(BaseModel):
    message: str
    thread_id: str | None = None
    model: str = "deepseek-v4-flash"
    file_data: dict | None = None


UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "cryptoexpert_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".csv", ".json", ".bin", ".py"}


def parse_file_content(file_path: str, filename: str) -> dict:
    ext = os.path.splitext(filename)[1].lower()
    result = {"filename": filename, "type": ext, "size": os.path.getsize(file_path)}

    try:
        if ext in (".txt", ".csv", ".json", ".py"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            result["content"] = content
            result["preview"] = content[:2000] if len(content) > 2000 else content

            if ext == ".csv":
                rows = [row.strip() for row in content.strip().split("\n") if row.strip()]
                values = []
                for row in rows:
                    cells = [c.strip() for c in row.replace("\t", ",").split(",") if c.strip()]
                    values.extend(cells)
                result["parsed_values"] = values[:512]

            elif ext == ".json":
                try:
                    data = json.loads(content)
                    result["parsed_json"] = str(data)[:2000]
                except json.JSONDecodeError:
                    pass

        elif ext == ".bin":
            with open(file_path, "rb") as f:
                raw = f.read()
            hex_str = raw.hex()
            result["content"] = hex_str
            result["preview"] = " ".join(hex_str[i:i+2] for i in range(0, min(1024, len(hex_str)), 2))
            result["total_bytes"] = len(raw)

        else:
            result["content"] = f"不支持的文件类型: {ext}"

    except Exception as e:
        result["content"] = f"文件解析失败: {str(e)}"
        result["error"] = True

    return result


# 1. 路由：提供前端 HTML 页面
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


# 2. 文件上传接口
@app.post("/api/upload/")
async def upload_file(file: UploadFile = File(...)) -> dict:
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}，支持: {', '.join(ALLOWED_EXTENSIONS)}")

    safe_name = f"{os.urandom(8).hex()}_{filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    logger.info(f"File uploaded: {filename} -> {safe_name}, size: {len(content)}")

    parsed = parse_file_content(file_path, filename)
    parsed["saved_path"] = file_path
    parsed["status"] = "success"
    return parsed


# 3. 路由：聊天接口
@app.post("/api/chat/")
async def chat_endpoint(chat_msg: ChatMessage) -> dict:
    thread_id = chat_msg.thread_id or os.urandom(16).hex()
    model = chat_msg.model
    logger.info(f"Chat request received, thread_id: {thread_id}, model: {model}")

    if model not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}")

    if model not in agents:
        logger.info(f"Creating new agent for model: {model}")
        try:
            config = MODEL_CONFIGS[model]
            agents[model] = create_crypto_agent(config["name"], config["provider"])
            logger.info(f"Agent created successfully for model: {model}")
        except Exception as e:
            logger.error(f"Failed to create agent: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")

    agent = agents[model]

    try:
        user_message = chat_msg.message
        if chat_msg.file_data:
            fd = chat_msg.file_data
            file_context = f"\n\n[用户上传了文件 {fd.get('filename', '')} ({fd.get('type', '')}, {fd.get('size', 0)} bytes)]\n"
            if fd.get("content"):
                file_context += f"文件内容：\n```\n{fd['content'][:8000]}\n```\n"
            if fd.get("parsed_values"):
                file_context += f"\nCSV解析数据（前{min(len(fd['parsed_values']), 512)}个值）：{fd['parsed_values'][:512]}\n"
            if fd.get("total_bytes"):
                file_context += f"\n二进制文件大小：{fd['total_bytes']} 字节，hex预览：{fd.get('preview', '')}\n"
            user_message = user_message + file_context

        config = {"configurable": {"thread_id": thread_id}}

        state = agent.get_state(config)
        if state and state.values and "messages" in state.values:
            messages = state.values["messages"]
            cleaned = []
            last_ai_tool_idx = None
            for i, msg in enumerate(messages):
                if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
                    last_ai_tool_idx = i
                if msg.type == "tool":
                    last_ai_tool_idx = None

            if last_ai_tool_idx is not None:
                cleaned = [msg for i, msg in enumerate(messages) if i <= last_ai_tool_idx]
                agent.update_state(config, {"messages": cleaned})

        response = agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
        )

        messages = response["messages"]
        logger.info(f"Response messages count: {len(messages)}")
        
        ai_reply = ""
        for msg in reversed(messages):
            if msg.type == "ai" and msg.content:
                ai_reply = msg.content
                break
            elif msg.type == "ai":
                logger.info(f"Found AI message with empty content: {msg}")

        if not ai_reply:
            logger.warning(f"No AI response found, last 3 messages: {messages[-3:] if len(messages) >= 3 else messages}")

        logger.info(f"Chat response sent, thread_id: {thread_id}")
        return {"status": "success", "reply": ai_reply, "thread_id": thread_id, "model": model}
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history/{thread_id}")
async def get_history(thread_id: str) -> dict:
    logger.info(f"History request for thread_id: {thread_id}")
    default_model = "deepseek-v4-flash"
    if default_model not in agents:
        config = MODEL_CONFIGS[default_model]
        agents[default_model] = create_crypto_agent(config["name"], config["provider"])
    agent = agents[default_model]
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = agent.get_state(config)

        if not state or "messages" not in state.values:
            return {"status": "success", "history": []}

        raw_messages = state.values.get("messages", [])
        cleaned_messages = []
        last_ai_tool_idx = None
        for i, msg in enumerate(raw_messages):
            if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
                last_ai_tool_idx = i
            if msg.type == "tool":
                last_ai_tool_idx = None

        if last_ai_tool_idx is not None:
            cleaned_messages = [msg for i, msg in enumerate(raw_messages) if i <= last_ai_tool_idx]
        else:
            cleaned_messages = raw_messages

        formatted_history = []

        for msg in cleaned_messages:
            if msg.type == "human":
                formatted_history.append({"role": "user", "content": msg.content})
            elif msg.type == "ai" and msg.content:
                formatted_history.append({"role": "ai", "content": msg.content})

        logger.info(f"History retrieved, {len(formatted_history)} messages")
        return {"status": "success", "history": formatted_history}
    except Exception as e:
        logger.error(f"History error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def get_models() -> dict:
    return {"status": "success", "models": MODEL_CONFIGS}


@app.delete("/api/chat/{thread_id}")
async def delete_chat(thread_id: str) -> dict:
    logger.info(f"Deleting chat thread: {thread_id}")
    try:
        import sqlite3
        conn = sqlite3.connect("resources/test.db", check_same_thread=False, timeout=10)
        conn.execute("PRAGMA busy_timeout = 5000")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,))
        conn.commit()
        conn.close()
        logger.info(f"Chat thread {thread_id} deleted from database")
        return {"status": "success", "message": "Chat deleted"}
    except Exception as e:
        logger.error(f"Delete error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(app, host="127.0.0.1", port=8001)