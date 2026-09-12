# <img src="static/logo.ico" width="32"> CryptoExpert Pro

> **AI 驱动的密码算法分析助手**
>
> 本项目是一款基于 LangChain + LangGraph 框架开发的智能分析工具，结合大语言模型（LLM）的逻辑推理能力与专业的密码学检测工具，为用户提供 S 盒性能评估、随机性检测及密码算法深度分析等功能。

---

## 核心特性

### S 盒安全性评估
集成专业工具集，可自动计算多种关键安全指标：

| 类别 | 指标 |
|------|------|
| **结构特性** | 置换阶(OP)、不动点(FP)、反不动点(OFP) |
| **扩散特性** | 差分均匀性(DU)、差分分支数(DBN)、线性分支数(LBN)、线性逼近概率(LAP)、非线性度(NL) |
| **密码准则** | 严格雪崩准则(SAC)、比特独立准则(BIC)、相关免疫性(CI)、传播特性(PC) |
| **代数攻击** | 代数免疫性(AI)、代数次数、透明阶(TO) |
| **侧信道攻击** | DPA-SNR、回旋均匀性(BU)、差分-线性均匀性(DLU)、无扰动比特密度(UBD) |
| **其他** | 线性结构检测、平方和指标(SSI)、线性近似表(LAT) |

### 随机性检测系统
基于 NIST SP800-22 及国标标准的随机性测试：

- 单比特频数检测 (Monobit Frequency Test)
- 游程总数检测 (Runs Test)
- 游程分布检测 (Runs Distribution Test)
- 扑克检测 (Poker Test)
- 重叠子序列检测 (Overlap Test)

### 分组密码结构性测试
基于卡方拟合优度检验的分组密码基础安全特性检测，支持 64 / 80 / 128 / 192 / 256 比特分组：

| 测试项 | 评估目标 |
|--------|----------|
| **明密文独立性测试** | 混淆能力 —— 明文与密文是否存在统计依赖 |
| **明文扩散雪崩测试** | 扩散能力 —— 单比特明文变化能否均匀扩散至密文 |
| **密钥雪崩有效性测试** | 密钥敏感度 —— 单比特密钥变化能否充分改变密文 |

- 支持两档采样规模：`demo`（快速，默认）和 `formal`（标准高精度，2^20 样本）
- 判定标准：`p-value >= 0.01` 视为测试通过
- 一键全测入口：`run_all_struct_tests` 自动顺序执行三项检测

### 文件上传分析
支持用户上传数据文件进行分析，告别手动粘贴：

| 文件类型 | 用途 |
|---------|------|
| `.txt` | 纯文本数据（S盒值、二进制序列、十六进制密文） |
| `.csv` | 表格格式的 S 盒数据（自动解析为值列表） |
| `.json` | 结构化数据（算法元信息、S盒定义） |
| `.bin` | 原始二进制文件（自动转为 hex 格式用于随机性检测） |
| `.py` | 用户自定义的加密算法脚本 |

支持点击上传和拖拽上传两种方式。

### 智能对话系统
- 支持多模型切换（DeepSeek V4 Flash / DeepSeek V4 Pro / Qwen 3.6 / Kimi K2.6 / Mimo V2.5 Pro）
- 自动工具调用（网络搜索 + 本地计算工具）
- 对话持久化（SQLite 存储上下文）
- 文件数据自动解析与上下文注入

---

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    前端 (HTML/CSS/JS)                   │
│  - 侧边栏会话管理  - 模型切换  - 文件上传/拖拽          │
│  - 文件预览  - Markdown渲染                             │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP (multipart/form-data)
┌────────────────────▼────────────────────────────────────┐
│                 FastAPI Server (main.py)                │
│  - RESTful API  - 文件解析  - 多模型路由  - 会话管理     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│           LangGraph Agent (agent.py)                    │
│  - ReAct 推理  - 工具编排  - Checkpointer               │
│  - 文件内容上下文注入                                    │
└───────┬────────────────────┬─────────────────┬─────────┘
        │                    │                 │
   ┌────▼────┐        ┌─────▼─────┐    ┌──────▼──────┐   ┌──────────────┐
   │ Web     │        │  S盒      │    │ 随机性      │   │ 结构性       │
   │ Search  │        │ Tools     │    │ Tools       │   │ Tools        │
   │ (Tavily)│        │ (22个)    │    │ (5个)       │   │ (4个)        │
   └─────────┘        └───────────┘    └─────────────┘   └──────────────┘
```

---

## 环境配置

### 1. 环境要求
- **Python**: 3.12 或更高版本
- **依赖管理**: uv（项目已包含 pyproject.toml）

### 2. 克隆项目
```bash
git clone https://github.com/Lee-hyun-R/CryptoExpert-Pro.git
cd CryptoExpert-Pro
```

### 3. 配置环境变量
新建 `.env` 文件，参考以下内容：
```bash
# DeepSeek 配置
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_API_BASE=https://api.deepseek.com/v1

# 通义千问配置
DASHSCOPE_API_KEY=your_qwen_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Kimi (Moonshot) 配置
KIMI_API_KEY=your_kimi_key
KIMI_BASE_URL=https://api.moonshot.cn/v1

# Mimo (小米) 配置
MIMO_API_KEY=your_mimo_key
MIMO_BASE_URL=https://api.xiaomimimo.com/v1

# 网络搜索配置
TAVILY_API_KEY=your_tavily_key

# 数据库路径（可选）
DB_PATH=resources/test.db
```

### 4. 安装依赖
```bash
# 使用 uv 安装
uv sync

# 或使用 pip
pip install -r pyproject.toml
```

### 5. 启动服务
```bash
# 创建数据目录
mkdir -p resources

# 启动服务
python main.py
# 访问 http://127.0.0.1:8001
```

启动后在浏览器打开 http://127.0.0.1:8001 即可使用。支持对话、文件上传（.txt/.csv/.json/.bin/.py）及拖拽上传。

### 6. Docker 部署
```bash
# 构建并运行
docker-compose up -d

# 停止服务
docker-compose down
```

---

## 项目结构

```
Agent_load/
├── main.py              # FastAPI 服务入口
├── agent.py             # LangGraph Agent 构建
├── clear_db.py          # 数据库清理工具
├── pyproject.toml       # 依赖配置
├── Dockerfile           # Docker 镜像配置
├── docker-compose.yml   # Docker Compose 配置
├── .env                 # 环境变量（需自行创建）
├── static/
│   └── index.html       # 前端页面
├── Tools/
│   ├── __init__.py
│   ├── code_executor.py       # Python 代码执行工具
│   ├── randomness_tools.py    # 随机性检测工具 (5个)
│   ├── Sbox_tools_1.py        # S盒工具集1 (11个)
│   ├── Sbox_tools_2.py        # S盒工具集2 (11个)
│   ├── structural_tools.py    # 结构性测试工具 (4个)
└── resources/
    └── test.db          # SQLite 数据库（自动创建）
```

---

## API 接口

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | 获取前端页面 |
| POST | `/api/upload/` | 上传文件（支持 .txt/.csv/.json/.bin/.py） |
| POST | `/api/chat/` | 发送聊天消息（可选携带 file_data） |
| GET | `/api/history/{thread_id}` | 获取历史记录 |
| GET | `/api/models` | 获取可用模型列表 |
| DELETE | `/api/chat/{thread_id}` | 删除对话记录 |

---

## 使用示例

### S盒分析
用户可以发送 S 盒数据进行分析：
```
请分析这个S盒 [0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, ...] 的安全性
```

### 随机性检测
发送二进制序列进行检测：
```
请检测这个序列 101100110101... 的随机性
```

### 文件上传分析
用户可以直接上传文件（点击上传按钮或拖拽文件到页面）：
1. 点击输入框左侧的上传按钮，选择文件（支持 `.txt` `.csv` `.json` `.bin` `.py`）
2. 或直接将文件拖拽到页面任意位置
3. 文件上传后会显示在输入框上方，可以点击 ✕ 移除
4. 输入问题后发送，文件内容会自动附加到对话中供 Agent 分析

**示例对话：**
```
📎 已上传文本文件：sbox_data.txt
帮我分析这个S盒的安全性
```

Agent 收到文件内容后会询问分析细节，确认后自动调用 S 盒工具进行全面评估。

### 结构性测试分析
对加密算法进行结构性安全检测：

**方式一：上传加密算法 `.py` 文件**
```
📎 已上传加密算法：my_cipher.py
帮我做结构性测试，分组长度128位，密钥128位，demo模式
```

**方式二：直接在对话中贴代码**
```
请对以下加密算法做结构性测试（64位分组/密钥，demo模式）：

def encrypt(key: int, plain: int) -> int:
    state = plain ^ key
    for _ in range(4):
        state ^= (state >> 3) ^ (state << 7)
        state &= (1 << 64) - 1
    return state
```

Agent 会自动识别 `encrypt(key, plain)` 函数，调用 `run_all_struct_tests` 执行三项检测（明密文独立性、明文扩散雪崩、密钥雪崩有效性），并返回每项的卡方统计量、p-value、频数分布与中文结论。

---

## 依赖库

```
fastapi>=0.136.1
langchain>=1.2.15
langchain-deepseek>=1.0.1
langchain-openai>=1.2.0
langchain-tavily>=0.2.18
langgraph>=1.1.9
langgraph-checkpoint-sqlite>=3.0.3
numpy>=2.4.4
scipy>=1.17.1
uvicorn>=0.46.0
python-dotenv>=1.2.2
```

---

