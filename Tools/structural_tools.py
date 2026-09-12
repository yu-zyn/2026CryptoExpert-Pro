# 分组密码结构性测试工具集合
# 三大测试：明密文独立性、明文扩散雪崩、密钥雪崩有效性
# 判定标准 p-value >= 0.01 测试通过

import random
import numpy as np
from scipy.special import gammaincc
from langchain.tools import tool


# ====================== 结构性测试核心逻辑 ======================

# 各分组长度对应5区间理论概率，参考结构性测试规范
GROUP_THEO_PROB = {
    64:  [0.190866, 0.163124, 0.292020, 0.163124, 0.190866],
    80:  [0.157153, 0.211624, 0.262446, 0.211624, 0.157153],
    128: [0.165467, 0.230036, 0.208994, 0.230036, 0.165467],
    192: [0.213677, 0.200654, 0.171337, 0.200654, 0.213677],
    256: [0.174261, 0.203105, 0.245269, 0.203105, 0.174261]
}


def calc_hamming_weight(num: int) -> int:
    """计算整数二进制汉明重量（1的个数）"""
    return bin(num).count("1")


def random_nbit(bit_len: int) -> int:
    """生成随机n比特整数，用于构造测试明文、密钥"""
    return random.getrandbits(bit_len)


def get_split_bounds(block_n: int) -> list:
    """获取对应分组长度的统计区间分界"""
    if block_n == 64:
        return [28, 30, 33, 35, 64]
    elif block_n == 80:
        return [35, 38, 41, 44, 80]
    elif block_n == 128:
        return [58, 62, 65, 69, 128]
    elif block_n == 192:
        return [90, 94, 97, 101, 192]
    elif block_n == 256:
        return [120, 125, 130, 135, 256]
    else:
        raise ValueError("仅支持分组长度：64、80、128、192、256")


def get_bin_idx(weight: int, bounds: list) -> int:
    """将汉明重量划分至对应统计区间，返回区间编号0-4"""
    b1, b2, b3, b4, _ = bounds
    if weight <= b1:
        return 0
    elif weight <= b2:
        return 1
    elif weight <= b3:
        return 2
    elif weight <= b4:
        return 3
    return 4


def chi_square_check(weight_list: list[int], block_len: int, total_sample: int) -> dict:
    """通用卡方拟合优度检验，返回标准化统计结果"""
    bounds = get_split_bounds(block_len)
    prob = GROUP_THEO_PROB[block_len]
    obs_freq = [0, 0, 0, 0, 0]
    for w in weight_list:
        idx = get_bin_idx(w, bounds)
        obs_freq[idx] += 1
    prob_np = np.array(prob)
    obs_np = np.array(obs_freq)
    exp_np = total_sample * prob_np
    chi2 = np.sum(np.square(obs_np - exp_np) / exp_np)
    p_val = gammaincc(2, chi2 / 2)
    pass_flag = p_val >= 0.01
    return {
        "chi2_observed": float(chi2),
        "p_value": float(p_val),
        "pass": pass_flag,
        "obs_frequency": obs_freq,
        "exp_frequency": exp_np.tolist()
    }


def _load_encrypt_func(encrypt_code: str):
    """
    从用户提供的Python代码字符串中动态加载加密函数。
    要求用户代码中定义一个签名为 encrypt(key: int, plain: int) -> int 的函数。
    """
    namespace = {}
    try:
        exec(encrypt_code, namespace)
    except SyntaxError as e:
        raise ValueError(f"加密算法代码存在语法错误: {e}")
    # 查找加密函数：优先找名为 encrypt 的，否则查找第一个接受2个参数的函数
    if "encrypt" in namespace and callable(namespace["encrypt"]):
        return namespace["encrypt"]
    for name, obj in namespace.items():
        if name.startswith("_"):
            continue
        if callable(obj) and not isinstance(obj, type):
            import inspect
            try:
                sig = inspect.signature(obj)
                if len(sig.parameters) == 2:
                    return obj
            except (ValueError, TypeError):
                continue
    raise ValueError(
        "未在代码中找到符合签名的加密函数。"
        "请确保代码中定义了 encrypt(key: int, plain: int) -> int 函数，"
        "或者至少有一个接受2个参数的加密函数。"
    )


# ====================== 结构性测试工具合集 ======================


@tool
def test_plain_cipher_independence(
    encrypt_code: str,
    block_size: int = 128,
    sample_m: int | None = None,
    sample_scale: str = "demo"
) -> dict:
    """
    Plaintext-Ciphertext Independence Test for Block Ciphers.
    分组密码明密文独立性检测，检验明文与密文是否存在统计依赖（混淆能力评估）。

    Args:
        encrypt_code: 加密算法Python代码字符串，需定义 encrypt(key:int, plain:int) -> cipher:int 函数
        block_size: 分组比特长度，支持 64 / 80 / 128 / 192 / 256，默认 128
        sample_m: 明文采样数量（可选，设置后覆盖 sample_scale）
        sample_scale: 采样规模预设："demo"（快速，2^14样本）或 "formal"（标准高精度，2^20样本），默认 "demo"

    Returns:
        dict: 检测结果，包含 test_name / desc / chi2_observed / p_value / pass / obs_frequency / exp_frequency / conclusion
    """
    if block_size not in GROUP_THEO_PROB:
        return {"error": f"不支持的分组长度 {block_size}，仅支持: {list(GROUP_THEO_PROB.keys())}"}
    if sample_m is None:
        sample_m = 2**20 if sample_scale == "formal" else 2**14
    try:
        encrypt_func = _load_encrypt_func(encrypt_code)
    except ValueError as e:
        return {"error": str(e)}
    test_key = random_nbit(block_size)
    weight_collect = []
    for _ in range(sample_m):
        p = random_nbit(block_size)
        try:
            c = encrypt_func(test_key, p)
        except Exception as e:
            return {"error": f"加密函数执行失败: {e}"}
        diff = p ^ c
        weight_collect.append(calc_hamming_weight(diff))
    res = chi_square_check(weight_collect, block_size, sample_m)
    res["test_name"] = "明密文独立性测试"
    res["desc"] = "检验明文与密文是否存在统计依赖"
    res["sample_count"] = sample_m
    if res["pass"]:
        res["conclusion"] = "测试通过，明密文无明显统计关联，混淆能力达标"
    else:
        res["conclusion"] = "测试不通过，明密文存在统计依赖，混淆能力不足"
    return res


@tool
def test_plain_avalanche(
    encrypt_code: str,
    block_size: int = 128,
    repeat_m: int | None = None,
    sample_scale: str = "demo"
) -> dict:
    """
    Plaintext Diffusion Avalanche Test for Block Ciphers.
    分组密码明文扩散雪崩检测，检验单比特明文变化能否充分扩散至密文所有比特（扩散能力评估）。

    Args:
        encrypt_code: 加密算法Python代码字符串，需定义 encrypt(key:int, plain:int) -> cipher:int 函数
        block_size: 分组比特长度，支持 64 / 80 / 128 / 192 / 256，默认 128
        repeat_m: 重复实验轮数（可选，设置后覆盖 sample_scale）
        sample_scale: 采样规模预设："demo"（快速，2^9轮）或 "formal"（标准高精度，2^20轮），默认 "demo"

    Returns:
        dict: 检测结果，包含 test_name / desc / chi2_observed / p_value / pass / obs_frequency / exp_frequency / conclusion
    """
    if block_size not in GROUP_THEO_PROB:
        return {"error": f"不支持的分组长度 {block_size}，仅支持: {list(GROUP_THEO_PROB.keys())}"}
    if repeat_m is None:
        repeat_m = 2**20 if sample_scale == "formal" else 2**9
    try:
        encrypt_func = _load_encrypt_func(encrypt_code)
    except ValueError as e:
        return {"error": str(e)}
    test_key = random_nbit(block_size)
    weight_collect = []
    for _ in range(repeat_m):
        p_ori = random_nbit(block_size)
        try:
            c_ori = encrypt_func(test_key, p_ori)
        except Exception as e:
            return {"error": f"加密函数执行失败: {e}"}
        for bit in range(block_size):
            mask = 1 << bit
            p_flip = p_ori ^ mask
            try:
                c_flip = encrypt_func(test_key, p_flip)
            except Exception as e:
                return {"error": f"加密函数执行失败: {e}"}
            diff = c_ori ^ c_flip
            weight_collect.append(calc_hamming_weight(diff))
    total = repeat_m * block_size
    res = chi_square_check(weight_collect, block_size, total)
    res["test_name"] = "明文扩散雪崩测试"
    res["desc"] = "检验单比特明文变化能否充分扩散至密文"
    res["sample_count"] = total
    res["repeat_rounds"] = repeat_m
    if res["pass"]:
        res["conclusion"] = "测试通过，明文扩散特性满足雪崩准则"
    else:
        res["conclusion"] = "测试不通过，明文扩散能力不足，不满足雪崩准则"
    return res


@tool
def test_key_avalanche(
    encrypt_code: str,
    block_size: int = 128,
    key_length: int = 128,
    repeat_m: int | None = None,
    sample_scale: str = "demo"
) -> dict:
    """
    Key Avalanche Effectiveness Test for Block Ciphers.
    分组密码密钥雪崩有效性检测，检验单比特密钥变化能否充分改变密文（密钥敏感度评估）。

    Args:
        encrypt_code: 加密算法Python代码字符串，需定义 encrypt(key:int, plain:int) -> cipher:int 函数
        block_size: 分组比特长度，支持 64 / 80 / 128 / 192 / 256，默认 128
        key_length: 密钥比特长度，需与加密算法匹配
        repeat_m: 重复实验轮数（可选，设置后覆盖 sample_scale）
        sample_scale: 采样规模预设："demo"（快速，2^9轮）或 "formal"（标准高精度，2^20轮），默认 "demo"

    Returns:
        dict: 检测结果，包含 test_name / desc / chi2_observed / p_value / pass / obs_frequency / exp_frequency / conclusion
    """
    if block_size not in GROUP_THEO_PROB:
        return {"error": f"不支持的分组长度 {block_size}，仅支持: {list(GROUP_THEO_PROB.keys())}"}
    if repeat_m is None:
        repeat_m = 2**20 if sample_scale == "formal" else 2**9
    try:
        encrypt_func = _load_encrypt_func(encrypt_code)
    except ValueError as e:
        return {"error": str(e)}
    weight_collect = []
    for _ in range(repeat_m):
        p_fix = random_nbit(block_size)
        k_ori = random_nbit(key_length)
        try:
            c_ori = encrypt_func(k_ori, p_fix)
        except Exception as e:
            return {"error": f"加密函数执行失败: {e}"}
        for bit in range(key_length):
            mask = 1 << bit
            k_flip = k_ori ^ mask
            try:
                c_flip = encrypt_func(k_flip, p_fix)
            except Exception as e:
                return {"error": f"加密函数执行失败: {e}"}
            diff = c_ori ^ c_flip
            weight_collect.append(calc_hamming_weight(diff))
    total = repeat_m * key_length
    res = chi_square_check(weight_collect, block_size, total)
    res["test_name"] = "密钥雪崩有效性测试"
    res["desc"] = "检验单比特密钥变化能否充分改变密文"
    res["sample_count"] = total
    res["repeat_rounds"] = repeat_m
    res["key_length"] = key_length
    if res["pass"]:
        res["conclusion"] = "测试通过，密钥敏感度达标"
    else:
        res["conclusion"] = "测试不通过，密钥敏感度不足，存在密钥冗余风险"
    return res


@tool
def run_all_struct_tests(
    encrypt_code: str,
    block_size: int = 128,
    key_length: int = 128,
    sample_scale: str = "demo"
) -> list[dict]:
    """
    Run All Three Structural Tests for Block Ciphers (one-click entry).
    一键执行分组密码全部三项结构性测试：明密文独立性 + 明文扩散雪崩 + 密钥雪崩有效性。

    Args:
        encrypt_code: 加密算法Python代码字符串，需定义 encrypt(key:int, plain:int) -> cipher:int 函数
        block_size: 分组比特长度，支持 64 / 80 / 128 / 192 / 256，默认 128
        key_length: 密钥比特长度，需与加密算法匹配，默认 128
        sample_scale: 采样规模预设："demo"（快速，默认）或 "formal"（标准高精度，样本量 2^20，耗时较长）

    Returns:
        list[dict]: 三项测试结果顺序列表，每项字段与单项工具一致
    """
    if sample_scale == "formal":
        m_indep = 2**20
        m_avalanche = 2**20
    else:
        m_indep = 2**14
        m_avalanche = 2**9
    try:
        encrypt_func = _load_encrypt_func(encrypt_code)
    except ValueError as e:
        return [{"error": str(e)}]

    # 测试1：明密文独立性
    test_key = random_nbit(block_size)
    weight_collect = []
    for _ in range(m_indep):
        p = random_nbit(block_size)
        try:
            c = encrypt_func(test_key, p)
        except Exception as e:
            return [{"error": f"[明密文独立性] 加密函数执行失败: {e}"}]
        diff = p ^ c
        weight_collect.append(calc_hamming_weight(diff))
    res1 = chi_square_check(weight_collect, block_size, m_indep)
    res1["test_name"] = "明密文独立性测试"
    res1["desc"] = "检验明文与密文是否存在统计依赖"
    res1["sample_count"] = m_indep
    if res1["pass"]:
        res1["conclusion"] = "测试通过，明密文无明显统计关联，混淆能力达标"
    else:
        res1["conclusion"] = "测试不通过，明密文存在统计依赖，混淆能力不足"

    # 测试2：明文扩散雪崩
    test_key2 = random_nbit(block_size)
    weight_collect2 = []
    for _ in range(m_avalanche):
        p_ori = random_nbit(block_size)
        try:
            c_ori = encrypt_func(test_key2, p_ori)
        except Exception as e:
            return [{"error": f"[明文扩散雪崩] 加密函数执行失败: {e}"}]
        for bit in range(block_size):
            mask = 1 << bit
            p_flip = p_ori ^ mask
            try:
                c_flip = encrypt_func(test_key2, p_flip)
            except Exception as e:
                return [{"error": f"[明文扩散雪崩] 加密函数执行失败: {e}"}]
            diff = c_ori ^ c_flip
            weight_collect2.append(calc_hamming_weight(diff))
    total2 = m_avalanche * block_size
    res2 = chi_square_check(weight_collect2, block_size, total2)
    res2["test_name"] = "明文扩散雪崩测试"
    res2["desc"] = "检验单比特明文变化能否充分扩散至密文"
    res2["sample_count"] = total2
    res2["repeat_rounds"] = m_avalanche
    if res2["pass"]:
        res2["conclusion"] = "测试通过，明文扩散特性满足雪崩准则"
    else:
        res2["conclusion"] = "测试不通过，明文扩散能力不足，不满足雪崩准则"

    # 测试3：密钥雪崩有效性
    weight_collect3 = []
    for _ in range(m_avalanche):
        p_fix = random_nbit(block_size)
        k_ori = random_nbit(key_length)
        try:
            c_ori = encrypt_func(k_ori, p_fix)
        except Exception as e:
            return [{"error": f"[密钥雪崩有效性] 加密函数执行失败: {e}"}]
        for bit in range(key_length):
            mask = 1 << bit
            k_flip = k_ori ^ mask
            try:
                c_flip = encrypt_func(k_flip, p_fix)
            except Exception as e:
                return [{"error": f"[密钥雪崩有效性] 加密函数执行失败: {e}"}]
            diff = c_ori ^ c_flip
            weight_collect3.append(calc_hamming_weight(diff))
    total3 = m_avalanche * key_length
    res3 = chi_square_check(weight_collect3, block_size, total3)
    res3["test_name"] = "密钥雪崩有效性测试"
    res3["desc"] = "检验单比特密钥变化能否充分改变密文"
    res3["sample_count"] = total3
    res3["repeat_rounds"] = m_avalanche
    res3["key_length"] = key_length
    if res3["pass"]:
        res3["conclusion"] = "测试通过，密钥敏感度达标"
    else:
        res3["conclusion"] = "测试不通过，密钥敏感度不足，存在密钥冗余风险"

    return [res1, res2, res3]


# 工具列表导出（供 agent.py 注册）
structural_tools = [
    test_plain_cipher_independence,
    test_plain_avalanche,
    test_key_avalanche,
    run_all_struct_tests,
]
