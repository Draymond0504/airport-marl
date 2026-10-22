import pandas as pd
import math
from typing import List, Dict, Any
import datetime

# --- 物理时间偏移配置 ---
# 进港航班：降落后经过外围滑行道，延迟到达局部网络入口
OFFSET_ARR_SEC = 5 * 60  # +5分钟
# 离港航班：为了赶上起飞时间，需提前推出机位并经过局部网络
OFFSET_DEP_SEC = 10 * 60 # -10分钟
# ------------------------

# 经过用户校正的准确字典
STAND_TO_NODE_MAP = {
    # 远西斜线及垂直段
    '128': 0, '129': 1, '130': 2, '131': 3,
    '125': 4, '126': 5, '127': 6,
    
    # 港湾区左下横线
    '124': 7, '123': 8, '122': 9,
    
    # 港湾区(中心U型)
    '121': 10, '120': 11, '119': 12, '118': 13, # 左竖
    '116': 14, '115': 15, '114': 16, '113': 17, # 顶横
    '109': 18, '110': 19, '111': 20, '112': 21, # 右竖
    '108': 22, 
    
    # 远东及外侧横向
    '107': 23,
    '106': 24, '105': 25, '104': 26, '103': 27, '102': 28, '101': 29
}

def load_cases_from_excel(
    filepath: str, 
    chunk_size: int = 15,
    time_scale: float = 5.0 # 每5秒对应一个env_step
) -> List[Dict[str, Any]]:
    """适配禄口真实格式: 一行包含进港(A)和离港(D)信息"""
    print(f"[DataLoader] 正在解析禄口计划表: {filepath} ...")
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"[DataLoader] 读取 Excel 失败: {e}。")
        return []

    # 识别列名 (映射用户提供的表头)
    # 进港时间优先用 ALDT (实际落地)，离港时间优先用 ATOT (实际推出/起飞)
    col_aldt = "ALDT"
    col_atot = "ATOT"
    col_gate_a = "机位(A)"
    col_gate_d = "机位(D)"
    
    # 降级备选
    if col_aldt not in df.columns: col_aldt = "ELDT"
    if col_atot not in df.columns: col_atot = "ETOT"
    
    flights_data = []

    for _, row in df.iterrows():
        # 1. 处理进港子任务 (Inbound)
        gate_a = str(row.get(col_gate_a, "")).strip()
        time_a = row.get(col_aldt)
        
        if pd.notna(time_a) and gate_a:
            # 提取数字机位
            import re
            m = re.search(r'\d+', gate_a)
            if m and m.group() in STAND_TO_NODE_MAP:
                # 加上进港延后偏移
                if hasattr(time_a, 'to_pydatetime'):
                    time_a = time_a.to_pydatetime()
                if isinstance(time_a, datetime.datetime):
                    time_a = time_a + datetime.timedelta(seconds=OFFSET_ARR_SEC)
                    
                flights_data.append({
                    "time": time_a,
                    "start": 100, # 跑道入口
                    "goal": STAND_TO_NODE_MAP[m.group()],
                    "desc": f"进港 {row.get('航班号(A)', '未知')}"
                })

        # 2. 处理离港子任务 (Outbound)
        gate_d = str(row.get(col_gate_d, "")).strip()
        time_d = row.get(col_atot)
        
        # 如果机位(D)为空，尝试复用机位(A) (通常飞机停在同一个位子)
        if not gate_d or gate_d == "nan":
            gate_d = gate_a

        if pd.notna(time_d) and gate_d:
            import re
            m = re.search(r'\d+', gate_d)
            if m and m.group() in STAND_TO_NODE_MAP:
                # 减去离港提前偏移
                if hasattr(time_d, 'to_pydatetime'):
                    time_d = time_d.to_pydatetime()
                if isinstance(time_d, datetime.datetime):
                    time_d = time_d - datetime.timedelta(seconds=OFFSET_DEP_SEC)
                    
                flights_data.append({
                    "time": time_d,
                    "start": STAND_TO_NODE_MAP[m.group()],
                    "goal": 105, # 跑道出口
                    "desc": f"离港 {row.get('航班号(D)', '未知')}"
                })

    if not flights_data:
        print("[DataLoader] 未能在表中找到有效的航班时间或机位列。")
        return []

    # 按时间排序所有独立动作
    flights_data.sort(key=lambda x: x["time"])
    
    # 转换时间为相对步数
    base_time = flights_data[0]["time"]
    
    starts, goals, t0_delays, descs = [], [], [], []
    for f in flights_data:
        starts.append(f["start"])
        goals.append(f["goal"])
        descs.append(f["desc"])
        if hasattr(f["time"], 'timestamp'):
            diff = (f["time"] - base_time).total_seconds()
        else:
            diff = 0
        t0_delays.append(int(diff / time_scale))

    print(f"[DataLoader] 成功提取并行轨迹任务数: {len(starts)}")

    # 分段成 cases
    cases = []
    for i in range(0, len(starts), chunk_size):
        sub_starts = starts[i:i+chunk_size]
        sub_goals = goals[i:i+chunk_size]
        sub_t0 = t0_delays[i:i+chunk_size]
        
        base_t = sub_t0[0]
        sub_t0 = [t - base_t for t in sub_t0]
        
        cases.append({
            "starts": sub_starts,
            "goals": sub_goals,
            "t0": sub_t0,
            "desc": f"实况片段 {i//chunk_size+1} ({descs[i]} 起)"
        })
        
    return cases
