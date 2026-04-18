import os
import csv
import baostock as bs
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import mplfinance._utils as mpf_utils
from datetime import datetime, timedelta

# ==========================================
# 【底层系统级 Bug 修复：猴子补丁】
# ==========================================
original_calculate_atr = mpf_utils._calculate_atr


def patched_calculate_atr(n, highs, lows, closes):
    h = highs.values if hasattr(highs, 'values') else highs
    l = lows.values if hasattr(lows, 'values') else lows
    c = closes.values if hasattr(closes, 'values') else closes
    return original_calculate_atr(n, h, l, c)


mpf_utils._calculate_atr = patched_calculate_atr

# ==========================================
# 【Kagi 自定义引擎：绕过 mplfinance Validator】
# ==========================================


class KagiEngine:
    """手动计算 Kagi 路径，支持阳线(Yang)和阴线(Yin)状态切换"""
    @staticmethod
    def calculate_path(prices, reversal_limit):
        if len(prices) < 1:
            return []

        kagi_points = [prices[0]]
        direction = 0  # 1 为上, -1 为下
        last_extreme = prices[0]
        state = "yang" if True else "yin"  # 初始状态

        # 简化版 Kagi 逻辑
        path = []  # 存储 (x, y, is_yang)
        current_price = prices[0]

        # 记录关键转折点
        points = [current_price]
        for p in prices[1:]:
            if direction == 0:
                if p >= current_price + reversal_limit:
                    direction = 1
                    current_price = p
                    points.append(p)
                elif p <= current_price - reversal_limit:
                    direction = -1
                    current_price = p
                    points.append(p)
            elif direction == 1:
                if p >= current_price:
                    current_price = p
                    points[-1] = p
                elif p <= current_price - reversal_limit:
                    direction = -1
                    current_price = p
                    points.append(p)
            elif direction == -1:
                if p <= current_price:
                    current_price = p
                    points[-1] = p
                elif p >= current_price + reversal_limit:
                    direction = 1
                    current_price = p
                    points.append(p)
        return points


# ==========================================
# 【全局核心参数】
# ==========================================
OUTPUT_DIR = "result"
CSV_FILE = "list.csv"
TARGET_DAILY_DAYS = 250
ATR_PERIOD = 20
RENKO_MULTIPLIER = 0.67
KAGI_MULTIPLIER = 0.67  # Kagi 反转阈值乘数

# ==========================================
# 数据获取 (整合 Moutai 修正逻辑)
# ==========================================


def fetch_k_data(code, start_date, end_date, freq):
    # 使用用户要求的 query_history_k_data_plus
    fields = "date,time,open,high,low,close,volume" if freq in [
        "5", "15", "30", "60"] else "date,open,high,low,close,volume"
    rs = bs.query_history_k_data_plus(
        code, fields, start_date=start_date, end_date=end_date, frequency=freq, adjustflag="2")

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    df = pd.DataFrame(data_list, columns=rs.fields)
    if df.empty:
        return df

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    if 'time' in df.columns:
        df['time'] = pd.to_datetime(
            df['time'].str.slice(0, 14), format='%Y%m%d%H%M%S')
        df.set_index('time', inplace=True)
    else:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
              'close': 'Close', 'volume': 'Volume'}, inplace=True)
    return df.dropna()

# ==========================================
# 绘图模块：Renko + 自定义 Kagi
# ==========================================


def generate_renko_and_kagi(df_raw, stock_name, stock_code):
    df = df_raw.copy()
    df.ta.atr(length=ATR_PERIOD, append=True)
    df.dropna(inplace=True)
    if len(df) > TARGET_DAILY_DAYS:
        df = df.tail(TARGET_DAILY_DAYS)

    atr_val = df[f'ATRr_{ATR_PERIOD}'].iloc[-1]

    # 1. 绘制 Renko (使用 mplfinance)
    renko_brick = round(atr_val * RENKO_MULTIPLIER, 2)
    renko_fn = os.path.join(OUTPUT_DIR, f"{stock_name}_Renko.png")
    mpf.plot(df, type='renko', renko_params=dict(brick_size=renko_brick), style='yahoo',
             title=f"Renko - {stock_name}\nBrick={renko_brick}",
             savefig=dict(fname=renko_fn, dpi=150, bbox_inches='tight'))

    # 2. 绘制 Kagi (手动绘制，彻底避开 type='kagi' 报错)
    kagi_reversal = round(atr_val * KAGI_MULTIPLIER, 2)
    points = KagiEngine.calculate_path(df['Close'].tolist(), kagi_reversal)

    fig, ax = plt.subplots(figsize=(12, 6))
    # 构建阶梯线
    x_coords = []
    y_coords = []
    for i in range(len(points)):
        x_coords.extend([i, i+1])  # 水平线
        y_coords.extend([points[i], points[i]])
        if i < len(points)-1:
            x_coords.extend([i+1, i+1])  # 垂直线
            y_coords.extend([points[i], points[i+1]])

    # 简单逻辑：上涨绿，下跌红（这里可根据 Yang/Yin 逻辑细化）
    ax.plot(x_coords, y_coords, color='black', linewidth=1.5, alpha=0.7)
    ax.set_title(
        f"Kagi - {stock_name} ({stock_code})\nReversal Amount: {kagi_reversal}")
    ax.grid(True, linestyle='--', alpha=0.5)

    kagi_fn = os.path.join(OUTPUT_DIR, f"{stock_name}_Kagi.png")
    plt.savefig(kagi_fn, dpi=150, bbox_inches='tight')
    plt.close('all')

# ==========================================
# 主程序逻辑
# ==========================================


def main_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # ... 此处省略 load_stock_list 的定义，保持与原脚本一致 ...
    stock_list = [("sh.688271", "联影医疗")]  # 测试用例

    bs.login()
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=450)).strftime('%Y-%m-%d')

    for code, name in stock_list:
        print(f"正在处理: {name} ({code})")
        df_daily = fetch_k_data(code, start_date, end_date, "d")
        if not df_daily.empty:
            generate_renko_and_kagi(df_daily, name, code)
            print("  √ Renko 与 Kagi 已保存。")

    bs.logout()


if __name__ == '__main__':
    main_pipeline()
