import baostock as bs
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import mplfinance._utils as mpf_utils  # 导入底层工具模块准备打补丁

# ==========================================
# 【底层系统级 Bug 修复：猴子补丁 Monkey Patch】
# 强制拦截 mplfinance 的 _calculate_atr 函数，将传入的 Pandas Series
# 转换为 numpy 数组 (.values)，恢复其数字索引 [i] 的能力。
# ==========================================
original_calculate_atr = mpf_utils._calculate_atr


def patched_calculate_atr(n, highs, lows, closes):
    return original_calculate_atr(n, highs.values, lows.values, closes.values)


mpf_utils._calculate_atr = patched_calculate_atr
# ==========================================

# ==========================================
# 【量化系统核心参数控制台】
# ==========================================
ATR_PERIOD = 14        # ATR 计算周期
BOX_MULTIPLIER = 0.5   # 箱体大小乘数 (Box Size)
REVERSAL_AMOUNT = 3    # 反转系数
# ==========================================


def fetch_data(code="sh.600000", start_date="2023-01-01", end_date="2024-01-01"):
    """获取数据"""
    bs.login()
    rs = bs.query_history_k_data_plus(
        code, "date,open,high,low,close,volume",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="3"
    )

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(data_list, columns=rs.fields)
    if df.empty:
        return df

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 转换为 mplfinance 强制要求的大写规范
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    return df


if __name__ == '__main__':
    stock_code = "sh.688271"  

    print(f"正在拉取 {stock_code} 数据...")
    df = fetch_data(stock_code, "2025-01-01", "2026-04-08")

    if not df.empty:
        # 1. 计算 ATR
        df.ta.atr(length=ATR_PERIOD, append=True)
        df.dropna(inplace=True)

        # 2. 提取最新 ATR 并计算箱体大小
        col_name = f'ATRr_{ATR_PERIOD}'
        raw_atr = df[col_name].iloc[-1]
        custom_box_size = round(raw_atr * BOX_MULTIPLIER, 2)

        print("-" * 40)
        print(f"设定箱体大小 (Box Size): {custom_box_size}")
        print(
            f"设定反转阈值 (Reversal): {REVERSAL_AMOUNT} 箱 ({custom_box_size * REVERSAL_AMOUNT:.2f} 元)")
        print("-" * 40)

        # 3. 配置 P&F 参数
        pnf_params = dict(box_size=custom_box_size, reversal=REVERSAL_AMOUNT)

        chart_title = (
            f"Point & Figure (P&F) Chart - {stock_code}\n"
            f"(Box={custom_box_size} | Reversal={REVERSAL_AMOUNT} | ATR_Base={ATR_PERIOD})"
        )

        # 4. 绘制 P&F 图
        mpf.plot(
            df,
            type='pnf',
            pnf_params=pnf_params,
            style='yahoo',
            title=chart_title,
            ylabel='Price'
        )
    else:
        print(f"\n【数据空值熔断】未能获取到 {stock_code} 的历史行情数据！")
