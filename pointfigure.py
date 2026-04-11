import baostock as bs
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import mplfinance._utils as mpf_utils
from datetime import datetime, timedelta

# ==========================================
# 【底层系统级 Bug 修复：猴子补丁】
# 解决 mplfinance 在 Pandas 2.0+ 环境下画 P&F 图的崩溃问题
# ==========================================
original_calculate_atr = mpf_utils._calculate_atr


def patched_calculate_atr(n, highs, lows, closes):
    return original_calculate_atr(n, highs.values, lows.values, closes.values)


mpf_utils._calculate_atr = patched_calculate_atr

# ==========================================
# 【量化系统核心参数控制台】
# ==========================================
ATR_PERIOD = 14        # ATR 计算周期
BOX_MULTIPLIER = 0.5   # 箱体大小乘数 (Box Size)
REVERSAL_AMOUNT = 3    # 反转系数
# ==========================================

# ==========================================
# 模块一：自动化时间推算引擎
# ==========================================


def calculate_start_date(end_date_str, trading_days_needed):
    """根据截止日期和需要的【交易日】数量，自动推算安全的起始【自然日】"""
    end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d')
    # 交易日转自然日 (*1.46) + 节假日余量 + 指标预热补偿 (ATR_PERIOD)
    calendar_days_to_subtract = int(
        (trading_days_needed + ATR_PERIOD) * 1.46) + 20
    start_date_obj = end_date_obj - timedelta(days=calendar_days_to_subtract)
    return start_date_obj.strftime('%Y-%m-%d')

# ==========================================
# 模块二：无缝接入时间引擎的 P&F 数据中枢
# ==========================================


def fetch_pnf_data_auto(code="sh.600000", end_date=None, target_days=250):
    """
    全自动数据拉取与清洗函数
    :param target_days: 你希望在图表上展现的最近 N 个交易日 (默认一年约250天)
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    start_date = calculate_start_date(end_date, target_days)

    print("-" * 40)
    print(f"目标请求: {code} 最近 {target_days} 个交易日的日线数据")
    print(f"时间引擎分配: 从 {start_date} 到 {end_date} (已包含指标预热与节假日余量)")
    print("-" * 40)

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

    # 数据类型与格式清洗
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

    # 计算 ATR 并执行空值清洗 (此时会消耗掉最前面的 ATR_PERIOD 根线)
    df.ta.atr(length=ATR_PERIOD, append=True)
    df.dropna(inplace=True)

    # 尾部精确截断：确保最终交给绘图引擎的数据，刚刚好就是用户要求的 target_days 根
    if len(df) > target_days:
        df = df.tail(target_days)

    return df


# ==========================================
# 模块三：实战执行与绘图区
# ==========================================
if __name__ == '__main__':

    # 惠泰医疗 688617
    # 联影医疗 688271
    # 洛阳钼业 603993 
    # 百济神州U 603235
    # 三一重工 600031
    
    stock_code = "sh.688271"

    # 你只需发号施令：“给我拉最近一年的数据（250个交易日）”，其余全部自动化
    df = fetch_pnf_data_auto(stock_code, end_date=None, target_days=250)

    if not df.empty:
        # 1. 提取最新 ATR 并计算动态箱体
        col_name = f'ATRr_{ATR_PERIOD}'
        raw_atr = df[col_name].iloc[-1]
        custom_box_size = round(raw_atr * BOX_MULTIPLIER, 2)

        print(f"数据清洗完毕，最终实际有效 K线数量: {len(df)} 根")
        print(f"设定箱体大小 (Box Size): {custom_box_size}")
        print(
            f"设定反转阈值 (Reversal): {REVERSAL_AMOUNT} 箱 ({custom_box_size * REVERSAL_AMOUNT:.2f} 元)")
        print("-" * 40)

        # 2. 配置 P&F 参数
        pnf_params = dict(box_size=custom_box_size, reversal=REVERSAL_AMOUNT)

        chart_title = (
            f"Point & Figure (P&F) Chart - {stock_code}\n"
            f"(Box={custom_box_size} | Reversal={REVERSAL_AMOUNT} | Base={ATR_PERIOD}d ATR)"
        )

        # 3. 绘制脱离时间轴的 P&F 图
        mpf.plot(
            df,
            type='pnf',
            pnf_params=pnf_params,
            style='yahoo',
            title=chart_title,
            ylabel='Price'
        )
    else:
        print(f"\n【数据空值熔断】未能获取到 {stock_code} 的历史行情数据！请检查标的或网络。")
