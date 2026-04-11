import baostock as bs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta

# ==========================================
# 模块一：自动化时间推算引擎
# ==========================================


def calculate_start_date(end_date_str, trading_days_needed):
    """根据截止日期和需要的交易日数量，自动推算安全的起始自然日"""
    end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d')
    # 交易日转自然日膨胀换算 (*1.46)，并增加 20 天节假日安全余量
    calendar_days_to_subtract = int(trading_days_needed * 1.46) + 20
    start_date_obj = end_date_obj - timedelta(days=calendar_days_to_subtract)
    return start_date_obj.strftime('%Y-%m-%d')

# ==========================================
# 模块二：自动化分钟级高频数据获取
# ==========================================


def fetch_minute_data_auto(code="sh.688271", end_date=None, target_days=20, freq="5"):
    """
    全自动获取分钟级数据
    :param target_days: 需要获取的最近 N 个交易日的数据
    """
    # 1. 自动生成时间范围
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    start_date = calculate_start_date(end_date, target_days)

    print("-" * 40)
    print(f"目标请求: {code} 最近 {target_days} 个交易日的 {freq}分钟 K线")
    print(f"时间引擎分配: 从 {start_date} 到 {end_date} (已包含安全余量)")
    print("-" * 40)

    # 2. 拉取数据
    bs.login()
    rs = bs.query_history_k_data_plus(
        code,
        "date,time,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency=freq,
        adjustflag="3"
    )

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(data_list, columns=rs.fields)
    if df.empty:
        print("警告：获取到的数据为空，请检查代码或网络。")
        return df

    # 3. 数据清洗与规范化
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    df['time'] = pd.to_datetime(
        df['time'].str.slice(0, 14), format='%Y%m%d%H%M%S')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
              'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df = df[df['Volume'] > 0]

    # 4. 尾部精确截断：将“交易日”转换为“K线根数”
    # 5分钟线一天48根，15分钟线一天16根，30分钟一天8根，60分钟一天4根
    bars_per_day = {"5": 48, "15": 16, "30": 8, "60": 4}
    target_k_count = target_days * bars_per_day.get(str(freq), 48)

    if len(df) > target_k_count:
        df = df.tail(target_k_count)

    print(f"数据清洗截断完毕，最终实际参与计算的 K线数量: {len(df)} 根")
    return df

# ==========================================
# 模块三：微观结构可视化 (Volume Profile)
# ==========================================


def plot_high_res_volume_profile(df, stock_code, bins=100):
    """基于分钟线绘制高精度 Volume Profile"""
    price_min = df['Low'].min()
    price_max = df['High'].max()

    price_bins = np.linspace(price_min, price_max, bins)

    df['Micro_Typical_Price'] = (df['High'] + df['Low']) / 2
    vprofile = df.groupby(pd.cut(
        df['Micro_Typical_Price'], bins=price_bins, include_lowest=True))['Volume'].sum()

    poc_idx = vprofile.idxmax()
    poc_price = (poc_idx.left + poc_idx.right) / 2

    fig = mpf.figure(style='yahoo', figsize=(14, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twiny()

    mpf.plot(df, type='candle', ax=ax1, volume=False, show_nontrading=False)

    bin_centers = [(b.left + b.right)/2 for b in vprofile.index]
    ax2.barh(bin_centers, vprofile.values, height=(price_max-price_min)/bins * 0.9,
             alpha=0.4, color='dodgerblue', align='center', edgecolor='none')

    ax1.axhline(poc_price, color='red', linestyle='-', linewidth=2, alpha=0.8,
                label=f'High-Res POC: {poc_price:.2f}')

    ax2.set_xticks([])
    ax1.legend(loc='upper left')

    # 动态标题，直接显示当前拉取的数据量
    k_count = len(df)
    chart_title = f"High-Res Volume Profile - {stock_code}\n(Based on {k_count} recent 5-Min bars)"
    ax1.set_title(chart_title)

    plt.show()


# ==========================================
# 实战执行区
# ==========================================
if __name__ == '__main__':
    
    # 惠泰医疗 688617
    # 联影医疗 688271
    # 洛阳钼业 603993 
    # 百济神州U 603235
    # 三一重工 600031

    stock_code = "sh.603235"

    # 你只需要告诉它你要看最近多少个交易日（比如最近 20 天，约一个月）
    # 它会自动推算时间、自动剔除节假日、自动精确截断所需 K 线
    df_min = fetch_minute_data_auto(
        code=stock_code, end_date=None, target_days=120, freq="5")

    if not df_min.empty:
        plot_high_res_volume_profile(df_min, stock_code, bins=80)
    else:
        print("未获取到数据，请检查网络或交易状态。")
