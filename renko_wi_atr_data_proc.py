import baostock as bs
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import os
from datetime import datetime, timedelta

# ==========================================
# 【全局核心参数区】
# ==========================================
ATR_MULTIPLIER = 0.9  # ATR 乘数
TEST_PERIODS = [14,]  # 需要遍历测试的 ATR 周期列表


# 惠泰医疗 688617
# 联影医疗 688271
# 洛阳钼业 603993

STOCK_CODE = "sh.603993"
TARGET_K_COUNT = 250  # 需要获取/展示的目标 K 线数量
END_DATE = None       # 截止日期。设为 None 则自动获取真实时间的今天


# ==========================================
# 【时间推算引擎】
# ==========================================
def calculate_start_date(end_date_str, trading_days_needed):
    """根据截止日期和所需交易日，自动推算包含节假日的起始日期"""
    end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d')
    # 乘以 1.46 (自然日/交易日比) + 30 天超额安全冗余
    calendar_days_to_subtract = int(trading_days_needed * 1.46) + 30
    start_date_obj = end_date_obj - timedelta(days=calendar_days_to_subtract)
    return start_date_obj.strftime('%Y-%m-%d')


def fetch_data_auto(code, end_date=None, target_k_count=120, max_indicator_period=20):
    """获取并清洗 A股 K线数据（自动包含指标预热期）"""
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # 实际需要拉取的数据量 = 目标展示量 + 最大指标周期预热
    total_needed = target_k_count + max_indicator_period
    start_date = calculate_start_date(end_date, total_needed)

    print("-" * 40)
    print(f"目标请求: {code} 的最后 {target_k_count} 根有效 K线")
    print(f"时间引擎分配: 从 {start_date} 到 {end_date} (已包含指标预热与节假日冗余)")
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

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    return df


if __name__ == '__main__':

    # 1. 在循环外部获取一次底层数据，避免重复请求 API 浪费时间
    print(f"正在从 Baostock 拉取 {STOCK_CODE} 的底层数据...")
    # 传入 max(TEST_PERIODS) 保证底层数据量足够应付最大的 20日 ATR 计算耗损
    df_raw = fetch_data_auto(STOCK_CODE, END_DATE, TARGET_K_COUNT, max(TEST_PERIODS))

    if not df_raw.empty:
        # 创建一个文件夹用于存放批量生成的图表
        output_dir = "Renko_Backtest_Charts"
        os.makedirs(output_dir, exist_ok=True)

        print("-" * 40)

        # 2. 开启核心参数遍历循环
        for current_period in TEST_PERIODS:
            print(f"正在运算 {current_period}日 ATR 模型...")

            # 【致命错误防御】：必须复制原始数据，不能污染底层底座
            df = df_raw.copy()

            # 动态计算 ATR
            df.ta.atr(length=current_period, append=True)
            df.dropna(inplace=True)

            # 【核心对齐逻辑】：确保每张图表展示的时间窗口严格一致
            # 指标预热计算完毕并剔除 NaN 后，精确截取最后的 TARGET_K_COUNT 根
            if len(df) > TARGET_K_COUNT:
                df = df.tail(TARGET_K_COUNT)

            # 动态提取数值
            col_name = f'ATRr_{current_period}'
            latest_atr = df[col_name].iloc[-1]
            custom_brick_size = round(latest_atr * ATR_MULTIPLIER, 2)

            # 3. 动态配置图表信息
            chart_title = (
                f"Renko Chart - {STOCK_CODE}\n"
                f"(Brick = {custom_brick_size} | {current_period}-Day ATR x {ATR_MULTIPLIER} | Last {len(df)} Bars)"
            )

            # 利用 f-string 动态生成带有参数标签的文件名
            file_name = f"{output_dir}/Renko_{STOCK_CODE}_ATR{current_period}.png"

            # 4. 绘图并输出到文件
            mpf.plot(
                df,
                type='renko',
                renko_params=dict(brick_size=custom_brick_size),
                style='yahoo',
                title=chart_title,
                ylabel='Price',
                savefig=file_name  # 静默高速输出
            )

            print(f"  -> 已生成图表: {file_name} (砖块大小: {custom_brick_size})")

        print("-" * 40)
        print(f"批量回测完成！所有对比图表已保存至所在目录下的 '{output_dir}' 文件夹中。")
    else:
        print("警告：未能获取到底层数据。")