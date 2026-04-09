

import baostock as bs
import pandas as pd
import mplfinance as mpf


def fetch_baostock_data(code, start_date, end_date):
    """
    使用 baostock 获取历史 A 股 K 线数据
    """
    # 登录 baostock
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return None

    # 获取历史K线数据 (复权状态: 3为后复权)
    print(f"正在获取 {code} 的数据...")
    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"
    )

    if rs.error_code != '0':
        print(f"获取数据失败: {rs.error_msg}")
        bs.logout()
        return None

    # 将数据存入列表
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    # 登出 baostock
    bs.logout()

    # 转换为 pandas DataFrame
    df = pd.DataFrame(data_list, columns=rs.fields)

    if df.empty:
        print("未获取到数据，请检查股票代码或日期范围。")
        return df

    # 数据格式清洗：mplfinance 需要 float 类型的 OHLCV 数据和 DatetimeIndex
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    return df


def plot_renko_chart(df, title="Renko Chart"):
    """
    使用 mplfinance 绘制砖图
    """
    if df is None or df.empty:
        return

    # 配置砖图参数
    # brick_size='atr' 表示使用平均真实波动幅度(Average True Range)自动计算砖块大小
    # 也可以传入具体的数值，例如 brick_size=0.5 (表示每波动 0.5 元画一块砖)
    renko_params = dict(brick_size='atr', atr_length=14)

    # 绘制图表
    mpf.plot(
        df,
        type='renko',
        renko_params=renko_params,
        style='yahoo',           # 图表风格：yahoo, charles, binance 等
        title=title,
        ylabel='Price',
        volume=False             # 砖图通常不配合成交量看，因为时间轴是非线性的
    )


if __name__ == '__main__':
    # 设置参数：股票代码，开始日期，结束日期
    stock_code = "sh.600000"  # 浦发银行
    start_date = "2023-01-01"
    end_date = "2024-01-01"

    # 1. 获取数据
    df_stock = fetch_baostock_data(stock_code, start_date, end_date)

    # 2. 绘制砖图
    if df_stock is not None and not df_stock.empty:
        plot_renko_chart(df_stock, title=f"Renko Chart - {stock_code}")
