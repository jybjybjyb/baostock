# 利用 pandas-ta 计算出历史 ATR 序列，提取最后一个交易日的 ATR 数值，将其作为固定参数传递给砖图绘制函数。
import baostock as bs
import pandas as pd
import pandas_ta as ta

import baostock as bs
import pandas as pd
import pandas_ta as ta  # 导入 pandas_ta


def calculate_atr_with_pandas_ta(stock_code="sh.600000", start_date="2023-01-01", end_date="2024-01-01"):
    # 1. 登录 baostock
    bs.login()

    # 2. 获取历史K线数据 (复权状态: 3为后复权)
    # pandas-ta 计算 ATR 只需要 high, low, close
    rs = bs.query_history_k_data_plus(
        stock_code,
        "date,high,low,close",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"
    )

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    bs.logout()

    df = pd.DataFrame(data_list, columns=rs.fields)
    if df.empty:
        return df

    # 3. 数据清洗：转换为浮点数并将日期设为索引
    # pandas-ta 非常依赖正确的数据类型
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # ==========================================
    # 4. 核心：使用 pandas-ta 一行代码计算 ATR
    # append=True 表示将计算结果直接作为新列添加到原 DataFrame 中
    # 默认列名会自动命名为 'ATRr_14' (r 代表使用了 RMA 平滑算法)
    # ==========================================
    df.ta.atr(length=14, append=True)

    return df


if __name__ == '__main__':
    # 获取浦发银行的 ATR 数据
    df_result = calculate_atr_with_pandas_ta(
        "sh.600000", "2026-01-01", "2026-04-08")

    if df_result is not None and not df_result.empty:
        # 打印最后5行数据，你会在最后看到自动生成的 'ATRr_14' 列
        print(df_result[['close', 'ATRr_14']].tail())
