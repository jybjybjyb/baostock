import baostock as bs
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta


def get_zz800_stocks():
    """获取中证800成分股（沪深300 + 中证500）"""
    hs300 = bs.query_hs300_stocks().get_data()
    zz500 = bs.query_zz500_stocks().get_data()
    zz800 = pd.concat([hs300, zz500])['code'].unique().tolist()
    return zz800


def get_factor_data(stock_list, end_date):
    """提取因子数据与目标收益率"""
    # 设定时间窗口：取过去30个交易日的数据来计算因子和收益
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") -
                  timedelta(days=60)).strftime("%Y-%m-%d")

    factor_list = []
    print(f"开始获取 {len(stock_list)} 只股票的数据，这可能需要1-2分钟...")

    for code in stock_list:
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,close,pctChg,pbMRQ",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2"  # 前复权
        )
        if rs.error_code != '0':
            continue

        df = rs.get_data()
        if len(df) < 25:  # 剔除新股或停牌时间过长的股票
            continue

        df[['close', 'pctChg', 'pbMRQ']] = df[['close', 'pctChg', 'pbMRQ']].apply(
            pd.to_numeric, errors='coerce')
        df = df.dropna()
        if len(df) < 25:
            continue

        # 划分时间窗口：
        # T-25 到 T-5 用于计算因子（避免使用未来数据）
        # T-5 到 T-0 用于计算近一周的目标收益率 (Y)
        df_factor = df.iloc[-25:-5]
        df_target = df.iloc[-5:]

        # 1. 动量因子 (Momentum)：T-25 到 T-5 的区间涨幅
        momentum = df_factor['close'].iloc[-1] / df_factor['close'].iloc[0] - 1

        # 2. 低波因子 (Low Vol)：日收益率标准差的负数
        volatility = df_factor['pctChg'].std()
        low_vol = -1 * volatility if pd.notnull(volatility) else np.nan

        # 3. 价值因子 (Value)：T-5那一天的账面市值比 (1 / PB)
        pb = df_factor['pbMRQ'].iloc[-1]
        value = 1 / pb if pb > 0 else np.nan

        # 目标收益率 Y：近一周的真实涨跌幅
        target_return = df_target['close'].iloc[-1] / \
            df_target['close'].iloc[0] - 1

        factor_list.append({
            'code': code,
            'Momentum': momentum,
            'Low_Vol': low_vol,
            'Value': value,
            'Target_Return': target_return
        })

    return pd.DataFrame(factor_list).dropna()


def analyze_factor_exposure(df):
    """数据截面标准化与多元回归归因"""
    print("\n--- 开始进行截面回归与归因分析 ---")

    # 1. 极值处理 (Winsorize) 与 标准化 (Z-score)
    # 这一步极其重要，否则个别妖股会带偏整个回归方程
    factors = ['Momentum', 'Low_Vol', 'Value']
    for f in factors:
        # 去极值：限制在上下3个标准差以内
        upper = df[f].mean() + 3 * df[f].std()
        lower = df[f].mean() - 3 * df[f].std()
        df[f] = np.clip(df[f], lower, upper)
        # Z-score标准化：(X - Mean) / Std
        df[f] = (df[f] - df[f].mean()) / df[f].std()

    # 2. 构建回归模型 (Y = B1*X1 + B2*X2 + B3*X3 + Alpha)
    X = df[factors]
    X = sm.add_constant(X)  # 添加截距项 (即全市场的基准Alpha)
    Y = df['Target_Return']

    model = sm.OLS(Y, X).fit()

    # 3. 解析与输出风向
    print("\n【当前市场超额收益暴露方向报告】")
    print("-" * 50)

    for factor in factors:
        coef = model.params[factor]
        t_value = model.tvalues[factor]

        # t值的绝对值大于1.96通常认为具有统计显著性
        significance = "显著 ★" if abs(t_value) > 1.96 else "不显著"
        direction = "正向暴露 (资金追捧)" if coef > 0 else "负向暴露 (资金抛售)"

        print(
            f"因子: {factor.ljust(10)} | 暴露系数(Beta): {coef:>8.4f} | T值: {t_value:>7.2f} ({significance}) -> {direction}")

    print("-" * 50)
    print(f"基准收益 (Alpha): {model.params['const']:.4f}")
    print(f"模型解释力 (R-squared): {model.rsquared:.4f}")


if __name__ == '__main__':
    bs.login()

    # 这里我们使用最近的一个交易日作为锚点，你可以随时更改
    current_date = datetime.now().strftime("%Y-%m-%d")
    # 为了保证能取到数据，如果是周末跑代码，可以手动把日期往前设为上周五，例如 '2026-04-10'

    zz800_codes = get_zz800_stocks()
    df_data = get_factor_data(zz800_codes, end_date=current_date)

    if not df_data.empty:
        analyze_factor_exposure(df_data)
    else:
        print("未获取到足够的数据，请检查日期或网络连接。")

    bs.logout()
