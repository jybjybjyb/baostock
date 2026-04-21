# ==========================================
# 引擎 2: 每日截面风向雷达 (OLS 归因监控)
# 无 For 循环，全量向量化，极速响应
# ==========================================
import os
import glob
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.linalg import fractional_matrix_power
# import baostock as bs
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "zz800_parquet_data"


# def get_static_industry_and_name(codes):
#     """向 BaoStock 获取全市场的行业标签与中文名称"""
#     print("🌐 正在向 BaoStock 获取全市场行业标签与股票名称...")
#     bs.login()
#     meta = []
#     for code in codes:
#         rs = bs.query_stock_industry(code)
#         if rs.error_code == '0' and rs.next():
#             row = rs.get_row_data()
#             meta.append({
#                 'code': code,
#                 'name': row[2],      # 索引 2 是 code_name (股票名称)
#                 'Industry': row[3]   # 索引 3 是 industry (申万一级行业)
#             })
#     bs.logout()
#     return pd.DataFrame(meta)

def load_local_metadata():
    """瞬间从本地读取股票名称与行业字典，彻底告别断网焦虑"""
    META_FILE = "zz800_metadata.csv"
    if not os.path.exists(META_FILE):
        raise FileNotFoundError(
            f"找不到 {META_FILE}！请先运行 update_metadata.py 生成静态花名册。")

    # 全部以字符串形式读取，防止 000001 变成 1
    return pd.read_csv(META_FILE, dtype=str, encoding='utf-8-sig')

def fast_radar_calculation():
    """彻底摒弃 For 循环的截面因子构建器"""
    print("📡 1. 启动雷达：全量向量化提取截面特征...")
    all_files = glob.glob(f"{DATA_DIR}/*.parquet")
    df_all = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all.sort_values(['code', 'date'], inplace=True)
    
    # 锁定最新日为 T0
    target_dt = df_all['date'].max()
    print(f"📅 监控锚点 (T0): {target_dt.strftime('%Y-%m-%d')}")
    
    # 极速特征工程
    grouped = df_all.groupby('code')
    
    # 关键逻辑：我们要解释最近 5 天 (T-5 到 T0) 的收益
    # 那么因子特征必须是 T-5 那一天的状态！
    # 收益率 Target_Y: T-5 到 T0 的涨跌幅 (使用 -5 的 shift，代表未来5天收益)
    df_all['Target_Y'] = grouped['close'].shift(-5) / df_all['close'] - 1
    
    df_all['Momentum'] = grouped['close'].pct_change(20)
    df_all['Short_Rev'] = -1 * grouped['close'].pct_change(5)
    df_all['Low_Vol'] = -1 * grouped['pctChg'].transform(lambda x: x.rolling(20).std())
    df_all['Liquidity'] = grouped['turn'].transform(lambda x: x.rolling(20).mean())
    df_all['Size'] = np.log(grouped['amount'].transform(lambda x: x.rolling(20).mean()) / (df_all['Liquidity']/100 + 1e-8))
    
    df_all['pbMRQ'] = pd.to_numeric(df_all['pbMRQ'], errors='coerce')
    df_all['Value_BP'] = np.where(df_all['pbMRQ'] > 0, 1 / df_all['pbMRQ'], np.nan)
    
    # 抽取 T-5 那一天作为横截面输入
    # 因为在 T-5 那一天，Target_Y (未来5天收益) 刚好包含了直到 T0 的表现
    unique_dates = sorted(df_all['date'].unique())
    if len(unique_dates) < 6: raise ValueError("数据过少")
    t_minus_5_date = unique_dates[-6] 
    
    df_cross_section = df_all[df_all['date'] == t_minus_5_date].copy()
    df_factors = df_cross_section[['code', 'Target_Y', 'Momentum', 'Short_Rev', 'Low_Vol', 'Liquidity', 'Size', 'Value_BP']].dropna()
    return df_factors

def symmetric_orthogonalize(df, factor_cols):
    F = df[factor_cols].values 
    M = np.dot(F.T, F)
    F_orth = np.dot(F, fractional_matrix_power(M, -0.5))
    df_orth = df.copy()
    df_orth[factor_cols] = np.real(F_orth) 
    return df_orth

def run_ols_radar(df):
    """截面方差分解"""
    print("\n🔬 3. 执行市场风向 OLS 归因分解...")
    factor_cols = ['Momentum', 'Short_Rev', 'Low_Vol', 'Liquidity', 'Size', 'Value_BP']
    
    # 标准化 & 正交化
    for c in factor_cols:
        df[c] = np.clip(df[c], df[c].mean()-3*df[c].std(), df[c].mean()+3*df[c].std())
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    df = symmetric_orthogonalize(df, factor_cols)
    
    industry_dummies = pd.get_dummies(df['Industry'], drop_first=True, dtype=float)
    Y = df['Target_Y'] * 100 
    
    r2_factors = sm.OLS(Y, sm.add_constant(df[factor_cols])).fit().rsquared
    r2_ind = sm.OLS(Y, sm.add_constant(industry_dummies)).fit().rsquared
    model_full = sm.OLS(Y, sm.add_constant(pd.concat([df[factor_cols], industry_dummies], axis=1))).fit()
    
    print(f"\n📈 风格因子解释力: {r2_factors * 100:>5.2f}% | 🏭 行业板块解释力: {r2_ind * 100:>5.2f}%")
    print("\n【剔除行业影响后的纯粹风向】")
    for factor in factor_cols:
        coef, t_val = model_full.params[factor], model_full.tvalues[factor]
        sig = "★" if abs(t_val) > 1.96 else " "
        direct = "追捧" if coef > 0 else "抛售"
        print(f"[{factor.ljust(10)}] Beta: {coef:>6.3f} | T值: {t_val:>5.2f} {sig} -> 资金{direct}")


def drill_down_industry_leaders(df, factor_cols, top_n_ind=5, top_n_stock=10):
    """
    深度下钻：提取领涨/领跌板块，并刻画强势板块内龙头股的因子画像
    """
    print("\n" + "="*65)
    print("🔍 【深度下钻】行业领涨板块与龙头股因子画像")
    print("="*65)

    # 1. 计算各板块平均涨跌幅
    ind_returns = df.groupby('Industry')['Target_Y'].mean() * 100
    ind_returns = ind_returns.sort_values(ascending=False)

    print("\n🏆 【当周强势板块 Top 】")
    for ind, ret in ind_returns.head(top_n_ind).items():
        print(f"  > {ind.ljust(8)} : 板块平均收益 {ret:>5.2f}%")

    print("\n📉 【当周弱势板块 Bottom 】")
    for ind, ret in ind_returns.tail(top_n_ind).items():
        print(f"  > {ind.ljust(8)} : 板块平均收益 {ret:>5.2f}%")

    print("\n🎯 【强势板块内部：领涨龙头股的风格特征】")
    # 2. 深度分析强势板块内的龙头股
    for ind in ind_returns.head(top_n_ind).index:
        print(f"\n[ 行业板块: {ind} ]")

        # 取出该板块的所有股票
        df_ind = df[df['Industry'] == ind]

        # 找出该板块内涨得最好的 Top 3 股票
        top_stocks = df_ind.sort_values(
            by='Target_Y', ascending=False).head(top_n_stock)

        for _, row in top_stocks.iterrows():
            code = row['code']
            # ✨ 核心增加 1：从 row 里把 name 提取出来，如果没有就显示未知
            name = row.get('name', '未知')
            ret = row['Target_Y'] * 100

            # 3. 寻找该股票最突出的因子画像 (Z-score 绝对值 > 1.0 的特征)
            features = []
            for f in factor_cols:
                z_score = row[f]
                # 因为数据是标准化过的，>1 表示排在全市场前 16%，属于极度突出
                if z_score > 1.0:
                    features.append(f"高{f}(+{z_score:.1f})")
                elif z_score < -1.0:
                    # 对于 Size，负数代表小盘；对于 Value_BP，负数代表高估值
                    if f == 'Size':
                        features.append(f"微盘股({z_score:.1f})")
                    else:
                        features.append(f"低{f}({z_score:.1f})")

            # 如果没有特别突出的因子，说明是跟着板块一起涨的“跟风股”
            feature_str = ", ".join(features) if features else "特征中庸 (随板块普涨)"

            # ✨ 核心增加 2：在 print 里把 name 加上，排版对齐
            print(
                f"  标的: {code:<10} {name:<8} | 涨幅: +{ret:>5.2f}% | 画像: {feature_str}")



if __name__ == "__main__":


    # 1. 极速算因子 (全本地)
    df_radar = fast_radar_calculation()

    # 2. 瞬间挂载本地静态花名册 (全本地)
    df_meta = load_local_metadata()

    # 3. 完美合并
    df_final = pd.merge(df_radar, df_meta, on='code', how='inner').dropna()

    # 4. 执行方差分解雷达
    run_ols_radar(df_final)

    # 5. 执行深度下钻
    factor_cols = ['Momentum', 'Short_Rev',
                   'Low_Vol', 'Liquidity', 'Size', 'Value_BP']
    drill_down_industry_leaders(df_final, factor_cols,top_n_ind=5, top_n_stock=10)


