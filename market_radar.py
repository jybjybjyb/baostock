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
import pandas as pd
import numpy as np
import glob





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
    """彻底摒弃 For 循环的截面因子构建器 (包含高阶量价因子)"""
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
    
    df_all['Target_Y'] = grouped['close'].shift(-5) / df_all['close'] - 1
    
    # --- 古典因子 ---
    df_all['Momentum'] = grouped['close'].pct_change(20)
    df_all['Short_Rev'] = -1 * grouped['close'].pct_change(5)
    
    # 提取 20日波动率供后续使用
    vol_20 = grouped['pctChg'].transform(lambda x: x.rolling(20).std())
    df_all['Low_Vol'] = -1 * vol_20
    
    df_all['Liquidity'] = grouped['turn'].transform(lambda x: x.rolling(20).mean())
    df_all['Size'] = np.log(grouped['amount'].transform(lambda x: x.rolling(20).mean()) / (df_all['Liquidity']/100 + 1e-8))
    
    df_all['pbMRQ'] = pd.to_numeric(df_all['pbMRQ'], errors='coerce')
    df_all['Value_BP'] = np.where(df_all['pbMRQ'] > 0, 1 / df_all['pbMRQ'], np.nan)
    
    # --- ✨ 新增：高阶量价因子计算逻辑 ---
    # 1. 夏普动量
    df_all['Mom_Sharpe'] = df_all['Momentum'] / (vol_20 + 1e-8)
    
    # 2. 量价相关性
    df_all['Vol_Price_Corr'] = grouped.apply(
        lambda x: x['pctChg'].rolling(20).corr(x['turn'])
    ).reset_index(level=0, drop=True)
    
    # 3. Amihud 绝对流动性溢价
    daily_amihud = np.abs(df_all['pctChg']) / (df_all['amount'] + 1e-8)
    df_all['Amihud'] = df_all.assign(amihud=daily_amihud).groupby('code')['amihud'].transform(
        lambda x: np.log(x.rolling(20).mean() + 1e-8)
    )

    # 抽取 T-5 那一天作为横截面输入
    unique_dates = sorted(df_all['date'].unique())
    if len(unique_dates) < 6: raise ValueError("数据过少")
    t_minus_5_date = unique_dates[-6] 
    
    df_cross_section = df_all[df_all['date'] == t_minus_5_date].copy()
    
    # 这里提取才不会报错，因为上面已经把这三列算出来了
    df_factors = df_cross_section[['code', 'Target_Y', 'Momentum', 'Short_Rev', 'Low_Vol', 'Liquidity', 'Size', 'Value_BP',
    'Mom_Sharpe', 'Vol_Price_Corr', 'Amihud']].dropna()
    
    return df_factors

def symmetric_orthogonalize(df, factor_cols):
    F = df[factor_cols].values 
    M = np.dot(F.T, F)
    F_orth = np.dot(F, fractional_matrix_power(M, -0.5))
    df_orth = df.copy()
    df_orth[factor_cols] = np.real(F_orth) 
    return df_orth
def run_ols_radar(df):
    """截面方差分解 (升级版：返回结构化统计数据)"""
    print("\n🔬 3. 执行市场风向 OLS 归因分解...")
    factor_cols = ['Momentum', 'Short_Rev', 'Low_Vol', 'Liquidity', 'Size', 'Value_BP',
    'Mom_Sharpe', 'Vol_Price_Corr', 'Amihud']
    
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
    
    # ✨ 核心改造：收集数据
    stats_list = []
    for factor in factor_cols:
        coef, t_val = model_full.params[factor], model_full.tvalues[factor]
        sig = "★" if abs(t_val) > 1.96 else " "
        direct = "追捧" if coef > 0 else "抛售"
        print(f"[{factor.ljust(10)}] Beta: {coef:>6.3f} | T值: {t_val:>5.2f} {sig} -> 资金{direct}")
        
        stats_list.append({
            '因子 (Factor)': factor, 
            'Beta系数': round(coef, 3), 
            'T值 (显著性)': round(t_val, 2), 
            '风向状态': f"{direct} {sig}"
        })
        
    return {
        "r2_factors": r2_factors,
        "r2_ind": r2_ind,
        "stats_df": pd.DataFrame(stats_list)
    }

def drill_down_industry_leaders(df, factor_cols, top_n_ind=3, top_n_stock=3):
    """深度下钻 (升级版：返回结构化画像)"""
    print("\n" + "="*65)
    print("🔍 【深度下钻】行业领涨板块与龙头股因子画像")
    print("="*65)

    ind_returns = df.groupby('Industry')['Target_Y'].mean() * 100
    ind_returns = ind_returns.sort_values(ascending=False)
    
    top_inds = ind_returns.head(top_n_ind)
    bottom_inds = ind_returns.tail(top_n_ind)

    print("\n🏆 【当周强势板块 Top】")
    for ind, ret in top_inds.items(): print(f"  > {ind.ljust(8)} : 板块平均收益 {ret:>5.2f}%")

    print("\n📉 【当周弱势板块 Bottom】")
    for ind, ret in bottom_inds.items(): print(f"  > {ind.ljust(8)} : 板块平均收益 {ret:>5.2f}%")

    print("\n🎯 【强势板块内部：领涨龙头股的风格特征】")
    
    # ✨ 核心改造：收集画像数据
    portrait_list = []
    for ind in top_inds.index:
        print(f"\n[ 行业板块: {ind} ]")
        df_ind = df[df['Industry'] == ind]
        top_stocks = df_ind.sort_values(by='Target_Y', ascending=False).head(top_n_stock)

        for _, row in top_stocks.iterrows():
            code = row['code']
            name = row.get('name', '未知')
            ret = row['Target_Y'] * 100

            features = []
            for f in factor_cols:
                z_score = row[f]
                if z_score > 1.0: features.append(f"高{f}(+{z_score:.1f})")
                elif z_score < -1.0:
                    if f == 'Size': features.append(f"微盘股({z_score:.1f})")
                    else: features.append(f"低{f}({z_score:.1f})")

            feature_str = ", ".join(features) if features else "中庸(随板块普涨)"
            print(f"  标的: {code:<10} {name:<8} | 涨幅: +{ret:>5.2f}% | 画像: {feature_str}")
            
            portrait_list.append({
                '强势板块': ind, '代码': code, '名称': name, 
                '预期超额涨幅': f"+{ret:.2f}%", '核心因子画像': feature_str
            })

    return {
        "top_industries": top_inds.reset_index().rename(columns={'Industry':'板块名称', 'Target_Y':'平均超额涨幅(%)'}),
        "bottom_industries": bottom_inds.reset_index().rename(columns={'Industry':'板块名称', 'Target_Y':'平均超额涨幅(%)'}),
        "portraits_df": pd.DataFrame(portrait_list)
    }


def analyze_sector_trends(lookback=5, volume_filter_quantile=0.3):
    """
    【全新升级】：结合绝对成交额（Amount），自动剔除流动性枯竭的微型板块噪音
    """
    print("\n" + "="*65)
    print(
        f"📊 【板块轮动透视】分析当日及近 {lookback} 日强弱势板块 (🛡️ 已开启资金量降噪: 剔除后 {volume_filter_quantile*100}% 板块)")
    print("="*65)

    # 1. 极速读取基础数据 (✨ 新增 amount 列用于计算真实资金体量)
    all_files = glob.glob(f"{DATA_DIR}/*.parquet")
    df_all = pd.concat([pd.read_parquet(f, columns=['code', 'date', 'pctChg', 'amount'])
                       for f in all_files], ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])

    # 2. 截取最近 N 天的数据
    recent_dates = sorted(df_all['date'].unique())[-lookback:]
    df_recent = df_all[df_all['date'].isin(recent_dates)]

    # 3. 挂载行业元数据
    df_meta = load_local_metadata()
    df_recent = pd.merge(df_recent, df_meta, on='code', how='inner')

    # ✨ 4. 核心改造：引入资金容量过滤 (Volume Profile 降噪)
    t0_date = recent_dates[-1]
    t1_date = recent_dates[-2]

    df_t0 = df_recent[df_recent['date'] == t0_date]
    # 计算 T0 各板块的总成交额
    sector_amount = df_t0.groupby('Industry')['amount'].sum()

    # 设定动态阈值：砍掉成交额垫底的尾部板块
    threshold = sector_amount.quantile(volume_filter_quantile)
    valid_sectors = sector_amount[sector_amount >= threshold].index

    print(
        f"🔪 降噪执行：全市场 {len(sector_amount)} 个板块，已剔除成交额垫底的 {len(sector_amount) - len(valid_sectors)} 个冷门板块。")

    # 仅保留主舞台上的有效板块
    df_valid = df_recent[df_recent['Industry'].isin(valid_sectors)]

    # 5. 透视表：计算每日各行业的平均涨跌幅
    pivot_df = df_valid.pivot_table(
        index='Industry', columns='date', values='pctChg', aggfunc='mean')

    # 6. 分析变动，并挂载真实成交额
    rank_df = pd.DataFrame({
        'T0_Rank': pivot_df[t0_date].rank(ascending=False),
        'T1_Rank': pivot_df[t1_date].rank(ascending=False),
        'T0_Amount': sector_amount[valid_sectors]  # 挂载绝对资金量
    })
    rank_df['Change'] = rank_df['T1_Rank'] - rank_df['T0_Rank']

    # 打印终端战报
    soaring = rank_df.sort_values(by='Change', ascending=False).head(3)
    print("\n🚀 【核心主舞台：异动飙升板块】(前排抢筹):")
    for ind, row in soaring.iterrows():
        amt_yi = row['T0_Amount'] / 1e8  # 转换为亿元
        print(
            f"  > {ind.ljust(8)} : 排名上升 {int(row['Change']):>2} 位 | 资金量: {amt_yi:>6.1f} 亿")

    return pivot_df, rank_df


def calculate_market_crowdedness(lookback=60):
    """
    量化帝国：双波段拥挤度与流动性断层扫描器 (包含 5%/10% 斜率监控)
    """
    print("\n" + "="*65)
    print("🌊 [系统启动] 正在扫描全市场资金拥挤度、宽幅共识与极值黑洞...")
    print("="*65)

    all_files = glob.glob(f"{DATA_DIR}/*.parquet")
    df_all = pd.concat([pd.read_parquet(f, columns=['code', 'date', 'amount'])
                       for f in all_files], ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])

    recent_dates = sorted(df_all['date'].unique())[-lookback:]
    df_recent = df_all[df_all['date'].isin(recent_dates)]

    records = []

    for dt, df_day in df_recent.groupby('date'):
        df_day = df_day.sort_values(
            'amount', ascending=False).reset_index(drop=True)
        df_day = df_day.dropna(subset=['amount'])

        if len(df_day) < 200:
            continue

        total_amount = df_day['amount'].sum()
        if total_amount == 0:
            continue

        # 维度 1: 绝对流动性水位 (亿元)
        amt_10 = df_day.iloc[9]['amount'] / 1e8
        amt_50 = df_day.iloc[49]['amount'] / 1e8
        amt_100 = df_day.iloc[99]['amount'] / 1e8
        amt_200 = df_day.iloc[199]['amount'] / 1e8
        median_amount = df_day['amount'].median() / 1e8

        # 维度 2A: 极端拥挤度 (前 5% / 40只)
        top_5_count = int(len(df_day) * 0.05)
        top_5_amt = df_day.iloc[:top_5_count]['amount'].sum()
        top_5_ratio = (top_5_amt / total_amount) * 100

        # 维度 2B: 宽幅赛道共识 (前 10% / 80只)
        top_10_count = int(len(df_day) * 0.10)
        top_10_amt = df_day.iloc[:top_10_count]['amount'].sum()
        top_10_ratio = (top_10_amt / total_amount) * 100

        # 🌟 终极杀器：拥挤度斜率 (Gradient Ratio) -> 衡量前5%龙头对后排的虹吸烈度
        gradient_ratio = (top_5_amt / top_10_amt) * \
            100 if top_10_amt > 0 else 0

        records.append({
            'date': dt,
            'Top10_Amt': amt_10,
            'Top50_Amt': amt_50,
            'Top100_Amt': amt_100,
            'Top200_Amt': amt_200,
            'Median_Amt': median_amount,
            'Top5_Ratio(%)': top_5_ratio,
            'Top10_Ratio(%)': top_10_ratio,
            'Gradient_Ratio(%)': gradient_ratio,
            'T0_Profile': df_day['amount'].values / 1e8 if dt == recent_dates[-1] else None
        })

    df_crowd = pd.DataFrame(records).set_index('date')
    print("✅ 双波段拥挤度与黑洞斜率计算完毕。")
    return df_crowd


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
    factor_cols = ['Momentum', 'Short_Rev', 'Low_Vol', 'Liquidity', 'Size', 'Value_BP',
    'Mom_Sharpe', 'Vol_Price_Corr', 'Amihud']
    drill_down_industry_leaders(df_final, factor_cols,top_n_ind=5, top_n_stock=10)


