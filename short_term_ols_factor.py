# ==========================================
# 引擎 2: 每日截面风向雷达 (OLS 归因监控)
# 无 For 循环，全量向量化，极速响应
# ==========================================
import glob
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.linalg import fractional_matrix_power
import baostock as bs
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "zz800_parquet_data"

def get_static_industry(codes):
    bs.login()
    meta = [{'code': c, 'Industry': bs.query_stock_industry(c).get_row_data()[3]} 
            for c in codes if bs.query_stock_industry(c).error_code == '0' and bs.query_stock_industry(c).next()]
    bs.logout()
    return pd.DataFrame(meta)

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

# --- 执行雷达 ---
start_time = time.time()
df_radar = fast_radar_calculation()
df_meta = get_static_industry(df_radar['code'].unique().tolist())
df_final = pd.merge(df_radar, df_meta, on='code', how='inner').dropna()
run_ols_radar(df_final)
print(f"\n⏱️ 全过程毫秒级响应，总耗时: {time.time() - start_time:.2f}s")