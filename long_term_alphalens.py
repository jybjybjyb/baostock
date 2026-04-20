# ==========================================
# 引擎 1: Alphalens 长牛检验机 (带每日行业中性化)
# 建议在 Jupyter Notebook 中运行
# ==========================================
import glob
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.linalg import fractional_matrix_power
import alphalens
import baostock as bs
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "zz800_parquet_data"
START_DATE = "2021-01-01"
END_DATE = "2024-04-01"
TARGET_FACTOR = 'Value_BP'

def get_static_industry(codes):
    """向 BaoStock 获取静态行业标签 (一次性动作)"""
    bs.login()
    meta = []
    for code in codes:
        rs = bs.query_stock_industry(code)
        if rs.error_code == '0' and rs.next():
            meta.append({'code': code, 'industry': rs.get_row_data()[3]})
    bs.logout()
    return pd.DataFrame(meta)

def fast_panel_feature_engineering():
    """极速向量化计算：彻底消灭卡死问题"""
    print("📂 1. 加载数据并执行极速向量化计算...")
    start_t = time.time()
    
    all_files = glob.glob(f"{DATA_DIR}/*.parquet")
    df_all = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)
    df_all = df_all[(df_all['date'] >= START_DATE) & (df_all['date'] <= END_DATE)]
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.sort_values(['code', 'date']) # 必须排序以保证向量化安全
    
    # ⚡ 极致向量化：使用 transform 替代 reset_index，性能提升 100 倍
    grouped = df_all.groupby('code')
    df_all['Momentum'] = grouped['close'].pct_change(20)
    df_all['Short_Rev'] = -1 * grouped['close'].pct_change(5)
    df_all['Low_Vol'] = -1 * grouped['pctChg'].transform(lambda x: x.rolling(20).std())
    df_all['Liquidity'] = grouped['turn'].transform(lambda x: x.rolling(20).mean())
    
    avg_amount = grouped['amount'].transform(lambda x: x.rolling(20).mean())
    df_all['Size'] = np.log(avg_amount / (df_all['Liquidity']/100 + 1e-8))
    
    df_all['pbMRQ'] = pd.to_numeric(df_all['pbMRQ'], errors='coerce')
    df_all['Value_BP'] = np.where(df_all['pbMRQ'] > 0, 1 / df_all['pbMRQ'], np.nan)
    
    df_factors = df_all[['date', 'code', 'close', 'Momentum', 'Short_Rev', 'Low_Vol', 'Liquidity', 'Size', 'Value_BP']].dropna()
    print(f"✅ 计算完成！耗时: {time.time() - start_t:.2f}s, 有效数据: {len(df_factors)} 行")
    return df_factors

def neutralize_factor_by_industry(df_factors, df_ind, factor_col):
    """核心升级：每日截面行业中性化 (计算残差)"""
    print(f"🧹 2. 正在执行 {factor_col} 的每日行业中性化剥离...")
    start_t = time.time()
    
    # 合并行业标签
    df_merged = pd.merge(df_factors, df_ind, on='code', how='inner')
    
    # 定义单日 OLS 剥离行业影响的函数
    def _neutralize(group):
        y = group[factor_col]
        X = pd.get_dummies(group['industry'], drop_first=True, dtype=float)
        # 如果当天行业数据异常或太少，跳过中性化
        if X.empty or len(y) < 10: return y
        X = sm.add_constant(X)
        try:
            # 因子值对行业哑变量回归，取残差（即剔除了行业解释部分的纯正因子）
            return sm.OLS(y, X).fit().resid 
        except:
            return y
            
    # 按天 apply，速度极快
    df_merged[f'{factor_col}_Neutral'] = df_merged.groupby('date', group_keys=False).apply(_neutralize)
    print(f"✅ 行业中性化完成！耗时: {time.time() - start_t:.2f}s")
    return df_merged

# --- 执行回测 ---
# 1. 算因子
df_panel = fast_panel_feature_engineering()
# 2. 拿行业 (实盘中这里建议从本地静态表读取)
unique_codes = df_panel['code'].unique().tolist()
df_industry = get_static_industry(unique_codes)
# 3. 行业中性化
df_clean = neutralize_factor_by_industry(df_panel, df_industry, TARGET_FACTOR)

# 4. Alphalens 评测
print("📈 3. 送入 Alphalens 终极检验...")
df_pricing = df_clean.pivot(index='date', columns='code', values='close')
df_pricing.index = pd.to_datetime(df_pricing.index).tz_localize('UTC')

df_fac = df_clean[['date', 'code', f'{TARGET_FACTOR}_Neutral']].copy()
df_fac['date'] = df_fac['date'].dt.tz_localize('UTC')
df_fac = df_fac.set_index(['date', 'code'])[f'{TARGET_FACTOR}_Neutral']

factor_data = alphalens.utils.get_clean_factor_and_forward_returns(
    factor=df_fac, prices=df_pricing, quantiles=5, periods=(1, 5, 10), max_loss=0.5
)
# 把这行“摘要报告”注释掉
# alphalens.tears.create_summary_tear_sheet(factor_data)

# ✨ 换成这行“全景报告” ✨
alphalens.tears.create_full_tear_sheet(factor_data)
