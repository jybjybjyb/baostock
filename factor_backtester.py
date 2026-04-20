# ==========================================
# 引擎 1: Alphalens 时序回测机 (批量印钞版)
# 纯本地极速版：一次运行，生成所有因子的体检报告
# ==========================================
import os
import glob
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
import alphalens
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "zz800_parquet_data"
# 回测区间
START_DATE = "2025-01-01"
END_DATE = "2026-04-20"

# ✨ 核心改造 1：把单个因子变成一个需要批量测试的“武器库列表”
TARGET_FACTORS = ['Momentum', 'Short_Rev',
                  'Low_Vol', 'Liquidity', 'Size', 'Value_BP']


def load_local_metadata():
    META_FILE = "zz800_metadata.csv"
    if not os.path.exists(META_FILE):
        raise FileNotFoundError(f"找不到 {META_FILE}！")
    return pd.read_csv(META_FILE, dtype=str, encoding='utf-8-sig')


def fast_panel_feature_engineering():
    print("📂 1. 加载数据并执行全量特征工程 (只跑一次)...")
    start_t = time.time()

    all_files = glob.glob(f"{DATA_DIR}/*.parquet")
    df_all = pd.concat([pd.read_parquet(f)
                       for f in all_files], ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all[df_all['date'] <= END_DATE]
    df_all = df_all.sort_values(['code', 'date'])

    grouped = df_all.groupby('code')
    df_all['Momentum'] = grouped['close'].pct_change(20)
    df_all['Short_Rev'] = -1 * grouped['close'].pct_change(5)
    df_all['Low_Vol'] = -1 * \
        grouped['pctChg'].transform(lambda x: x.rolling(20).std())
    df_all['Liquidity'] = grouped['turn'].transform(
        lambda x: x.rolling(20).mean())

    avg_amount = grouped['amount'].transform(lambda x: x.rolling(20).mean())
    df_all['Size'] = np.log(avg_amount / (df_all['Liquidity']/100 + 1e-8))

    df_all['pbMRQ'] = pd.to_numeric(df_all['pbMRQ'], errors='coerce')
    df_all['Value_BP'] = np.where(
        df_all['pbMRQ'] > 0, 1 / df_all['pbMRQ'], np.nan)

    df_factors = df_all[df_all['date'] >= START_DATE]
    # 保留所有我们需要的因子列
    df_factors = df_factors[['date', 'code',
                             'close'] + TARGET_FACTORS].dropna()

    print(
        f"✅ 计算完成！耗时: {time.time() - start_t:.2f}s, 有效数据: {len(df_factors)} 行")
    return df_factors


def neutralize_factor_by_industry(df_factors, df_ind, factor_col):
    df_merged = pd.merge(
        df_factors, df_ind[['code', 'Industry']], on='code', how='inner')

    def _neutralize(group):
        y = group[factor_col]
        X = pd.get_dummies(group['Industry'], drop_first=True, dtype=float)
        if X.empty or len(y) < 10:
            return y
        X = sm.add_constant(X)
        try:
            return sm.OLS(y, X).fit().resid
        except:
            return y

    df_merged[f'{factor_col}_Neutral'] = df_merged.groupby(
        'date', group_keys=False).apply(_neutralize)
    return df_merged


if __name__ == "__main__":
    # 1. 基础数据准备 (极其消耗算力，所以放在循环外面只做一次)
    df_panel = fast_panel_feature_engineering()
    df_industry = load_local_metadata()

    # ✨ 核心改造 2：把所有因子都要用的“收盘价矩阵”提前算好，省下巨量时间
    print("⏳ 2. 正在构建全局价格矩阵 (全因子共享)...")
    df_pricing = df_panel.pivot(index='date', columns='code', values='close')
    df_pricing.index = pd.to_datetime(df_pricing.index).tz_localize('UTC')

    os.makedirs("PKL", exist_ok=True)

    # ✨ 核心改造 3：开启因子批量回测流水线
    print("\n" + "="*50)
    print(f"🏭 开启批量回测流水线，共 {len(TARGET_FACTORS)} 个因子待检阅...")
    print("="*50)

    for factor in TARGET_FACTORS:
        print(f"\n[ 正在检验武器: {factor} ]")
        try:
            # 3.1 仅对当前因子做行业中性化
            df_clean = neutralize_factor_by_industry(
                df_panel, df_industry, factor)

            # 3.2 准备 Alphalens 格式
            df_fac = df_clean[['date', 'code', f'{factor}_Neutral']].copy()
            df_fac['date'] = df_fac['date'].dt.tz_localize('UTC')
            df_fac = df_fac.set_index(['date', 'code'])[f'{factor}_Neutral']

            # 3.3 前向收益对齐与打包 (这里 periods 可以根据你是做长线还是波段来定)
            factor_data = alphalens.utils.get_clean_factor_and_forward_returns(
                factor=df_fac, prices=df_pricing, quantiles=5, periods=(1, 3, 5), max_loss=1.0
            )

            # 3.4 写入专属 PKL 文件
            cache_file = f"PKL\\{factor}_alphalens_cache.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(factor_data, f)
            print(f"  > ✅ 成功生成缓存: {cache_file}")

        except Exception as e:
            # 容错机制：如果某个因子报错（比如该因子缺失值太多无法分箱），打印错误并跳过，不影响其他因子
            print(f"  > ❌ 生成失败，跳过该因子。原因: {e}")

    print("\n" + "="*50)
    print("🎉 批量回测流水线工作完毕！所有因子的体检报告已安静地躺在 PKL 文件夹中。")
    print("="*50)
