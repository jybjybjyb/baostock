import alphalens
import os
import glob
import time
import baostock as bs
import pandas as pd
import numpy as np
import statsmodels.api as sm

# --- 配置区 ---
DATA_DIR = "zz800_parquet_data"
# 回测横截面锚点：为了能计算“未来5天”的收益率，锚点定在几周前
# (如果你用今天的日期，就没有“未来”可以用来计算 Target Return 了)
TARGET_DATE = "2026-04-13"


import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power

def symmetric_orthogonalize(df, factor_cols):
    """
    对指定的风格因子进行对称正交化，消除多重共线性
    """
    print(f"🧹 正在对 {len(factor_cols)} 个因子执行对称正交化...")
    
    # 1. 提取因子矩阵 F (N行 x K列)
    # 确保没有 NaN，且最好已经做过标准化 (Z-score)
    F = df[factor_cols].values 
    
    # 2. 计算内积矩阵 M = F^T * F (相当于协方差矩阵)
    M = np.dot(F.T, F)
    
    # 3. 计算 M 的负二分之一次方 (M^(-1/2))
    # 使用 scipy 的 fractional_matrix_power 处理矩阵分数次幂
    M_inv_half = fractional_matrix_power(M, -0.5)
    
    # 4. 生成正交化后的矩阵: F_orth = F * M^(-1/2)
    F_orth = np.dot(F, M_inv_half)
    
    # 5. 将结果覆盖回 DataFrame
    df_orth = df.copy()
    # 取实数部分（防止计算精度导致的极小复数）
    df_orth[factor_cols] = np.real(F_orth) 
    
    print("✅ 正交化完成！因子间的相关系数已全部归零。")
    return df_orth


def run_alphalens_tearsheet(df_k_history, df_factors, target_factor='Value_BP'):
    """
    一键生成 Alphalens 因子评估报告
    参数:
    - df_k_history: 我们之前用 Parquet 引擎加载的完整 K 线数据
    - df_factors: 包含历史每天每只股票因子值的 DataFrame
    - target_factor: 你想测试的具体因子名称
    """
    print(f"\n📊 正在准备 [{target_factor}] 的 Alphalens 数据格式...")

    # 1. 准备价格矩阵 (Pricing Data)
    # 必须是：行是 datetime，列是股票代码，值是收盘价
    df_pricing = df_k_history.pivot(
        index='date', columns='code', values='close')
    # 确保索引是 tz-localized 的 datetime 格式 (Alphalens 的强迫症要求)
    df_pricing.index = pd.to_datetime(df_pricing.index).tz_localize('UTC')

    # 2. 准备因子数据 (Factor Data)
    # 必须是：MultiIndex (date, asset) 的 Series
    df_fac = df_factors[['date', 'code', target_factor]].copy()
    df_fac['date'] = pd.to_datetime(df_fac['date']).dt.tz_localize('UTC')
    df_fac = df_fac.set_index(['date', 'code'])
    factor_series = df_fac[target_factor]

    # 3. 核心步骤：数据对齐与前向收益计算 (Clean Factor)
    # 这个函数会自动帮你计算未来 1天、5天、10天 的收益率，并剔除停牌/退市数据
    print("⏳ 正在计算前向收益与分位数对齐...")
    try:
        factor_data = alphalens.utils.get_clean_factor_and_forward_returns(
            factor=factor_series,
            prices=df_pricing,
            quantiles=5,          # 分为 5 组进行回测
            periods=(1, 5, 10),   # 分别看未来 1 天、5 天、10 天的预测力
            max_loss=0.35         # 允许最大 35% 的数据因停牌等原因丢失
        )
    except Exception as e:
        print(f"数据清洗失败: {e}")
        return

    # 4. 一键召唤神龙：生成全套评估报告
    print("📈 生成因子全景评估报告 (请在 Jupyter Notebook 中查看效果最佳)...")

    # 打印文字版的 IC/IR 和收益率总结表格
    alphalens.tears.create_summary_tear_sheet(factor_data)

    # 如果你在 Jupyter 跑，用下面这行可以画出所有的图表：
    # alphalens.tears.create_full_tear_sheet(factor_data)
    


def load_local_k_data(target_date):
    """瞬间读取本地所有 Parquet 文件，并动态截取我们需要的时间窗口"""
    print(f"📂 正在从本地加载 {target_date} 附近的 K 线数据...")
    start_time = time.time()

    all_files = glob.glob(f"{DATA_DIR}/*.parquet")
    if not all_files:
        raise ValueError("本地没有找到 Parquet 数据，请先运行数据下载引擎！")

    # --- 修复核心：动态计算时间窗口 ---
    # 我们需要目标日期往前推 60 天（算因子），往后推 20 天（算目标收益）
    target_dt = pd.to_datetime(target_date)
    start_date = (target_dt - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    end_date = (target_dt + pd.Timedelta(days=20)).strftime("%Y-%m-%d")

    df_list = []
    for f in all_files:
        df = pd.read_parquet(f)
        # 动态截取数据
        df_window = df[(df['date'] <= end_date) & (df['date'] >= start_date)]
        if not df_window.empty:
            df_list.append(df_window)

    if not df_list:
        raise ValueError(
            f"⚠️ 在 {start_date} 到 {end_date} 期间本地没有任何数据！请检查目标日期是否在你的下载范围内。")

    df_all = pd.concat(df_list, ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all.sort_values(['code', 'date'], inplace=True)

    print(
        f"✅ 加载完成！读取了 {len(df_list)} 只股票，共 {len(df_all)} 行数据，耗时: {time.time() - start_time:.2f} 秒")
    return df_all


def calculate_cross_section_factors(df_k, df_meta, target_date):
    """计算横截面因子和目标收益率，并增加防爆防空锁"""
    print(f"🧮 正在计算 {target_date} 截面的 10 维因子矩阵...")
    target_dt = pd.to_datetime(target_date)

    factors = []
    for code, group in df_k.groupby('code'):
        past_df = group[group['date'] <= target_dt].tail(20)
        future_df = group[group['date'] > target_dt].head(5)

        if len(past_df) < 15 or len(future_df) < 5:
            continue  # 数据不足剔除

        p_close = past_df['close'].values
        p_pct = past_df['pctChg'].values
        p_amount = past_df['amount'].values
        p_turn = past_df['turn'].values

        # 因子计算
        momentum = p_close[-1] / p_close[0] - 1
        short_rev = -1 * (p_close[-1] / p_close[-5] - 1)
        low_vol = -1 * np.nanstd(p_pct)

        row_t0 = past_df.iloc[-1]
        pb, pe, ps = row_t0['pbMRQ'], row_t0['peTTM'], row_t0['psTTM']
        val_bp = 1 / pb if pb > 0 else np.nan
        val_ep = 1 / pe if pe > 0 else np.nan
        val_sp = 1 / ps if ps > 0 else np.nan

        avg_turn = np.nanmean(p_turn)
        size = np.log(np.nanmean(p_amount) / (avg_turn/100 + 1e-8)
                      ) if avg_turn > 0 else np.nan
        liquidity = avg_turn

        target_return = future_df['close'].iloc[-1] / p_close[-1] - 1

        factors.append({
            'code': code,
            'Target_Y': target_return,
            'Momentum': momentum, 'Short_Rev': short_rev, 'Low_Vol': low_vol,
            'Value_BP': val_bp, 'Value_EP': val_ep, 'Value_SP': val_sp,
            'Size': size, 'Liquidity': liquidity
        })

    df_factors = pd.DataFrame(factors)

    # --- 修复核心：安全锁 ---
    if df_factors.empty:
        raise ValueError(
            f"⚠️ 致命错误：在 {target_date} 截面上，没有任何股票满足计算因子的条件（可能由于停牌、节假日或本地数据未覆盖未来5天）。请尝试更换 TARGET_DATE！")

    df_final = pd.merge(df_factors, df_meta, on='code', how='inner')
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.dropna(inplace=True)
    return df_final


def fetch_industry_and_finance(codes):
    """从 BaoStock 补充行业标签和最新的 ROE/Growth 财报数据"""
    print("🌐 正在向 BaoStock 补充截面行业与财务特征，预计需 1-2 分钟...")
    bs.login()
    meta_data = []

    for i, code in enumerate(codes):
        # 1. 行业数据
        ind_rs = bs.query_stock_industry(code)
        industry = "未知"
        if ind_rs.error_code == '0' and ind_rs.next():
            industry = ind_rs.get_row_data()[3]  # industry 字段

        # 2. 财务数据 (以 2023 年三季报为例，因为锚点是 24年3月)
        prof_rs = bs.query_profit_data(code=code, year=2023, quarter=3)
        roe, growth = 0.0, 0.0
        if prof_rs.error_code == '0' and prof_rs.next():
            row = prof_rs.get_row_data()
            roe = float(row[5]) if row[5] else 0.0
            growth = float(row[10]) if row[10] else 0.0

        meta_data.append({
            'code': code,
            'Industry': industry,
            'ROE': roe,
            'Growth': growth
        })

        if (i+1) % 100 == 0:
            print(f"   已获取 {i+1}/{len(codes)} 只股票元数据...")

    bs.logout()
    return pd.DataFrame(meta_data)




def variance_decomposition_analysis(df):
    """执行三步法回归，对比方差解释力 R-squared"""
    print("\n" + "="*50)
    print("🔬 开始执行多重方差分解分析 (Variance Decomposition)")
    print("="*50)

    factor_cols = ['Momentum', 'Short_Rev', 'Low_Vol', 'Value_BP',
                   'Value_EP', 'Value_SP', 'Size', 'Liquidity', 'ROE', 'Growth']

    # 1. 因子标准化处理 (Z-Score & 去极值)
    for c in factor_cols:
        upper, lower = df[c].mean() + 3 * \
            df[c].std(), df[c].mean() - 3 * df[c].std()
        df[c] = np.clip(df[c], lower, upper)
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    # 2. 行业哑变量处理
    industry_dummies = pd.get_dummies(
        df['Industry'], drop_first=True, dtype=float)

    # 目标 Y
    Y = df['Target_Y'] * 100  # 放大为百分比方便看

    # --- 模型 1：纯风格因子 ---
    X_factors = sm.add_constant(df[factor_cols])
    model_1 = sm.OLS(Y, X_factors).fit()
    r2_factors = model_1.rsquared

    # --- 模型 2：纯行业哑变量 ---
    X_ind = sm.add_constant(industry_dummies)
    model_2 = sm.OLS(Y, X_ind).fit()
    r2_ind = model_2.rsquared

    # --- 模型 3：全模型 (因子 + 行业) ---
    X_full = sm.add_constant(
        pd.concat([df[factor_cols], industry_dummies], axis=1))
    model_3 = sm.OLS(Y, X_full).fit()
    r2_full = model_3.rsquared

    # --- 输出深度结论 ---
    print("\n【第一部分：决定系数 R² 对比 (谁主导了市场的涨跌？)】")
    print(f"📈 1. 纯风格因子解释力 : {r2_factors * 100:>5.2f} %")
    print(f"🏭 2. 纯行业板块解释力 : {r2_ind * 100:>5.2f} %")
    print(f"🌍 3. 全模型综合解释力 : {r2_full * 100:>5.2f} %")

    print("\n💡 [洞察结论]:")
    if r2_ind > r2_factors * 1.5:
        print(">> 当前市场是典型的【行业主导市】。买对板块比选对个股重要得多。")
    else:
        print(">> 当前市场风格特征明显，因子选股(Alpha)大有可为。")

    print(f">> 剔除行业影响后，10大因子提供的纯增量解释力为: {(r2_full - r2_ind) * 100:.2f} %")

    print("\n【第二部分：全模型下的纯净因子风向标 (剔除行业干扰后)】")
    print("-" * 65)
    for factor in factor_cols:
        coef = model_3.params[factor]
        t_value = model_3.tvalues[factor]
        sig = "显著 ★" if abs(t_value) > 1.96 else "不显著"
        direct = "正向暴露" if coef > 0 else "负向暴露"
        print(
            f"因子: {factor.ljust(10)} | Beta(收益率): {coef:>7.3f}% | T值: {t_value:>6.2f} ({sig}) -> {direct}")
    print("-" * 65)


if __name__ == "__main__":
    # 1. 极速读取本地数据
    df_k = load_local_k_data(TARGET_DATE)

    # 2. 提取唯一的股票代码，去补充元数据
    unique_codes = df_k['code'].unique().tolist()
    df_meta = fetch_industry_and_finance(unique_codes)

    # 3. 矩阵计算
    df_final = calculate_cross_section_factors(df_k, df_meta, TARGET_DATE)

    # 4. 执行归因分解
    variance_decomposition_analysis(df_final)
