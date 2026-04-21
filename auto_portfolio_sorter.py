import os
import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
import re

warnings.filterwarnings('ignore')

# --- 配置区 ---
DATA_DIR = "zz800_parquet_data"
LIST_FILE = "list.csv"
FACTOR_COLS = ['Momentum', 'Short_Rev',
               'Low_Vol', 'Liquidity', 'Size', 'Value_BP']


def get_market_data_and_factors():
    """读取全量数据，计算雷达所需(T-5)和最新截面(T0)的因子"""
    print("📂 1. 正在加载全市场 Parquet 数据...")
    all_files = glob.glob(f"{DATA_DIR}/*.parquet")
    if not all_files:
        raise ValueError("未找到 Parquet 数据！")

    df_all = pd.concat([pd.read_parquet(f)
                       for f in all_files], ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all.sort_values(['code', 'date'], inplace=True)

    unique_dates = sorted(df_all['date'].unique())
    if len(unique_dates) < 25:
        raise ValueError("数据过少，无法计算均线。")
    t0_date = unique_dates[-1]
    t5_date = unique_dates[-6]
    print(
        f"📅 最新截面(T0): {t0_date.strftime('%Y-%m-%d')} | 测算基准(T-5): {t5_date.strftime('%Y-%m-%d')}")

    grouped = df_all.groupby('code')
    df_all['Target_Y'] = grouped['close'].shift(-5) / df_all['close'] - 1
    df_all['Momentum'] = grouped['close'].pct_change(20)
    df_all['Short_Rev'] = -1 * grouped['close'].pct_change(5)
    df_all['Low_Vol'] = -1 * \
        grouped['pctChg'].transform(lambda x: x.rolling(20).std())
    df_all['Liquidity'] = grouped['turn'].transform(
        lambda x: x.rolling(20).mean())
    df_all['Size'] = np.log(grouped['amount'].transform(
        lambda x: x.rolling(20).mean()) / (df_all['Liquidity']/100 + 1e-8))
    df_all['pbMRQ'] = pd.to_numeric(df_all['pbMRQ'], errors='coerce')
    df_all['Value_BP'] = np.where(
        df_all['pbMRQ'] > 0, 1 / df_all['pbMRQ'], np.nan)

    # df_radar 用于 OLS 训练，必须极其纯净，保留 dropna
    df_radar = df_all[df_all['date'] == t5_date].copy().dropna(
        subset=['Target_Y'] + FACTOR_COLS)

    # ✨ 核心修复：df_latest 仅用于给个股打分，去掉 dropna()，允许偏科生存在！
    df_latest = df_all[df_all['date'] == t0_date].copy()

    return df_radar, df_latest


def find_king_factor(df_radar):
    """运行截面回归，找出当前Beta最高的最强因子"""
    print("\n📡 2. 启动雷达扫描，寻找当前市场 [最强因子]...")
    df = df_radar.copy()

    for c in FACTOR_COLS:
        df[c] = np.clip(df[c], df[c].mean() - 3*df[c].std(),
                        df[c].mean() + 3*df[c].std())
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    Y = df['Target_Y'] * 100
    X = sm.add_constant(df[FACTOR_COLS])
    model = sm.OLS(Y, X).fit()

    best_factor = None
    max_beta = -float('inf')

    for factor in FACTOR_COLS:
        beta = model.params[factor]
        t_val = model.tvalues[factor]
        print(
            f"  > {factor.ljust(10)} : Beta = {beta:>6.3f} | T值 = {t_val:>6.2f}")
        if beta > max_beta and t_val > 1.0:
            max_beta = beta
            best_factor = factor

    if best_factor is None:
        best_factor = 'Momentum'
    else:
        print(f"👑 雷达锁定！当前市场超额暴露最大的核心因子是: 【{best_factor}】")

    return best_factor

# --- 核心升级：前缀自动转换函数 ---


def normalize_code(code_str):
    """终极修复版：解决 Excel 吞掉前导零的问题"""
    code_str = str(code_str).strip()

    # 提取字符串中所有的纯数字
    numbers = re.sub(r'\D', '', code_str)
    if not numbers:
        return code_str

    # ✨ 核心修复：不足 6 位自动在前面补 0 (400 -> 000400)
    num = numbers.zfill(6)

    if num.startswith('6'):
        return f"sh.{num}"
    if num.startswith(('0', '3')):
        return f"sz.{num}"
    if num.startswith(('8', '4')):
        return f"bj.{num}"
    return code_str


def update_and_sort_list(df_latest, best_factor):
    """读取 list.csv，自动映射前缀，更新真实 Z-score 并安全覆写"""
    print(f"\n📝 3. 正在读取并更新 {LIST_FILE}...")
    if not os.path.exists(LIST_FILE):
        raise FileNotFoundError(f"找不到 {LIST_FILE}！")

    # 智能读取：容错处理有无表头的情况
    try:
        df_list = pd.read_csv(LIST_FILE, encoding='utf-8')
        # 如果第一行被误认成了表头(比如列名变成了 600031 和 三一重工)
        if 'code' not in df_list.columns:
            df_list = pd.read_csv(LIST_FILE, header=None, names=[
                                  'code', 'name'], encoding='utf-8')
    except UnicodeDecodeError:
        # 兼容 GBK 编码（国内 Excel 默认编码）
        df_list = pd.read_csv(LIST_FILE, header=None, names=[
                              'code', 'name'], encoding='gbk')

    df_list['code'] = df_list['code'].astype(str).str.strip()
    if 'name' in df_list.columns:
        df_list['name'] = df_list['name'].astype(str).str.strip()

    # 生成 BaoStock 标准代码用于匹配
    df_list['bs_code'] = df_list['code'].apply(normalize_code)
    pool_codes = df_list['bs_code'].tolist()

    # 市场全量 Z-score 标准化
    df = df_latest.copy()
    for c in FACTOR_COLS:
        df[c] = np.clip(df[c], df[c].mean() - 3*df[c].std(),
                        df[c].mean() + 3*df[c].std())
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    # 匹配并重命名列
    df_filtered = df[df['code'].isin(pool_codes)][['code'] + FACTOR_COLS]
    df_filtered = df_filtered.rename(columns={'code': 'bs_code'})

    # 清理旧因子列（防止多次运行后出现 Momentum_x, Momentum_y）
    cols_to_keep = [col for col in df_list.columns if col not in FACTOR_COLS]

    # 完美合并
    df_final = pd.merge(df_list[cols_to_keep],
                        df_filtered, on='bs_code', how='left')

    # 按最强因子降序排列
    df_final.sort_values(by=best_factor, ascending=False, inplace=True)
    df_final.reset_index(drop=True, inplace=True)

    # 删掉临时的映射前缀，保持你原始的 600031 格式
    df_final.drop(columns=['bs_code'], inplace=True)

    # 写入 CSV（使用 utf-8-sig 编码，确保用 Excel 打开绝对不会中文乱码）
    df_final.to_csv(LIST_FILE, index=False, encoding='utf-8-sig')

    print("=" * 60)
    print(f"🎉 任务完成！你的底仓已根据 【{best_factor}】 重新洗牌。")
    print(f"前 5 名最强金股如下：")
    for i, row in df_final.head(50).iterrows():
        name = row['name'] if 'name' in row else ''
        print(
            f"  [Top {i+1}] {row['code']} {name.ljust(8)} | {best_factor} 得分: {row[best_factor]:>5.2f}")
    print("=" * 60)

    return df_final


if __name__ == "__main__":
    df_radar, df_latest = get_market_data_and_factors()
    king_factor = find_king_factor(df_radar)
    update_and_sort_list(df_latest, king_factor)


