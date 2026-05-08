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
FACTOR_COLS = ['Momentum', 'Short_Rev', 'Low_Vol', 'Liquidity', 'Size', 'Value_BP',
        'Mom_Sharpe', 'Vol_Price_Corr', 'Amihud']

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
    
    # --- 古典因子 ---
    df_all['Momentum'] = grouped['close'].pct_change(20)
    df_all['Short_Rev'] = -1 * grouped['close'].pct_change(5)
    
    # 提取 20日波动率 (给低波因子和夏普动量共用)
    vol_20 = grouped['pctChg'].transform(lambda x: x.rolling(20).std())
    df_all['Low_Vol'] = -1 * vol_20
    
    df_all['Liquidity'] = grouped['turn'].transform(
        lambda x: x.rolling(20).mean())
    df_all['Size'] = np.log(grouped['amount'].transform(
        lambda x: x.rolling(20).mean()) / (df_all['Liquidity']/100 + 1e-8))
    df_all['pbMRQ'] = pd.to_numeric(df_all['pbMRQ'], errors='coerce')
    df_all['Value_BP'] = np.where(
        df_all['pbMRQ'] > 0, 1 / df_all['pbMRQ'], np.nan)

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

    # df_radar 用于 OLS 训练，必须极其纯净，保留 dropna
    df_radar = df_all[df_all['date'] == t5_date].copy().dropna(
        subset=['Target_Y'] + FACTOR_COLS)

    # df_latest 仅用于给个股打分，去掉 dropna()，允许偏科生存在！
    df_latest = df_all[df_all['date'] == t0_date].copy()

    return df_radar, df_latest

# 替换原有的 find_king_factor 函数


def find_king_factor(df_radar):
    """运行截面回归，找出当前Beta最高的最强因子与次强因子"""
    print("\n📡 2. 启动雷达扫描，寻找当前市场 [最强因子] & [次强因子]...")
    df = df_radar.copy()

    for c in FACTOR_COLS:
        df[c] = np.clip(df[c], df[c].mean() - 3*df[c].std(),
                        df[c].mean() + 3*df[c].std())
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    Y = df['Target_Y'] * 100
    X = sm.add_constant(df[FACTOR_COLS])
    model = sm.OLS(Y, X).fit()

    factor_stats = []
    for factor in FACTOR_COLS:
        beta = model.params[factor]
        t_val = model.tvalues[factor]
        print(
            f"  > {factor.ljust(15)} : Beta = {beta:>6.3f} | T值 = {t_val:>6.2f}")
        factor_stats.append({
            '因子名称': factor, 'Beta (强度)': round(beta, 4), 'T值 (显著性)': round(t_val, 2)
        })

    # ✨ 核心改进：按 Beta 强度降序排列，提取前两名
    df_stats = pd.DataFrame(factor_stats).sort_values(
        by='Beta (强度)', ascending=False)

    best_factor = df_stats.iloc[0]['因子名称'] if not df_stats.empty else 'Momentum'
    sub_factor = df_stats.iloc[1]['因子名称'] if len(df_stats) > 1 else 'Low_Vol'

    print(f"👑 雷达锁定！当前市场核心最强因子: 【{best_factor}】 | 次强因子: 【{sub_factor}】")

    # 返回三个对象：最强因子、次强因子、完整战报表
    return best_factor, sub_factor, df_stats


# 在文件末尾追加全新的 rank_zz800_top100 函数
def rank_zz800_top100(df_latest, best_factor, sub_factor):
    """全市场清洗：基于最强与次强因子构建复合得分，输出中证800前100强"""
    print(
        f"\n🌐 4. 正在对全市场 ZZ800 进行重新排序 (权重: 70% {best_factor} + 30% {sub_factor})...")
    df = df_latest.copy()

    # 1. 市场全量 Z-score 标准化 (必须重新标准化，保证横向可比)
    for c in FACTOR_COLS:
        df[c] = np.clip(df[c], df[c].mean() - 3*df[c].std(),
                        df[c].mean() + 3*df[c].std())
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    # 2. 构建上帝视角的复合得分 (Composite Score)
    df['Composite_Score'] = df[best_factor] * 0.7 + df[sub_factor] * 0.3

    # 3. 挂载名称和行业方便阅读
    import market_radar as radar
    try:
        df_meta = radar.load_local_metadata()
        df = pd.merge(df, df_meta, on='code', how='left')
    except Exception:
        df['name'] = '未知'
        df['Industry'] = '未知'

    # 4. 截取前 100 名最强阿尔法标的
    df_top100 = df.sort_values(by='Composite_Score', ascending=False).head(100)
    df_top100.reset_index(drop=True, inplace=True)

    # 5. 落盘归档
    out_file = "zz800_top100.csv"
    columns_to_export = ['code', 'name', 'Industry',
                         'Composite_Score', best_factor, sub_factor]
    df_top100[columns_to_export].to_csv(
        out_file, index=False, encoding='utf-8-sig')

    print("=" * 65)
    print(f"🌟 ZZ800 全市场 Top 100 终极榜单已保存至: {out_file}！(前10名展示)")
    for i, row in df_top100.head(10).iterrows():
        print(
            f"  [Top {i+1:>2}] {row['code']} {row['name']:<6} | 行业: {row['Industry']:<6} | 综合得分: {row['Composite_Score']:>5.2f}")
    print("=" * 65)

    return df_top100


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
    
    # ✨ 核心修复：用两个变量分别接住“王冠”和“战绩表”
    king_factor, king_stats = find_king_factor(df_radar)
    
    # 排行榜排序时，只传入因子的名字 (king_factor)
    update_and_sort_list(df_latest, king_factor)


