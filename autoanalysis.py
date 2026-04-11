import os
import csv
import baostock as bs
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import mplfinance._utils as mpf_utils
from datetime import datetime, timedelta



# ==========================================
# 【底层系统级 Bug 修复：猴子补丁 (增强自适应版)】
# 兼容 Pandas Series 与 Numpy ndarray 双重类型
# ==========================================
original_calculate_atr = mpf_utils._calculate_atr

def patched_calculate_atr(n, highs, lows, closes):
    # 动态探测并安全提取底层数据
    h = highs.values if hasattr(highs, 'values') else highs
    l = lows.values if hasattr(lows, 'values') else lows
    c = closes.values if hasattr(closes, 'values') else closes
    return original_calculate_atr(n, h, l, c)

mpf_utils._calculate_atr = patched_calculate_atr
# ==========================================


# ==========================================
# 【全局核心参数控制台】
# ==========================================
# 目录配置
OUTPUT_DIR = "result"
CSV_FILE = "list.csv"

# 1. 宏观时间配置
TARGET_DAILY_DAYS = 250   # 日线图表展示天数 (Renko & PnF)
TARGET_MINUTE_DAYS = 120   # 分钟图表展示天数 (Volume Profile)

# 2. 波动率与砖块/箱体配置
ATR_PERIOD = 20
RENKO_MULTIPLIER = 0.67   # Renko 砖块乘数 (用于趋势防守)
PNF_MULTIPLIER = 0.5      # P&F 箱体乘数 (用于精细测算)
PNF_REVERSAL = 3          # P&F 反转系数

# 3. VP 微观结构配置
VP_FREQ = "5"             # Volume Profile 采样频率
VP_BINS = 120              # 筹码分布精细度

# ==========================================
# 工具模块：时间引擎与数据格式化
# ==========================================
def calculate_start_date(end_date_str, trading_days_needed):
    """根据交易日需求自动推算自然日，包含指标预热余量"""
    end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d')
    calendar_days_to_subtract = int((trading_days_needed + ATR_PERIOD) * 1.46) + 20
    start_date_obj = end_date_obj - timedelta(days=calendar_days_to_subtract)
    return start_date_obj.strftime('%Y-%m-%d')

def format_stock_code(raw_code):
    """自动为 A 股代码补充交易所前缀"""
    raw_code = str(raw_code).strip()
    if raw_code.startswith(('sh.', 'sz.', 'bj.')):
        return raw_code
    # 简易推断逻辑：6/5开头为沪市，0/3/1开头为深市，8/4为北交所
    if raw_code.startswith(('6', '5')):
        return f"sh.{raw_code}"
    elif raw_code.startswith(('0', '3', '1')):
        return f"sz.{raw_code}"
    elif raw_code.startswith(('8', '4')):
        return f"bj.{raw_code}"
    return raw_code

def load_stock_list(csv_path):
    """读取目标清单，支持多种中文编码"""
    stocks = []
    if not os.path.exists(csv_path):
        print(f"【严重错误】未找到清单文件: {csv_path}")
        return stocks
        
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    code = format_stock_code(row[0])
                    name = row[1].strip()
                    stocks.append((code, name))
    except UnicodeDecodeError:
        with open(csv_path, 'r', encoding='gbk') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    code = format_stock_code(row[0])
                    name = row[1].strip()
                    stocks.append((code, name))
    return stocks

# ==========================================
# 数据获取模块：日线与分钟线引擎
# ==========================================
def fetch_k_data(code, start_date, end_date, freq):
    """底层通用数据拉取通道"""
    fields = "date,time,open,high,low,close,volume" if freq in ["5", "15", "30", "60"] else "date,open,high,low,close,volume"
    rs = bs.query_history_k_data_plus(
        code, fields,
        start_date=start_date, end_date=end_date,
        frequency=freq, adjustflag="3"
    )
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    if df.empty: return df
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'].str.slice(0, 14), format='%Y%m%d%H%M%S')
        df.set_index('time', inplace=True)
    else:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
    df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}, inplace=True)
    df = df[df['Volume'] > 0] # 剔除死水数据
    return df

# ==========================================
# 图表生成模块：三大核心画图引擎
# ==========================================
def generate_renko_and_pnf(df_raw, stock_name, stock_code):
    """复用同一个日线 DataFrame，同时生成 Renko 和 P&F，极大节省算力"""
    df = df_raw.copy()
    
    # 计算 ATR 并清洗 NaN
    df.ta.atr(length=ATR_PERIOD, append=True)
    df.dropna(inplace=True)
    
    # 截取目标天数
    if len(df) > TARGET_DAILY_DAYS:
        df = df.tail(TARGET_DAILY_DAYS)
        
    col_name = f'ATRr_{ATR_PERIOD}'
    raw_atr = df[col_name].iloc[-1]
    
    # 1. 生成 Renko
    renko_brick = round(raw_atr * RENKO_MULTIPLIER, 2)
    renko_filename = os.path.join(OUTPUT_DIR, f"{stock_name}_Renko_ATR{ATR_PERIOD}_Mult{RENKO_MULTIPLIER}.png")
    
    # 2. 生成 P&F
    pnf_box = round(raw_atr * PNF_MULTIPLIER, 2)
    pnf_params = dict(box_size=pnf_box, reversal=PNF_REVERSAL)
    pnf_filename = os.path.join(OUTPUT_DIR, f"{stock_name}_PnF_Box{pnf_box}_Rev{PNF_REVERSAL}.png")
    
    # 执行绘图并保存
    try:
        mpf.plot(
            df, type='renko', renko_params=dict(brick_size=renko_brick), style='yahoo',
            title=f"Renko - {stock_name} ({stock_code})\nBrick={renko_brick} | {ATR_PERIOD}d ATR x {RENKO_MULTIPLIER}",
            savefig=dict(fname=renko_filename, dpi=150, bbox_inches='tight')
        )
        
        mpf.plot(
            df, type='pnf', pnf_params=pnf_params, style='yahoo',
            title=f"Point & Figure - {stock_name} ({stock_code})\nBox={pnf_box} | Rev={PNF_REVERSAL} | {ATR_PERIOD}d ATR Base",
            savefig=dict(fname=pnf_filename, dpi=150, bbox_inches='tight')
        )
    finally:
        plt.close('all') # 【极其重要】防止内存泄漏

def generate_volume_profile(df_raw, stock_name, stock_code):
    """生成高精度筹码分布并自动保存"""
    df = df_raw.copy()
    
    # 截取目标 K 线数 (天数 * 每天的分钟线数量)
    bars_per_day = {"5": 48, "15": 16, "30": 8, "60": 4}.get(VP_FREQ, 48)
    target_k_count = TARGET_MINUTE_DAYS * bars_per_day
    if len(df) > target_k_count:
        df = df.tail(target_k_count)
        
    price_min, price_max = df['Low'].min(), df['High'].max()
    price_bins = np.linspace(price_min, price_max, VP_BINS)
    
    df['Micro_Typical_Price'] = (df['High'] + df['Low']) / 2
    vprofile = df.groupby(pd.cut(df['Micro_Typical_Price'], bins=price_bins, include_lowest=True))['Volume'].sum()
    
    poc_idx = vprofile.idxmax()
    poc_price = (poc_idx.left + poc_idx.right) / 2

    # 自定义绘图架构
    fig = mpf.figure(style='yahoo', figsize=(14, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twiny()

    mpf.plot(df, type='candle', ax=ax1, volume=False, show_nontrading=False)

    bin_centers = [(b.left + b.right)/2 for b in vprofile.index]
    ax2.barh(bin_centers, vprofile.values, height=(price_max-price_min)/VP_BINS * 0.9, 
             alpha=0.4, color='dodgerblue', align='center', edgecolor='none')
    
    ax1.axhline(poc_price, color='red', linestyle='-', linewidth=2, alpha=0.8, label=f'High-Res POC: {poc_price:.2f}')
    
    ax2.set_xticks([]) 
    ax1.legend(loc='upper left')
    ax1.set_title(f"Volume Profile - {stock_name} ({stock_code})\nLast {len(df)} {VP_FREQ}-Min bars")
    
    vp_filename = os.path.join(OUTPUT_DIR, f"{stock_name}_VP_{VP_FREQ}Min_Bins{VP_BINS}.png")
    
    try:
        fig.savefig(vp_filename, dpi=150, bbox_inches='tight')
    finally:
        plt.close(fig) # 阻断内存泄漏

# ==========================================
# 核心中枢：流水线调度器
# ==========================================
def main_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stock_list = load_stock_list(CSV_FILE)
    
    if not stock_list:
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 成功加载 {len(stock_list)} 个监控标的。")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在连接数据通道...")
    
    bs.login() # 启动全局长连接
    
    end_date = datetime.today().strftime('%Y-%m-%d')
    daily_start_date = calculate_start_date(end_date, TARGET_DAILY_DAYS)
    minute_start_date = calculate_start_date(end_date, TARGET_MINUTE_DAYS)

    total = len(stock_list)
    for idx, (code, name) in enumerate(stock_list, 1):
        # 净化文件名（防止股票名包含非法字符）
        safe_name = "".join([c for c in name if c.isalnum() or c in ['-', '_']])
        
        print(f"\n--- [{idx}/{total}] 正在处理: {safe_name} ({code}) ---")
        
        try:
            # 任务 1：提取日线并生成 Renko 与 P&F
            df_daily = fetch_k_data(code, daily_start_date, end_date, "d")
            if not df_daily.empty and len(df_daily) > ATR_PERIOD:
                generate_renko_and_pnf(df_daily, safe_name, code)
                print(f"  √ 日线结构生成完毕 (Renko & P&F)")
            else:
                print(f"  × 警告：{safe_name} 日线数据缺失或过短。")

            # 任务 2：提取分钟线并生成 Volume Profile
            df_minute = fetch_k_data(code, minute_start_date, end_date, VP_FREQ)
            if not df_minute.empty:
                generate_volume_profile(df_minute, safe_name, code)
                print(f"  √ 微观结构生成完毕 (Volume Profile)")
            else:
                print(f"  × 警告：{safe_name} 分钟级数据缺失。")
                
        except Exception as e:
            print(f"  【异常跳过】处理 {safe_name} 时发生系统错误: {str(e)}")
            plt.close('all') # 发生异常时也强制清理画布

    bs.logout() # 关闭全局长连接
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 批量任务全部执行完毕！图表已保存至 '{OUTPUT_DIR}' 文件夹。")

if __name__ == '__main__':
    main_pipeline()