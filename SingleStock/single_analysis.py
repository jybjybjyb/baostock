import os
import csv
import re
import baostock as bs
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import mplfinance._utils as mpf_utils
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 【底层系统级 Bug 修复：猴子补丁】
# ==========================================
original_calculate_atr = mpf_utils._calculate_atr


def patched_calculate_atr(n, highs, lows, closes):
    h = highs.values if hasattr(highs, 'values') else highs
    l = lows.values if hasattr(lows, 'values') else lows
    c = closes.values if hasattr(closes, 'values') else closes
    return original_calculate_atr(n, h, l, c)


mpf_utils._calculate_atr = patched_calculate_atr

# ==========================================
# 🎛️ 全局核心参数控制台
# ==========================================
# 📂 目录配置
OUTPUT_DIR = os.path.join("SingleStock", "result")
CSV_FILE = os.path.join("SingleStock", "target.csv")
PARQUET_DIR = "zz800_parquet_data"

# 🔭 时间视野
TARGET_DAILY_DAYS = 250
TARGET_VP_DAYS = 120

# ⚔️ 算力与防守线配置
ATR_PERIOD = 20
RENKO_MULTIPLIER = 0.67
PNF_MULTIPLIER = 0.5

# 🔬 筹码微观参数
VP_BINS = 80
VP_FREQ = "d"

# ==========================================
# 🛠️ 引擎底层功能
# ==========================================


def auto_format_code(raw_code):
    raw_code = str(raw_code).strip()
    if raw_code.startswith(('sh.', 'sz.', 'bj.')):
        return raw_code
    numbers = re.sub(r'\D', '', raw_code)
    if not numbers:
        return raw_code
    num = numbers.zfill(6)
    if num.startswith(('6', '5')):
        return f"sh.{num}"
    elif num.startswith(('0', '3', '1')):
        return f"sz.{num}"
    elif num.startswith(('8', '4')):
        return f"bj.{num}"
    return raw_code


def load_stock_list(csv_path):
    """智能读取 CSV，提取代码和【中文名称】"""
    stocks = []
    if not os.path.exists(csv_path):
        print(f"❌ 未找到目标文件: {csv_path}")
        return stocks
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if 'code' not in df.columns:
            df = pd.read_csv(csv_path, header=None, names=[
                             'code', 'name'], encoding='utf-8-sig')
    except:
        df = pd.read_csv(csv_path, header=None, names=[
                         'code', 'name'], encoding='gbk')

    for _, row in df.iterrows():
        code = auto_format_code(str(row['code']))
        name = str(row['name']).strip() if 'name' in df.columns else "未知"
        if name == "nan":
            name = "未知"

        if code and code != "nan":
            stocks.append((code, name))
    return stocks

# ==========================================
# 🚀 主力画图流水线
# ==========================================


def main_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stock_list = load_stock_list(CSV_FILE)

    if not stock_list:
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 成功加载 {len(stock_list)} 个标的，启动批量绘图引擎...")

    bs.login()

    total = len(stock_list)
    for idx, (code, name) in enumerate(stock_list, 1):
        print(f"\n--- [{idx}/{total}] 正在处理: {name} ({code}) ---")

        # ✨ 核心调整：不建文件夹，而是把名称拼装成“文件名前缀”
        safe_name = "".join([c for c in name if c.isalnum()
                            or c in ['-', '_', '\u4e00-\u9fa5']])
        clean_code = code.replace('.', '_')
        file_prefix = f"{safe_name}_{clean_code}"

        file_path = os.path.join(PARQUET_DIR, f"{code}.parquet")
        df_daily = pd.DataFrame()
        df_vp = pd.DataFrame()

        try:
            # -----------------------------------
            # 1. 抓取日线数据
            # -----------------------------------
            if os.path.exists(file_path):
                df_daily = pd.read_parquet(file_path)
            else:
                start_dt = (pd.Timestamp.now(
                ) - pd.Timedelta(days=TARGET_DAILY_DAYS*2)).strftime("%Y-%m-%d")
                rs = bs.query_history_k_data_plus(code, "date,open,high,low,close,volume", start_date=start_dt, end_date=pd.Timestamp.now(
                ).strftime("%Y-%m-%d"), frequency="d", adjustflag="2")
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                if data_list:
                    df_daily = pd.DataFrame(data_list, columns=rs.fields)

            if df_daily.empty:
                print(f"  × 警告: 无法获取 {code} 的日线数据，跳过。")
                continue

            df_daily['date'] = pd.to_datetime(df_daily['date'])
            df_daily.set_index('date', inplace=True)
            df_daily.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                            'close': 'Close', 'volume': 'Volume'}, inplace=True)
            for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df_daily[c] = df_daily[c].astype(float)
            df_daily = df_daily[df_daily['Volume'] > 0]

            # -----------------------------------
            # 2. 抓取筹码数据
            # -----------------------------------
            if VP_FREQ == "d":
                df_vp = df_daily.copy()
            else:
                start_dt = (
                    pd.Timestamp.now() - pd.Timedelta(days=TARGET_VP_DAYS)).strftime("%Y-%m-%d")
                rs = bs.query_history_k_data_plus(code, "date,time,open,high,low,close,volume", start_date=start_dt, end_date=pd.Timestamp.now(
                ).strftime("%Y-%m-%d"), frequency="5", adjustflag="2")
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                if data_list:
                    df_vp = pd.DataFrame(data_list, columns=rs.fields)
                    df_vp['time'] = pd.to_datetime(
                        df_vp['time'].str.slice(0, 14), format='%Y%m%d%H%M%S')
                    df_vp.set_index('time', inplace=True)
                    df_vp.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                                 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        df_vp[c] = df_vp[c].astype(float)
                    df_vp = df_vp[df_vp['Volume'] > 0]

            # -----------------------------------
            # 3. 渲染图表并直接保存至根目录
            # -----------------------------------
            df_chart = df_daily.tail(TARGET_DAILY_DAYS).copy()
            df_chart.ta.atr(length=ATR_PERIOD, append=True)
            df_chart.dropna(inplace=True)
            raw_atr = df_chart[f'ATRr_{ATR_PERIOD}'].iloc[-1]
            renko_brick = round(raw_atr * RENKO_MULTIPLIER, 2)
            pnf_box = round(raw_atr * PNF_MULTIPLIER, 2)

            # ✨ 取消创建文件夹，直接将中文前缀拼接在文件名上
            f_renko = os.path.join(OUTPUT_DIR, f"{file_prefix}_1_Renko.png")
            f_pnf = os.path.join(OUTPUT_DIR, f"{file_prefix}_2_PnF.png")
            f_vp = os.path.join(OUTPUT_DIR, f"{file_prefix}_3_VP.png")

            mpf.plot(df_chart, type='renko', renko_params=dict(brick_size=renko_brick), style='yahoo',
                     title=f"Renko - {code}", savefig=dict(fname=f_renko, dpi=120, bbox_inches='tight'))

            mpf.plot(df_chart, type='pnf', pnf_params=dict(box_size=pnf_box, reversal=3), style='yahoo',
                     title=f"Point & Figure - {code}", savefig=dict(fname=f_pnf, dpi=120, bbox_inches='tight'))

            if not df_vp.empty:
                df_vp = df_vp.tail(
                    TARGET_VP_DAYS * 48 if VP_FREQ == "5" else TARGET_VP_DAYS)
                price_min, price_max = df_vp['Low'].min(), df_vp['High'].max()
                df_vp['Mid'] = (df_vp['High'] + df_vp['Low']) / 2
                vprofile = df_vp.groupby(pd.cut(df_vp['Mid'], bins=np.linspace(
                    price_min, price_max, VP_BINS), include_lowest=True), observed=False)['Volume'].sum()
                poc_price = (vprofile.idxmax().left +
                             vprofile.idxmax().right) / 2

                fig = mpf.figure(style='yahoo', figsize=(14, 7))
                ax1 = fig.add_subplot(1, 1, 1)
                ax2 = ax1.twiny()
                mpf.plot(df_vp.tail(150 if VP_FREQ == "5" else 100),
                         type='candle', ax=ax1, volume=False)
                ax2.barh([(b.left + b.right)/2 for b in vprofile.index], vprofile.values,
                         height=(price_max-price_min)/VP_BINS * 0.8, alpha=0.3, color='dodgerblue')
                ax1.axhline(poc_price, color='red', lw=1.5, ls='--')
                ax2.set_xticks([])
                ax1.set_title(f"Volume Profile - {code}")

                fig.savefig(f_vp, dpi=120, bbox_inches='tight')
                plt.close(fig)

            print(f"  √ 已出图: {file_prefix}")

        except Exception as e:
            print(f"  × 错误: 处理 {code} 时发生异常 ({str(e)})")

        finally:
            plt.close('all')

    bs.logout()
    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] 🎉 批量任务全部完成！请去 '{OUTPUT_DIR}' 文件夹查收战果。")


if __name__ == '__main__':
    main_pipeline()
