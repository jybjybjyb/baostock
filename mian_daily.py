# ==========================================
# 量化帝国中央控制台 (Daily Pipeline)
# ==========================================
import os
import time
import pickle
import pandas as pd
from datetime import datetime

import local_data_engine as data_engine
import market_radar as radar
import auto_portfolio_sorter as sorter
import warnings

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    print("="*65)
    print("🌟 启动量化交易每日自动化流水线 (Daily Pipeline) 🌟")
    print("="*65)

    pipeline_start_t = time.time()
    today_str = datetime.now().strftime("%Y-%m-%d")

    # [步骤 1/4] 启动底层数据引擎
    print("\n>>> [步骤 1/4] 启动底层数据引擎 (Data Engine)...")
    try:
        data_engine.run_data_engine()
    except Exception as e:
        print(f"❌ 数据同步失败。原因: {e}")
        exit()

    # [步骤 2/4] 启动市场风向雷达与板块时序分析
    print("\n>>> [步骤 2/4] 启动市场风向雷达 (Market Radar)...")
    try:
        # 1. 执行原有的截面归因
        df_radar_raw = radar.fast_radar_calculation()
        df_meta = radar.load_local_metadata()
        df_final_radar = pd.merge(
            df_radar_raw, df_meta, on='code', how='inner').dropna()

        ols_report = radar.run_ols_radar(df_final_radar)
        radar_factors = ['Momentum', 'Short_Rev', 'Low_Vol', 'Liquidity', 'Size', 'Value_BP',
                         'Mom_Sharpe', 'Vol_Price_Corr', 'Amihud']
        drill_report = radar.drill_down_industry_leaders(
            df_final_radar, radar_factors, top_n_ind=5, top_n_stock=15)

        # ✨ 新增：2. 执行板块轮动时序透视
        sector_pivot, sector_rank = radar.analyze_sector_trends(lookback=5)

    except Exception as e:
        print(f"❌ 雷达扫描失败。原因: {e}")
        exit()

    # [步骤 3/4] 底仓洗牌与 ZZ800 全市场排名
    print("\n>>> [步骤 3/4] 启动底仓自动洗牌机与全市场扫描器...")
    try:
        df_radar_sorter, df_latest = sorter.get_market_data_and_factors()

        # ✨ 改进：用三个变量接住王冠、次王冠和战绩表
        king_factor, sub_factor, king_stats = sorter.find_king_factor(
            df_radar_sorter)

        # 1. 刷新你的个人底仓 list.csv
        df_top_picks = sorter.update_and_sort_list(df_latest, king_factor)

        # ✨ 新增：2. 输出全市场 ZZ800 的 Top 100 猎物池
        df_zz800_top100 = sorter.rank_zz800_top100(
            df_latest, king_factor, sub_factor)

        # 从最新的洗牌数据中，提取真实的 T0 截面日期
        actual_t0 = pd.to_datetime(
            df_latest['date'].iloc[0]).strftime("%Y-%m-%d")

    except Exception as e:
        print(f"❌ 底仓洗牌失败。原因: {e}")
        exit()

    # [步骤 4/4] 每日战报自动归档为 PKL
    print("\n>>> [步骤 4/4] 正在生成并打包今日战报 (Daily Snapshot)...")

    if actual_t0 != today_str:
        print("\n" + "!"*65)
        print(f"🚨 【时序防呆拦截触发】检测到底层数据时间滞后！(实际底层数据: {actual_t0})")
        print("!"*65)

    try:
        os.makedirs("Daily_Reports", exist_ok=True)

        # ✨ 数据字典全面扩容，将板块分析和 Top100 全都装进战报
        daily_snapshot = {
            "date": actual_t0,
            "king_factor": king_factor,
            "sub_factor": sub_factor,         # 新增次强因子
            "king_stats": king_stats,
            "ols_report": ols_report,
            "drill_report": drill_report,
            "sector_pivot": sector_pivot,     # 新增板块动量表
            "sector_rank": sector_rank,       # 新增板块排名变动
            "top_picks": df_top_picks,
            "zz800_top100": df_zz800_top100,  # 新增全市场 Top100
            "radar_data": df_final_radar
        }

        report_file = f"Daily_Reports/Report_{actual_t0}.pkl"
        with open(report_file, 'wb') as f:
            pickle.dump(daily_snapshot, f)

        print(f"📦 战报已加密归档至: {report_file}")
    except Exception as e:
        print(f"❌ 战报归档失败。原因: {e}")

    # ==========================================
    # 收尾汇报
    # ==========================================
    print("\n" + "="*65)
    print(f"✅ 每日流水线全部执行完毕！全流程总耗时: {time.time() - pipeline_start_t:.2f} 秒")
    print(f"👉 【截面基准 {actual_t0}】主线风口: 【{king_factor}】 | 辅助风口: 【{sub_factor}】")
    print(f"📁 请检查生成的个人底仓 (list.csv) 及全市场金股池 (zz800_top100.csv)！")
    print("="*65)
