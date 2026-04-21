# ==========================================
# 量化帝国中央控制台 (Daily Pipeline)
# 运行频率：每天收盘后 15:30 运行一次
# 升级：新增每日战报 PKL 自动归档功能
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

    # [步骤 2/4] 启动市场风向雷达
    print("\n>>> [步骤 2/4] 启动市场风向雷达 (Market Radar)...")
    try:
        df_radar_raw = radar.fast_radar_calculation()
        df_meta = radar.load_local_metadata()
        df_final_radar = pd.merge(
            df_radar_raw, df_meta, on='code', how='inner').dropna()

        radar.run_ols_radar(df_final_radar)
        radar_factors = ['Momentum', 'Short_Rev',
                         'Low_Vol', 'Liquidity', 'Size', 'Value_BP']
        radar.drill_down_industry_leaders(
            df_final_radar, radar_factors, top_n_ind=3, top_n_stock=10)
    except Exception as e:
        print(f"❌ 雷达扫描失败。原因: {e}")
        exit()

    # [步骤 3/4] 底仓自动打分与洗牌
    print("\n>>> [步骤 3/4] 启动底仓自动洗牌机 (Portfolio Sorter)...")
    try:
        df_radar_sorter, df_latest = sorter.get_market_data_and_factors()
        king_factor = sorter.find_king_factor(df_radar_sorter)
        # ✨ 接收洗好的底仓数据
        df_top_picks = sorter.update_and_sort_list(df_latest, king_factor)
    except Exception as e:
        print(f"❌ 底仓洗牌失败。原因: {e}")
        exit()

    # ✨ [步骤 4/4] 每日战报自动归档为 PKL
    print("\n>>> [步骤 4/4] 正在生成并打包今日战报 (Daily Snapshot)...")
    try:
        os.makedirs("Daily_Reports", exist_ok=True)

        # 将我们最关心的核心数据装进一个字典
        daily_snapshot = {
            "date": today_str,
            "king_factor": king_factor,
            "radar_data": df_final_radar,   # 全市场的雷达探测底稿
            "top_picks": df_top_picks       # 洗牌后的最终金股名单
        }

        report_file = f"Daily_Reports/Report_{today_str}.pkl"
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
    print(f"👉 今日主线风口: 【{king_factor}】 | 请检查 list.csv 挂单明天的狙击标的！")
    print("="*65)
