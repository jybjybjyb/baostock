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

    # [步骤 2/4] 启动市场风向雷达 (修改部分)
    print("\n>>> [步骤 2/4] 启动市场风向雷达 (Market Radar)...")
    try:
        df_radar_raw = radar.fast_radar_calculation()
        df_meta = radar.load_local_metadata()
        df_final_radar = pd.merge(
            df_radar_raw, df_meta, on='code', how='inner').dropna()

        # ✨ 核心改动：用变量接住报告结果
        ols_report = radar.run_ols_radar(df_final_radar)
        radar_factors = ['Momentum', 'Short_Rev', 'Low_Vol', 'Liquidity', 'Size', 'Value_BP',
        'Mom_Sharpe', 'Vol_Price_Corr', 'Amihud']
        drill_report = radar.drill_down_industry_leaders(
            df_final_radar, radar_factors, top_n_ind=5, top_n_stock=15)    # 提示：改完这里后，明天生成的战报 PKL 里，强势板块画像就会自动变多。
    except Exception as e:
        print(f"❌ 雷达扫描失败。原因: {e}")
        exit()
    
    # [步骤 3/4] 底仓自动打分与洗牌
    print("\n>>> [步骤 3/4] 启动底仓自动洗牌机 (Portfolio Sorter)...")
    try:
        df_radar_sorter, df_latest = sorter.get_market_data_and_factors()
        king_factor, king_stats = sorter.find_king_factor(df_radar_sorter)
        df_top_picks = sorter.update_and_sort_list(df_latest, king_factor)
        
        # ✨ 核心抓取：从最新的洗牌数据中，提取真实的 T0 截面日期
        actual_t0 = pd.to_datetime(df_latest['date'].iloc[0]).strftime("%Y-%m-%d")
        
    except Exception as e:
        print(f"❌ 底仓洗牌失败。原因: {e}")
        exit()

    # [步骤 4/4] 每日战报自动归档为 PKL
    print("\n>>> [步骤 4/4] 正在生成并打包今日战报 (Daily Snapshot)...")
    
    # ✨ 显式纠错：时序防呆熔断机制
    if actual_t0 != today_str:
        print("\n" + "!"*65)
        print(f"🚨 【时序防呆拦截触发】检测到底层数据时间滞后！")
        print(f"   > 你的系统日历: {today_str}")
        print(f"   > 实际底层数据: {actual_t0}")
        print(f"   (注: 若今天是周末则属正常；若是工作日，说明数据源尚未更新)")
        print(f"   🛡️ 纠错动作 1：拒绝生成名为 {today_str} 的虚假战报。")
        print(f"   🛡️ 纠错动作 2：强制将本次结果归档为 Report_{actual_t0}.pkl。")
        print("!"*65)

    try:
        os.makedirs("Daily_Reports", exist_ok=True)
        
        # 将我们最关心的核心数据装进一个字典
        daily_snapshot = {
            "date": actual_t0,            # ⚠️ 强行修正为实际数据日期
            "king_factor": king_factor,
            "king_stats": king_stats,     
            "ols_report": ols_report,
            "drill_report": drill_report,
            "top_picks": df_top_picks,
            "radar_data": df_final_radar  
        }
        
        # ⚠️ 文件名强行由实际数据日期决定
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
    # 打印时也提示真实日期
    print(f"👉 【基于 {actual_t0} 截面】主线风口: 【{king_factor}】 | 请检查 list.csv！")
    print("="*65)
