import os
import time
import baostock as bs
import pandas as pd
from datetime import datetime, timedelta

# --- 配置区 ---
DATA_DIR = "zz800_parquet_data" # 本地数据库文件夹
START_DATE_INIT = "2020-01-01"  # 如果没有本地数据，默认拉取的起点
# 需要转为数值类型的列，提前定义好，防止类型污染
NUMERIC_COLS = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'pctChg', 'pbMRQ', 'peTTM', 'psTTM', 'turn']

def init_environment():
    """初始化目录"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 创建本地数据仓库: {DATA_DIR}")

def get_zz800_codes():
    """获取最新的中证800成分股"""
    print("⏳ 正在获取中证800成分股名单...")
    hs300 = bs.query_hs300_stocks().get_data()
    zz500 = bs.query_zz500_stocks().get_data()
    
    # 合并并去重
    zz800_df = pd.concat([hs300, zz500])
    codes = zz800_df['code'].unique().tolist()
    print(f"✅ 成功获取 {len(codes)} 只成分股。")
    return codes

def update_single_stock(code, today_str):
    """更新单只股票的数据并保存为 Parquet"""
    file_path = f"{DATA_DIR}/{code}.parquet"
    
    # 1. 判断是全量还是增量
    if os.path.exists(file_path):
        # Parquet 读取速度是毫秒级的
        df_local = pd.read_parquet(file_path)
        last_date_str = df_local['date'].max()
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        is_incremental = True
    else:
        df_local = pd.DataFrame()
        start_date = START_DATE_INIT
        is_incremental = False
        
    # 2. 如果本地已经是最新，直接跳过
    if start_date > today_str:
        return 0, "已是最新"

    # 3. 向 BaoStock 请求缺失的数据
    rs = bs.query_history_k_data_plus(
        code, 
        "date,code,open,high,low,close,preclose,volume,amount,pctChg,pbMRQ,peTTM,psTTM,turn",
        start_date=start_date, end_date=today_str, 
        frequency="d", adjustflag="2" # 前复权
    )
    
    if rs.error_code == '0' and rs.next():
        df_new = rs.get_data()
        
        # ⚠️ 核心防御：严格转换数据类型！这在量化中极其重要
        for col in NUMERIC_COLS:
            if col in df_new.columns:
                df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
        
        # 4. 合并数据
        if is_incremental:
            df_combined = pd.concat([df_local, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset=['date'], keep='last', inplace=True) # 去重保险
            status = f"增量 +{len(df_new)}条"
        else:
            df_combined = df_new
            status = f"初始化 {len(df_new)}条"
            
        # 5. 覆盖保存为 Parquet 格式
        # 存为 Parquet 文件，比 CSV 小得多，下次读进来瞬间恢复所有 float64 类型
        df_combined.to_parquet(file_path, engine='pyarrow', index=False)
        return len(df_new), status
    else:
        return 0, "停牌或无数据"


def run_data_engine():
    """主引擎 (升级版：加入工业级网络防断线重试机制)"""
    print("\n🔌 正在尝试连接 BaoStock 服务器...")

    # ✨ 核心修复：重试机制 (最多尝试 3 次)
    login_success = False
    for attempt in range(3):
        try:
            # 尝试登录
            lg = bs.login()
            # BaoStock 的 error_code 为 '0' 才是真正成功
            if lg.error_code == '0':
                print("✅ BaoStock 登录成功！")
                login_success = True
                break
            else:
                print(f"⚠️ 登录失败，错误信息: {lg.error_msg} (尝试 {attempt+1}/3)")
        except Exception as e:
            print(f"⚠️ 网络 Socket 异常: {e} (尝试 {attempt+1}/3)")

        # 如果失败了，不要立刻猛烈请求，休息 5 秒钟等服务器喘口气
        print("⏳ 等待 5 秒后进行下一次重连...")
        time.sleep(5)

    # 如果 3 次都失败了，抛出严重异常，让 main.py 里的 try-except 捕获并熔断
    if not login_success:
        raise ConnectionError("❌ BaoStock 服务器彻底无响应，可能是官方在维护，请半小时后再试！")

    # 登录成功后，继续执行原来的逻辑
    init_environment()

    codes = get_zz800_codes()
    today_str = datetime.now().strftime("%Y-%m-%d")

    print("\n🚀 开始执行中证800本地数据同步引擎...")
    start_time = time.time()

    updated_count = 0
    total_new_rows = 0

    # 遍历更新
    for i, code in enumerate(codes):
        new_rows, status = update_single_stock(code, today_str)
        if new_rows > 0:
            updated_count += 1
            total_new_rows += new_rows

        # 打印简易进度条
        if (i + 1) % 50 == 0 or i == len(codes) - 1:
            print(f"进度: {i+1}/{len(codes)} | 最近处理: {code} [{status}]")

    end_time = time.time()
    bs.logout()

    print("\n" + "="*40)
    print("🎉 同步任务完美收官！")
    print(f"⏱️ 耗时: {end_time - start_time:.2f} 秒")
    print(f"📈 本次发生更新的股票数: {updated_count} 只")
    print(f"📥 累计向本地注入新数据: {total_new_rows} 行")
    print("="*40)

if __name__ == "__main__":
    run_data_engine()