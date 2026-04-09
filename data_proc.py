import baostock as bs
import pandas as pd
from datetime import datetime, timedelta

# ==========================================
# 工具函数：自动推算起始时间
# ==========================================
def calculate_start_date(end_date_str, trading_days_needed):
    """
    根据给定的截止日期和需要的【交易日】数量，自动推算安全的【自然日】起始日期
    """
    # 1. 字符串转 datetime 对象
    end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    # 2. 交易日到自然日的膨胀换算 (乘以 1.46 并加上 20 天容错余量)
    # 这样可以确保跨越春节、国庆等长假时，依然能获取足够数量的 K 线
    calendar_days_to_subtract = int(trading_days_needed * 1.46) + 20
    
    # 3. 时间倒推
    start_date_obj = end_date_obj - timedelta(days=calendar_days_to_subtract)
    
    # 4. 转回 Baostock 识别的字符串格式
    return start_date_obj.strftime('%Y-%m-%d')

# ==========================================
# 核心数据接口：无缝接入时间引擎
# ==========================================
def fetch_data_auto(code="sh.600000", end_date=None, target_k_count=120):
    """
    全自动数据拉取函数
    :param code: 股票代码
    :param end_date: 截止日期 (若为 None，则自动使用今天真实日期)
    :param target_k_count: 你最终希望能拿到的有效 K 线数量 (默认 120 根)
    """
    # 如果未指定结束时间，自动获取系统当前日期
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
        
    # 调用时间引擎，生成起始时间
    start_date = calculate_start_date(end_date, target_k_count)
    
    print("-" * 40)
    print(f"目标请求: {code} 的最后 {target_k_count} 根 K线")
    print(f"时间引擎分配: 从 {start_date} 到 {end_date} (已包含节假日安全余量)")
    print("-" * 40)

    # 登录 Baostock
    bs.login()
    rs = bs.query_history_k_data_plus(
        code, "date,open,high,low,close,volume",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="3"
    )
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    
    if df.empty: 
        print("警告：获取到的数据为空，请检查代码或网络。")
        return df
        
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # 【最后一道清洗】：由于我们加了容错余量，获取的数据通常会比 target_k_count 多
    # 这里我们只精确截取用户需要的最后 N 根线，确保数据绝对纯净
    actual_rows = len(df)
    if actual_rows > target_k_count:
        df = df.tail(target_k_count)
    
    print(f"数据清洗完毕，最终实际返回数据量: {len(df)} 行")
    
    return df

# ==========================================
# 实战调用演示
# ==========================================
if __name__ == '__main__':
    # 场景 1：完全自动化（我要看浦发银行最近 120 个交易日的数据，截至今天）
    # 只要不传 end_date，它永远获取到你跑代码这一天的最新数据
    df_auto = fetch_data_auto(code="sh.600000", target_k_count=120)
    
    if not df_auto.empty:
        print("\n头部数据预览:")
        print(df_auto.head(3))
        print("\n尾部数据预览:")
        print(df_auto.tail(3))
        
    print("\n" + "="*50 + "\n")
    
    # 场景 2：历史回测模式（我想假装今天是 2023年6月1日，往前看 60 个交易日）
    # 用于复盘验证你的策略在过去某一天到底看到了什么图表
    df_history = fetch_data_auto(code="sh.600000", end_date="2023-06-01", target_k_count=60)