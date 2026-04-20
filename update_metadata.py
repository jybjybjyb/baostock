# update_metadata.py
import os
import glob
import pandas as pd
import baostock as bs

DATA_DIR = "zz800_parquet_data"
META_FILE = "zz800_metadata.csv"


def update_local_metadata():
    print(f"📂 1. 正在扫描本地 Parquet 仓库提取股票池...")
    all_files = glob.glob(f"{DATA_DIR}/*.parquet")
    if not all_files:
        raise ValueError("本地仓库为空，请先下载 K 线数据！")

    # 从本地文件名或文件内容中提取所有的股票代码
    # 假设你的文件是按股票存的，或者直接读一点数据提取 unique codes
    codes = set()
    for f in all_files:
        # 为了极速，我们只读每个文件的第一行来获取代码
        df_tmp = pd.read_parquet(f, columns=['code'])
        if not df_tmp.empty:
            codes.add(df_tmp['code'].iloc[0])

    unique_codes = list(codes)
    print(f"🔍 发现 {len(unique_codes)} 只独立股票，准备向 BaoStock 获取户籍信息...")

    print("🌐 2. 正在联网获取名称与申万行业标签 (过程较慢，请耐心等待)...")
    bs.login()
    meta_data = []
    for i, code in enumerate(unique_codes):
        rs = bs.query_stock_industry(code)
        if rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            meta_data.append({
                'code': code,
                'name': row[2],      # 股票名称
                'Industry': row[3]   # 所属行业
            })
        if (i + 1) % 100 == 0:
            print(f"  > 已拉取 {i + 1}/{len(unique_codes)} 只...")

    bs.logout()

    # 3. 固化到本地 CSV
    df_meta = pd.DataFrame(meta_data)
    # 使用 utf-8-sig 防止 Excel 打开乱码
    df_meta.to_csv(META_FILE, index=False, encoding='utf-8-sig')
    print(f"\n🎉 户籍登记完成！全市场元数据已永久固化至: {META_FILE}")


if __name__ == "__main__":
    update_local_metadata()
