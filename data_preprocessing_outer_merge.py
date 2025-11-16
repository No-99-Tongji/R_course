import pandas as pd
import os


def load_and_merge_specific_tables():
    """
    加载并合并特定的表格文件
    """
    # 假设您有以下表格文件
    tables_to_merge = [
        'csv_data/annual-co2-emissions-per-country.csv',
        'csv_data/carbon-emission-intensity-vs-gdp-per-capita.csv',
        'co2-by-source.csv',
        'co2-emissions-vs-gdp.csv',
        'life-expectancy-at-birth-vs-co2-emissions-per-capita.csv',
        'share-co2-embedded-in-trade.csv'
    ]

    # 存储所有数据框
    dfs = {}

    for table_file in tables_to_merge:
        if os.path.exists(table_file):
            try:
                df = pd.read_csv(table_file)
                # 提取表格名称（不含扩展名）
                table_name = os.path.splitext(table_file)[0]
                dfs[table_name] = df
                print(f"加载 {table_file}: {df.shape[0]} 行, {df.shape[1]} 列")
            except Exception as e:
                print(f"加载 {table_file} 失败: {e}")
        else:
            print(f"文件不存在: {table_file}")

    if not dfs:
        print("没有找到任何可用的表格文件")
        return None

    # 开始合并过程
    merged_df = None

    for name, df in dfs.items():
        if merged_df is None:
            merged_df = df
            print(f"\n以 {name} 作为基础表格")
        else:
            # 确定合并键（优先使用国家和年份）
            merge_on = ['Entity', 'Year'] if all(col in df.columns for col in ['Entity', 'Year']) else ['Entity']

            if all(col in merged_df.columns for col in merge_on):
                # 重命名重复的列（除了合并键）
                common_cols = set(merged_df.columns) & set(df.columns) - set(merge_on)
                rename_dict = {col: f"{col}_{name}" for col in common_cols if col not in merge_on}
                df_renamed = df.rename(columns=rename_dict)

                merged_df = pd.merge(merged_df, df_renamed,
                                     on=merge_on,
                                     how='outer',
                                     suffixes=('', f'_dup'))
                print(f"合并 {name}: 新增 {len(df)} 行可能数据")
            else:
                print(f"无法合并 {name}: 缺少共同的合并键")

    # 清理重复的列
    merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('_dup')]

    return merged_df


# 使用示例
if __name__ == "__main__":
    # 合并数据
    final_dataset = load_and_merge_specific_tables()

    if final_dataset is not None:
        print(f"\n最终数据集大小: {final_dataset.shape}")
        print(f"列名: {list(final_dataset.columns)}")

        # 保存结果
        final_dataset.to_csv('complete_merged_dataset.csv', index=False)
        print("合并完成! 数据已保存为 'complete_merged_dataset.csv'")

        # 显示数据样例
        print("\n数据样例:")
        print(final_dataset.head(10))

        # 统计信息
        print("\n基本统计:")
        print(f"包含国家数量: {final_dataset['Entity'].nunique()}")
        print(f"时间跨度: {final_dataset['Year'].min()} - {final_dataset['Year'].max()}")
        print(f"总数据点数: {len(final_dataset)}")