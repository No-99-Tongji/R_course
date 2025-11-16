#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理：将多个CO2排放相关的CSV文件合并成一个数据集
使用Entity、Code和Year作为复合主键进行外连接
"""

import pandas as pd
from pathlib import Path

def load_and_merge_datasets():
    """
    加载所有CSV文件并合并成一个数据集
    """
    # 定义数据文件夹路径
    csv_dir = Path('./csv_data')

    # 定义文件列表及其对应的简化列名前缀
    files_info = {
        'annual-co2-emissions-per-country.csv': 'annual_co2',
        'carbon-emission-intensity-vs-gdp-per-capita.csv': 'carbon_intensity',
        'co2-by-source.csv': 'co2_source',
        'co2-emissions-vs-gdp.csv': 'co2_gdp',
        'life-expectancy-at-birth-vs-co-emissions-per-capita.csv': 'life_expectancy',
        'share-co2-embedded-in-trade.csv': 'co2_trade'
    }

    # 存储所有数据集
    datasets = {}

    print("正在加载数据文件...")

    for filename, prefix in files_info.items():
        filepath = csv_dir / filename
        if filepath.exists():
            print(f"加载文件: {filename}")
            df = pd.read_csv(filepath)

            # 清理列名，移除特殊字符和长名称
            df.columns = df.columns.str.strip()

            # 重命名列名，保持Entity、Code、Year不变，其他列加上前缀
            new_columns = {}
            for col in df.columns:
                if col in ['Entity', 'Code', 'Year']:
                    new_columns[col] = col
                else:
                    # 简化列名
                    simplified_col = col.replace('Annual CO₂ emissions', 'co2_emissions')\
                                       .replace('Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)', 'co2_per_capita')\
                                       .replace('Life expectancy - Sex: all - Age: 0 - Variant: estimates', 'life_expectancy')\
                                       .replace('Carbon intensity of GDP (kg CO2e per 2021 PPP $)', 'carbon_intensity_gdp')\
                                       .replace('GDP per capita, PPP (constant 2021 international $)', 'gdp_per_capita_ppp')\
                                       .replace('GDP per capita', 'gdp_per_capita')\
                                       .replace('Population (historical)', 'population')\
                                       .replace('World regions according to OWID', 'world_region')\
                                       .replace('Share of annual CO₂ emissions embedded in trade', 'co2_trade_share')\
                                       .replace('900793-annotations', 'annotations')

                    # 移除特殊字符并替换空格为下划线
                    simplified_col = simplified_col.replace('₂', '2')\
                                                   .replace('（', '(')\
                                                   .replace('）', ')')\
                                                   .replace(' ', '_')\
                                                   .replace('-', '_')\
                                                   .replace(',', '')\
                                                   .replace('(', '')\
                                                   .replace(')', '')

                    # 如果不是基本列，添加前缀以避免冲突
                    if simplified_col not in ['Entity', 'Code', 'Year']:
                        new_columns[col] = f"{prefix}_{simplified_col}"
                    else:
                        new_columns[col] = simplified_col

            df = df.rename(columns=new_columns)
            datasets[prefix] = df
            print(f"  - 形状: {df.shape}")
            print(f"  - 列名: {list(df.columns)}")
        else:
            print(f"警告: 文件不存在 {filepath}")

    # 开始合并数据集
    print("\n开始合并数据集...")

    # 从第一个数据集开始
    merged_df = None
    merge_keys = ['Entity', 'Code', 'Year']

    for i, (prefix, df) in enumerate(datasets.items()):
        if merged_df is None:
            merged_df = df.copy()
            print(f"初始数据集 ({prefix}): {merged_df.shape}")
        else:
            # 确保merge_keys在当前数据集中存在
            available_keys = [key for key in merge_keys if key in df.columns]

            if len(available_keys) >= 2:  # 至少需要Entity和Year或Entity和Code
                print(f"合并数据集 ({prefix}) 使用键: {available_keys}")
                merged_df = pd.merge(merged_df, df, on=available_keys, how='outer', suffixes=('', f'_{prefix}'))
                print(f"合并后形状: {merged_df.shape}")
            else:
                print(f"警告: 数据集 {prefix} 缺少足够的合并键，跳过")

    return merged_df

def save_merged_dataset(df, output_filename='merged_co2_dataset.csv'):
    """
    保存合并后的数据集
    """
    print(f"\n保存合并数据集到: {output_filename}")
    df.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"保存完成！")

def analyze_merged_dataset(df):
    """
    分析合并后的数据集
    """
    print("\n=== 合并数据集分析 ===")
    print(f"总行数: {len(df)}")
    print(f"总列数: {len(df.columns)}")

    print(f"\n列名:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")

    print(f"\n缺失值统计:")
    missing_stats = df.isnull().sum()
    missing_percent = (missing_stats / len(df) * 100).round(2)

    for col, missing_count in missing_stats.items():
        if missing_count > 0:
            print(f"  {col}: {missing_count} ({missing_percent[col]}%)")

    print(f"\n数据类型:")
    print(df.dtypes)

    print(f"\n前5行数据:")
    print(df.head())

    print(f"\n唯一国家/地区数量: {df['Entity'].nunique()}")
    print(f"年份范围: {df['Year'].min()} - {df['Year'].max()}")

def main():
    """
    主函数
    """
    print("开始数据预处理和合并...")

    # 加载并合并数据集
    merged_df = load_and_merge_datasets()

    if merged_df is not None:
        # 分析合并后的数据集
        analyze_merged_dataset(merged_df)

        # 保存合并后的数据集
        save_merged_dataset(merged_df)

        print("\n数据处理完成！")
        return merged_df
    else:
        print("错误: 无法合并数据集")
        return None

if __name__ == "__main__":
    merged_dataset = main()
