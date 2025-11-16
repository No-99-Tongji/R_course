#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CO2数据集合并工具
将多个CSV文件通过Entity、Code和Year作为复合主键进行外连接合并
保留所有数据，缺失的值用NaN填充
"""

import pandas as pd
from pathlib import Path

def merge_co2_datasets():
    """
    合并所有CO2相关的CSV文件
    """
    # 定义CSV文件目录
    csv_dir = Path('./raw_data')

    # 获取所有CSV文件
    csv_files = list(csv_dir.glob('*.csv'))

    if not csv_files:
        print("错误：未找到CSV文件")
        return None

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 读取并合并所有文件
    merged_df = None

    for i, csv_file in enumerate(csv_files):
        print(f"\n处理文件 {i+1}/{len(csv_files)}: {csv_file.name}")

        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            print(f"  - 读取成功，形状: {df.shape}")
            print(f"  - 列名: {list(df.columns)}")

            # 确定合并键
            merge_keys = []
            if 'Entity' in df.columns:
                merge_keys.append('Entity')
            if 'Code' in df.columns:
                merge_keys.append('Code')
            if 'Year' in df.columns:
                merge_keys.append('Year')

            print(f"  - 合并键: {merge_keys}")

            # 第一个文件作为基础
            if merged_df is None:
                merged_df = df.copy()
                print(f"  - 作为基础数据集")
            else:
                # 与之前的数据合并
                if len(merge_keys) > 0:
                    # 检查是否有重复的列名（除了合并键）
                    overlapping_cols = set(merged_df.columns) & set(df.columns) - set(merge_keys)
                    if overlapping_cols:
                        print(f"  - 发现重复列名: {overlapping_cols}")
                        # 为重复列添加文件名后缀
                        suffix = '_' + csv_file.stem.replace('-', '_')
                        merged_df = pd.merge(merged_df, df, on=merge_keys, how='outer', suffixes=('', suffix))
                    else:
                        merged_df = pd.merge(merged_df, df, on=merge_keys, how='outer')

                    print(f"  - 合并完成，新形状: {merged_df.shape}")
                else:
                    print(f"  - 警告：无法找到合适的合并键，跳过此文件")

        except Exception as e:
            print(f"  - 错误：读取文件失败 - {e}")
            continue

    # 过滤年份，只保留1700年及以后的数据
    if merged_df is not None and 'Year' in merged_df.columns:
        print(f"\n应用年份过滤器...")
        original_rows = len(merged_df)
        merged_df = merged_df[merged_df['Year'] >= 1700]
        filtered_rows = len(merged_df)
        print(f"  - 原始行数: {original_rows:,}")
        print(f"  - 过滤后行数: {filtered_rows:,}")
        print(f"  - 删除行数: {original_rows - filtered_rows:,}")

        if filtered_rows > 0:
            year_range = merged_df['Year'].dropna()
            if not year_range.empty:
                print(f"  - 新的年份范围: {int(year_range.min())} - {int(year_range.max())}")

    return merged_df

def analyze_dataset(df):
    """
    分析合并后的数据集
    """
    if df is None:
        print("无数据可分析")
        return

    print("\n" + "="*50)
    print("数据集分析")
    print("="*50)

    print(f"总行数: {len(df):,}")
    print(f"总列数: {len(df.columns)}")

    # 显示所有列名
    print(f"\n所有列名 ({len(df.columns)} 列):")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")

    # 基本统计信息
    print(f"\n基本信息:")
    if 'Entity' in df.columns:
        print(f"  - 唯一实体数量: {df['Entity'].nunique()}")
        print(f"  - 实体示例: {list(df['Entity'].dropna().unique()[:5])}")

    if 'Year' in df.columns:
        year_range = df['Year'].dropna()
        if not year_range.empty:
            print(f"  - 年份范围: {int(year_range.min())} - {int(year_range.max())}")

    # 缺失值统计
    print(f"\n缺失值统计:")
    missing_stats = df.isnull().sum()
    missing_percent = (missing_stats / len(df) * 100).round(1)

    print(f"{'列名':<50} {'缺失数量':<10} {'缺失比例'}")
    print("-" * 70)
    for col in df.columns:
        missing_count = missing_stats[col]
        missing_pct = missing_percent[col]
        print(f"{col:<50} {missing_count:<10} {missing_pct}%")

    # 显示前几行数据
    print(f"\n前5行数据:")
    print(df.head())

def save_dataset(df, filename='merged_co2_dataset.csv'):
    """
    保存合并后的数据集
    """
    if df is None:
        print("无数据可保存")
        return False

    try:
        print(f"\n保存数据集到: {filename}")
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"保存成功！文件大小: {Path(filename).stat().st_size / 1024 / 1024:.1f} MB")
        return True
    except Exception as e:
        print(f"保存失败: {e}")
        return False

def main():
    """
    主函数
    """
    print("开始合并CO2数据集...")
    print("="*50)

    # 合并数据集
    merged_df = merge_co2_datasets()

    if merged_df is not None:
        # 分析数据集
        analyze_dataset(merged_df)

        # 保存数据集
        save_dataset(merged_df)

        print("\n" + "="*50)
        print("数据合并完成！")
        print("="*50)

        return merged_df
    else:
        print("合并失败！")
        return None

if __name__ == "__main__":
    result = main()
