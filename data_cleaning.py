#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CO2数据集清洗工具
针对合并后的CSV文件进行数据清洗，处理空值过多的行和列
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_merged_dataset(filename='merged_co2_dataset.csv'):
    """
    加载合并后的数据集
    """
    filepath = Path(filename)
    if not filepath.exists():
        print(f"错误：文件 {filename} 不存在")
        return None

    try:
        print(f"加载数据集: {filename}")
        df = pd.read_csv(filepath)
        print(f"成功加载，形状: {df.shape}")
        return df
    except Exception as e:
        print(f"加载失败: {e}")
        return None

def analyze_missing_values(df):
    """
    分析缺失值情况
    """
    print("\n" + "="*60)
    print("缺失值分析")
    print("="*60)

    if df is None:
        return None

    # 基础列和数据列分类
    base_columns = ['Entity', 'Code', 'Year']
    data_columns = [col for col in df.columns if col not in base_columns]

    print(f"总行数: {len(df):,}")
    print(f"总列数: {len(df.columns)}")
    print(f"基础列: {len(base_columns)} 个")
    print(f"数据列: {len(data_columns)} 个")

    # 计算缺失值统计
    missing_stats = df.isnull().sum()
    missing_percent = (missing_stats / len(df) * 100).round(1)

    # 按缺失比例排序
    missing_df = pd.DataFrame({
        '列名': df.columns,
        '缺失数量': missing_stats.values,
        '缺失比例(%)': missing_percent.values,
        '类型': ['基础列' if col in base_columns else '数据列' for col in df.columns]
    }).sort_values('缺失比例(%)', ascending=False)

    print(f"\n缺失值最多的10列:")
    print(missing_df.head(10).to_string(index=False))

    # 统计高缺失列
    high_missing_cols = missing_df[missing_df['缺失比例(%)'] > 80]['列名'].tolist()
    very_high_missing_cols = missing_df[missing_df['缺失比例(%)'] > 95]['列名'].tolist()

    print(f"\n高缺失列 (>80%): {len(high_missing_cols)} 列")
    print(f"极高缺失列 (>95%): {len(very_high_missing_cols)} 列")

    # 按行分析缺失情况
    row_missing_counts = df[data_columns].isnull().sum(axis=1)
    row_missing_percent = (row_missing_counts / len(data_columns) * 100).round(1)

    print(f"\n行缺失值分布:")
    print(f"完全无数据的行: {(row_missing_percent == 100).sum():,}")
    print(f"缺失>90%的行: {(row_missing_percent > 90).sum():,}")
    print(f"缺失>70%的行: {(row_missing_percent > 70).sum():,}")
    print(f"缺失>50%的行: {(row_missing_percent > 50).sum():,}")

    return missing_df

def clean_columns(df, max_missing_ratio=0.85):
    """
    清洗列：删除缺失值过多的列
    """
    print(f"\n步骤2: 清洗列 (删除缺失值>{max_missing_ratio*100}%的列)")

    if df is None:
        return None

    base_columns = ['Entity', 'Code', 'Year']
    original_cols = len(df.columns)

    # 计算每列的缺失比例
    missing_ratio = df.isnull().sum() / len(df)

    # 找出需要删除的列（保护基础列）
    cols_to_drop = []
    for col in df.columns:
        if col not in base_columns and missing_ratio[col] > max_missing_ratio:
            cols_to_drop.append(col)

    if cols_to_drop:
        print(f"删除 {len(cols_to_drop)} 列:")
        for col in cols_to_drop:
            print(f"  - {col} (缺失率: {missing_ratio[col]:.1%})")
        df = df.drop(columns=cols_to_drop)
    else:
        print("没有需要删除的列")

    print(f"列数变化: {original_cols} -> {len(df.columns)}")
    return df

def clean_rows(df, min_data_columns=2, min_data_ratio=0.2):
    """
    清洗行：删除数据过少的行
    """
    print(f"\n步骤1: 清洗行 (保留至少{min_data_columns}列有数据或数据比例>{min_data_ratio*100}%的行)")

    if df is None:
        return None

    base_columns = ['Entity', 'Code', 'Year']
    data_columns = [col for col in df.columns if col not in base_columns]
    original_rows = len(df)

    if len(data_columns) == 0:
        print("警告: 没有数据列，跳过行清洗")
        return df

    # 计算每行的非空数据列数量
    non_null_counts = df[data_columns].notna().sum(axis=1)

    # 设定阈值：至少要有的数据列数量
    min_required = max(min_data_columns, int(len(data_columns) * min_data_ratio))

    # 筛选符合条件的行
    valid_rows = non_null_counts >= min_required
    df_cleaned = df[valid_rows]

    rows_removed = original_rows - len(df_cleaned)
    print(f"删除 {rows_removed:,} 行 (保留条件: 至少{min_required}列有数据)")
    print(f"行数变化: {original_rows:,} -> {len(df_cleaned):,}")

    return df_cleaned

def remove_duplicates(df):
    """
    删除重复行
    """
    print(f"\n步骤3: 删除重复行")

    if df is None:
        return None

    original_rows = len(df)
    df_dedup = df.drop_duplicates()
    duplicates_removed = original_rows - len(df_dedup)

    if duplicates_removed > 0:
        print(f"删除 {duplicates_removed:,} 个重复行")
    else:
        print("没有重复行")

    print(f"行数变化: {original_rows:,} -> {len(df_dedup):,}")
    return df_dedup

def sort_and_organize(df):
    """
    排序和整理数据
    """
    print(f"\n步骤4: 排序和整理")

    if df is None:
        return None

    # 按Entity和Year排序
    if 'Entity' in df.columns and 'Year' in df.columns:
        df = df.sort_values(['Entity', 'Year'])
        print(f"按Entity和Year排序完成")

    # 重置索引
    df = df.reset_index(drop=True)
    print(f"重置索引完成")

    return df

def generate_quality_report(df_original, df_cleaned):
    """
    生成数据质量报告
    """
    print("\n" + "="*60)
    print("数据质量报告")
    print("="*60)

    if df_original is None or df_cleaned is None:
        print("无法生成报告：数据为空")
        return

    base_columns = ['Entity', 'Code', 'Year']

    print(f"原始数据集:")
    print(f"  - 行数: {len(df_original):,}")
    print(f"  - 列数: {len(df_original.columns)}")

    print(f"\n清洗后数据集:")
    print(f"  - 行数: {len(df_cleaned):,}")
    print(f"  - 列数: {len(df_cleaned.columns)}")

    print(f"\n数据保留率:")
    print(f"  - 行保留率: {len(df_cleaned)/len(df_original)*100:.1f}%")
    print(f"  - 列保留率: {len(df_cleaned.columns)/len(df_original.columns)*100:.1f}%")

    # 计算数据完整性
    if len(df_cleaned) > 0:
        data_columns = [col for col in df_cleaned.columns if col not in base_columns]
        if data_columns:
            total_cells = len(df_cleaned) * len(data_columns)
            non_null_cells = df_cleaned[data_columns].notna().sum().sum()
            completeness = non_null_cells / total_cells * 100
            print(f"  - 数据完整性: {completeness:.1f}%")

    # 实体统计
    if 'Entity' in df_cleaned.columns:
        print(f"\n实体统计:")
        print(f"  - 唯一实体数: {df_cleaned['Entity'].nunique()}")

        if 'Year' in df_cleaned.columns:
            year_range = df_cleaned['Year'].dropna()
            if not year_range.empty:
                print(f"  - 年份范围: {int(year_range.min())} - {int(year_range.max())}")

    # 显示缺失值最少的前10列
    if len(df_cleaned.columns) > len(base_columns):
        data_cols = [col for col in df_cleaned.columns if col not in base_columns]
        missing_stats = df_cleaned[data_cols].isnull().sum()
        missing_percent = (missing_stats / len(df_cleaned) * 100).round(1)

        best_cols = missing_percent.sort_values().head(10)
        print(f"\n数据质量最好的10列:")
        for col, pct in best_cols.items():
            print(f"  - {col}: {pct}% 缺失")

def save_cleaned_dataset(df, filename='initially_processed_data/cleaned_co2_dataset.csv'):
    """
    保存清洗后的数据集
    """
    if df is None:
        print("无数据可保存")
        return False

    try:
        # 确保输出目录存在
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n保存清洗后的数据集到: {filename}")
        df.to_csv(filename, index=False, encoding='utf-8')
        file_size = Path(filename).stat().st_size / 1024 / 1024
        print(f"保存成功！文件大小: {file_size:.1f} MB")

        # 显示保存的数据集基本信息
        print(f"保存的数据集信息:")
        print(f"  - 行数: {len(df):,}")
        print(f"  - 列数: {len(df.columns)}")
        print(f"  - 文件: {filename}")

        return True
    except Exception as e:
        print(f"保存失败: {e}")
        return False

def clean_co2_dataset(input_file='merged_co2_dataset.csv',
                     output_file='initially_processed_data/cleaned_co2_dataset.csv',
                     max_missing_ratio=0.85,
                     min_data_columns=2,
                     min_data_ratio=0.2):
    """
    完整的数据清洗流程

    参数:
    - input_file: 输入文件名
    - output_file: 输出文件名
    - max_missing_ratio: 列的最大缺失比例阈值
    - min_data_columns: 行的最少数据列数量
    - min_data_ratio: 行的最小数据比例
    """
    print("开始CO2数据集清洗...")
    print("="*60)

    # 1. 加载数据
    df_original = load_merged_dataset(input_file)
    if df_original is None:
        return None

    # 2. 分析缺失值
    missing_analysis = analyze_missing_values(df_original)

    # 3. 开始清洗
    print(f"\n开始数据清洗流程...")
    print(f"清洗参数:")
    print(f"  - 列缺失率阈值: {max_missing_ratio*100}%")
    print(f"  - 行最少数据列: {min_data_columns}")
    print(f"  - 行最小数据比例: {min_data_ratio*100}%")

    df_cleaned = df_original.copy()

    # 步骤1: 清洗行 (先删除质量不好的行)
    df_cleaned = clean_rows(df_cleaned, min_data_columns, min_data_ratio)

    # 步骤2: 清洗列 (基于清洗后的行重新评估列质量)
    df_cleaned = clean_columns(df_cleaned, max_missing_ratio)

    # 步骤3: 删除重复行
    df_cleaned = remove_duplicates(df_cleaned)

    # 步骤4: 排序整理
    df_cleaned = sort_and_organize(df_cleaned)

    # 5. 生成质量报告
    generate_quality_report(df_original, df_cleaned)

    # 6. 保存清洗后的数据
    success = save_cleaned_dataset(df_cleaned, output_file)

    if success:
        print("\n" + "="*60)
        print("数据清洗完成！")
        print("="*60)
        print(f"原始文件: {input_file}")
        print(f"清洗文件: {output_file}")

    return df_cleaned

def main():
    """
    主函数 - 提供多种清洗选项
    """
    print("CO2数据集清洗工具")
    print("="*60)

    # 选项1: 标准清洗 (删除>85%缺失的列，保留至少2列数据的行)
    print("\n选项1: 标准清洗")
    cleaned_standard = clean_co2_dataset(
        input_file='merged_co2_dataset.csv',
        output_file='initially_processed_data/cleaned_co2_dataset_standard.csv',
        max_missing_ratio=0.85,
        min_data_columns=2,
        min_data_ratio=0.2
    )

    # 选项2: 严格清洗 (删除>70%缺失的列，保留至少3列数据的行)
    print("\n" + "="*60)
    print("\n选项2: 严格清洗")
    cleaned_strict = clean_co2_dataset(
        input_file='merged_co2_dataset.csv',
        output_file='initially_processed_data/cleaned_co2_dataset_strict.csv',
        max_missing_ratio=0.70,
        min_data_columns=3,
        min_data_ratio=0.3
    )

    # 选项3: 宽松清洗 (删除>95%缺失的列，保留至少1列数据的行)
    print("\n" + "="*60)
    print("\n选项3: 宽松清洗")
    cleaned_loose = clean_co2_dataset(
        input_file='merged_co2_dataset.csv',
        output_file='initially_processed_data/cleaned_co2_dataset_loose.csv',
        max_missing_ratio=0.95,
        min_data_columns=1,
        min_data_ratio=0.1
    )

    print("\n" + "="*60)
    print("所有清洗任务完成！")
    print("="*60)
    print("生成的文件:")
    print("- initially_processed_data/cleaned_co2_dataset_standard.csv (推荐)")
    print("- initially_processed_data/cleaned_co2_dataset_strict.csv")
    print("- initially_processed_data/cleaned_co2_dataset_loose.csv")

if __name__ == "__main__":
    main()
