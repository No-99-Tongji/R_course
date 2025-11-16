#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CO2数据集完整处理工具
包含清洗 + 多种缺失值填充方法，生成不同版本的数据集
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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
        df = pd.read_csv(filepath, low_memory=False)
        print(f"成功加载，形状: {df.shape}")
        return df
    except Exception as e:
        print(f"加载失败: {e}")
        return None

def basic_clean(df, max_missing_ratio=0.85, min_data_columns=2, min_data_ratio=0.2):
    """
    基础清洗：先清洗行，再清洗列
    """
    print(f"\n基础清洗 (列缺失率阈值: {max_missing_ratio*100}%, 行最少数据列: {min_data_columns})")

    if df is None:
        return None

    base_columns = ['Entity', 'Code', 'Year']
    data_columns = [col for col in df.columns if col not in base_columns]

    # 步骤1: 清洗行
    original_rows = len(df)
    if len(data_columns) > 0:
        non_null_counts = df[data_columns].notna().sum(axis=1)
        min_required = max(min_data_columns, int(len(data_columns) * min_data_ratio))
        valid_rows = non_null_counts >= min_required
        df_cleaned = df[valid_rows]
        rows_removed = original_rows - len(df_cleaned)
        print(f"  - 删除 {rows_removed:,} 行 (保留条件: 至少{min_required}列有数据)")
    else:
        df_cleaned = df.copy()

    # 步骤2: 清洗列
    original_cols = len(df_cleaned.columns)
    missing_ratio = df_cleaned.isnull().sum() / len(df_cleaned)
    cols_to_drop = [col for col in df_cleaned.columns
                   if col not in base_columns and missing_ratio[col] > max_missing_ratio]

    if cols_to_drop:
        df_cleaned = df_cleaned.drop(columns=cols_to_drop)
        print(f"  - 删除 {len(cols_to_drop)} 列 (缺失率>{max_missing_ratio*100}%)")

    # 步骤3: 删除重复行并排序
    df_cleaned = df_cleaned.drop_duplicates()
    if 'Entity' in df_cleaned.columns and 'Year' in df_cleaned.columns:
        df_cleaned = df_cleaned.sort_values(['Entity', 'Year']).reset_index(drop=True)

    print(f"  - 最终形状: {df_cleaned.shape}")
    return df_cleaned

def fill_missing_forward(df):
    """
    方法1: 前向填充 (按Entity分组)
    """
    print(f"\n方法1: 前向填充")

    if df is None:
        return None

    df_filled = df.copy()
    base_columns = ['Entity', 'Code', 'Year']
    data_columns = [col for col in df_filled.columns if col not in base_columns]

    before_missing = df_filled[data_columns].isnull().sum().sum()

    # 确保按Entity和Year排序
    if 'Entity' in df_filled.columns and 'Year' in df_filled.columns:
        df_filled = df_filled.sort_values(['Entity', 'Year'])

        # 按Entity分组进行前向填充
        for col in data_columns:
            df_filled[col] = df_filled.groupby('Entity')[col].fillna(method='ffill')

    after_missing = df_filled[data_columns].isnull().sum().sum()
    filled_count = before_missing - after_missing

    print(f"  - 填充前缺失值: {before_missing:,}")
    print(f"  - 填充后缺失值: {after_missing:,}")
    print(f"  - 成功填充: {filled_count:,}")

    return df_filled

def fill_missing_interpolate(df):
    """
    方法2: 线性插值 (按Entity分组)
    """
    print(f"\n方法2: 线性插值")

    if df is None:
        return None

    df_filled = df.copy()
    base_columns = ['Entity', 'Code', 'Year']
    data_columns = [col for col in df_filled.columns if col not in base_columns]

    before_missing = df_filled[data_columns].isnull().sum().sum()

    # 确保按Entity和Year排序
    if 'Entity' in df_filled.columns and 'Year' in df_filled.columns:
        df_filled = df_filled.sort_values(['Entity', 'Year'])

        # 按Entity分组进行插值
        for col in data_columns:
            if df_filled[col].dtype in ['float64', 'int64']:
                df_filled[col] = df_filled.groupby('Entity')[col].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )

    after_missing = df_filled[data_columns].isnull().sum().sum()
    filled_count = before_missing - after_missing

    print(f"  - 填充前缺失值: {before_missing:,}")
    print(f"  - 填充后缺失值: {after_missing:,}")
    print(f"  - 成功填充: {filled_count:,}")

    return df_filled

def fill_missing_median(df):
    """
    方法3: 中位数填充 (按年份分组)
    """
    print(f"\n方法3: 中位数填充")

    if df is None:
        return None

    df_filled = df.copy()
    base_columns = ['Entity', 'Code', 'Year']
    data_columns = [col for col in df_filled.columns if col not in base_columns]

    before_missing = df_filled[data_columns].isnull().sum().sum()

    # 按年份计算中位数并填充
    if 'Year' in df_filled.columns:
        for col in data_columns:
            if df_filled[col].dtype in ['float64', 'int64']:
                yearly_median = df_filled.groupby('Year')[col].transform('median')
                df_filled[col] = df_filled[col].fillna(yearly_median)

    after_missing = df_filled[data_columns].isnull().sum().sum()
    filled_count = before_missing - after_missing

    print(f"  - 填充前缺失值: {before_missing:,}")
    print(f"  - 填充后缺失值: {after_missing:,}")
    print(f"  - 成功填充: {filled_count:,}")

    return df_filled

def fill_missing_global(df):
    """
    方法4: 全局统计值填充
    """
    print(f"\n方法4: 全局统计值填充")

    if df is None:
        return None

    df_filled = df.copy()
    base_columns = ['Entity', 'Code', 'Year']
    data_columns = [col for col in df_filled.columns if col not in base_columns]

    before_missing = df_filled[data_columns].isnull().sum().sum()

    for col in data_columns:
        missing_count = df_filled[col].isnull().sum()
        if missing_count > 0:
            if df_filled[col].dtype in ['float64', 'int64']:
                # 数值型用中位数
                fill_value = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(fill_value)
            else:
                # 分类型用众数
                mode_value = df_filled[col].mode()
                if not mode_value.empty:
                    df_filled[col] = df_filled[col].fillna(mode_value.iloc[0])

    after_missing = df_filled[data_columns].isnull().sum().sum()
    filled_count = before_missing - after_missing

    print(f"  - 填充前缺失值: {before_missing:,}")
    print(f"  - 填充后缺失值: {after_missing:,}")
    print(f"  - 成功填充: {filled_count:,}")

    return df_filled

def fill_missing_multiple(df):
    """
    方法5: 多重填充 (组合多种方法)
    """
    print(f"\n方法5: 多重填充 (前向填充 + 插值 + 中位数)")

    if df is None:
        return None

    df_filled = df.copy()
    base_columns = ['Entity', 'Code', 'Year']
    data_columns = [col for col in df_filled.columns if col not in base_columns]

    before_missing = df_filled[data_columns].isnull().sum().sum()

    # 确保按Entity和Year排序
    if 'Entity' in df_filled.columns and 'Year' in df_filled.columns:
        df_filled = df_filled.sort_values(['Entity', 'Year'])

        # 步骤1: 前向填充
        for col in data_columns:
            df_filled[col] = df_filled.groupby('Entity')[col].fillna(method='ffill')

        # 步骤2: 线性插值
        for col in data_columns:
            if df_filled[col].dtype in ['float64', 'int64']:
                df_filled[col] = df_filled.groupby('Entity')[col].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )

        # 步骤3: 年份中位数填充
        for col in data_columns:
            if df_filled[col].dtype in ['float64', 'int64']:
                yearly_median = df_filled.groupby('Year')[col].transform('median')
                df_filled[col] = df_filled[col].fillna(yearly_median)

        # 步骤4: 全局中位数填充剩余缺失值
        for col in data_columns:
            missing_count = df_filled[col].isnull().sum()
            if missing_count > 0:
                if df_filled[col].dtype in ['float64', 'int64']:
                    fill_value = df_filled[col].median()
                    df_filled[col] = df_filled[col].fillna(fill_value)
                else:
                    mode_value = df_filled[col].mode()
                    if not mode_value.empty:
                        df_filled[col] = df_filled[col].fillna(mode_value.iloc[0])

    after_missing = df_filled[data_columns].isnull().sum().sum()
    filled_count = before_missing - after_missing

    print(f"  - 填充前缺失值: {before_missing:,}")
    print(f"  - 填充后缺失值: {after_missing:,}")
    print(f"  - 成功填充: {filled_count:,}")

    return df_filled

def fill_missing_zero(df):
    """
    方法6: 零值填充
    """
    print(f"\n方法6: 零值填充")

    if df is None:
        return None

    df_filled = df.copy()
    base_columns = ['Entity', 'Code', 'Year']
    data_columns = [col for col in df_filled.columns if col not in base_columns]

    before_missing = df_filled[data_columns].isnull().sum().sum()

    # 数值列用0填充，文本列用'Unknown'填充
    for col in data_columns:
        missing_count = df_filled[col].isnull().sum()
        if missing_count > 0:
            if df_filled[col].dtype in ['float64', 'int64']:
                df_filled[col] = df_filled[col].fillna(0)
            else:
                df_filled[col] = df_filled[col].fillna('Unknown')

    after_missing = df_filled[data_columns].isnull().sum().sum()
    filled_count = before_missing - after_missing

    print(f"  - 填充前缺失值: {before_missing:,}")
    print(f"  - 填充后缺失值: {after_missing:,}")
    print(f"  - 成功填充: {filled_count:,}")

    return df_filled

def drop_missing_rows(df):
    """
    方法7: 删除含缺失值的行
    """
    print(f"\n方法7: 删除含缺失值的行")

    if df is None:
        return None

    base_columns = ['Entity', 'Code', 'Year']
    data_columns = [col for col in df.columns if col not in base_columns]

    original_rows = len(df)

    if data_columns:
        df_clean = df.dropna(subset=data_columns, how='any')
    else:
        df_clean = df.copy()

    final_rows = len(df_clean)
    removed_rows = original_rows - final_rows

    print(f"  - 原始行数: {original_rows:,}")
    print(f"  - 最终行数: {final_rows:,}")
    print(f"  - 删除行数: {removed_rows:,}")
    print(f"  - 剩余缺失值: 0")

    return df_clean

def calculate_data_quality(df, name):
    """
    计算数据质量指标
    """
    if df is None:
        return None

    base_columns = ['Entity', 'Code', 'Year']
    data_columns = [col for col in df.columns if col not in base_columns]

    stats = {
        'dataset_name': name,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'data_columns': len(data_columns),
        'total_entities': df['Entity'].nunique() if 'Entity' in df.columns else 0,
        'year_range': f"{df['Year'].min()}-{df['Year'].max()}" if 'Year' in df.columns else 'N/A',
        'total_missing': df[data_columns].isnull().sum().sum() if data_columns else 0,
        'completeness': 0
    }

    if data_columns and len(df) > 0:
        total_cells = len(df) * len(data_columns)
        non_null_cells = df[data_columns].notna().sum().sum()
        stats['completeness'] = (non_null_cells / total_cells * 100) if total_cells > 0 else 100

    return stats

def save_dataset_with_info(df, filename, description):
    """
    保存数据集并显示信息
    """
    if df is None:
        print(f"  - 无法保存 {filename}: 数据为空")
        return False

    # 确保文件保存到refined_data目录
    refined_filename = f"refined_data/{filename}"

    try:
        # 确保输出目录存在
        output_path = Path(refined_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(refined_filename, index=False, encoding='utf-8')
        file_size = Path(refined_filename).stat().st_size / 1024 / 1024

        # 计算数据质量
        stats = calculate_data_quality(df, refined_filename)

        print(f"  - 保存成功: {refined_filename}")
        print(f"    * 文件大小: {file_size:.1f} MB")
        print(f"    * 行数: {stats['total_rows']:,}")
        print(f"    * 列数: {stats['total_columns']}")
        print(f"    * 实体数: {stats['total_entities']}")
        print(f"    * 年份范围: {stats['year_range']}")
        print(f"    * 缺失值: {stats['total_missing']:,}")
        print(f"    * 完整性: {stats['completeness']:.1f}%")
        print(f"    * 描述: {description}")

        return True
    except Exception as e:
        print(f"  - 保存失败 {refined_filename}: {e}")
        return False

def process_all_methods(input_file='merged_co2_dataset.csv'):
    """
    使用所有方法处理数据并生成不同的数据集
    """
    print("CO2数据集完整处理 - 生成多种版本")
    print("="*80)

    # 加载原始数据
    df_original = load_merged_dataset(input_file)
    if df_original is None:
        return

    print(f"\n原始数据信息:")
    print(f"  - 形状: {df_original.shape}")
    print(f"  - 缺失值总数: {df_original.isnull().sum().sum():,}")

    # 基础清洗 (标准参数)
    print(f"\n" + "="*80)
    print("第一步: 基础清洗")
    print("="*80)
    df_cleaned = basic_clean(df_original, max_missing_ratio=0.85, min_data_columns=2, min_data_ratio=0.2)

    if df_cleaned is None:
        print("基础清洗失败，终止处理")
        return

    # 保存基础清洗版本
    print(f"\n保存基础清洗版本:")
    save_dataset_with_info(df_cleaned, 'co2_dataset_01_basic_cleaned.csv',
                          '基础清洗版本 - 删除低质量行和列')

    # 生成不同填充方法的版本
    print(f"\n" + "="*80)
    print("第二步: 应用不同的缺失值处理方法")
    print("="*80)

    methods = [
        (fill_missing_forward, 'co2_dataset_02_forward_fill.csv', '前向填充版本 - 按实体时间序列填充'),
        (fill_missing_interpolate, 'co2_dataset_03_interpolate.csv', '线性插值版本 - 按实体插值填充'),
        (fill_missing_median, 'co2_dataset_04_median_fill.csv', '中位数填充版本 - 按年份中位数填充'),
        (fill_missing_global, 'co2_dataset_05_global_fill.csv', '全局统计填充版本 - 全局中位数/众数填充'),
        (fill_missing_multiple, 'co2_dataset_06_multiple_fill.csv', '多重填充版本 - 组合多种填充方法'),
        (fill_missing_zero, 'co2_dataset_07_zero_fill.csv', '零值填充版本 - 数值用0，文本用Unknown填充'),
        (drop_missing_rows, 'co2_dataset_08_complete_cases.csv', '完整案例版本 - 删除所有含缺失值的行')
    ]

    results = []

    for i, (method_func, filename, description) in enumerate(methods, 1):
        print(f"\n{'-'*60}")
        print(f"处理方法 {i}/7: {description}")
        print(f"{'-'*60}")

        # 应用处理方法
        df_processed = method_func(df_cleaned.copy())

        # 保存结果
        if df_processed is not None:
            success = save_dataset_with_info(df_processed, filename, description)
            if success:
                stats = calculate_data_quality(df_processed, filename)
                results.append(stats)

        print()

    # 生成严格清洗版本 + 多重填充
    print(f"\n" + "="*80)
    print("第三步: 严格清洗版本")
    print("="*80)

    df_strict = basic_clean(df_original, max_missing_ratio=0.70, min_data_columns=3, min_data_ratio=0.3)
    if df_strict is not None:
        df_strict_filled = fill_missing_multiple(df_strict.copy())
        save_dataset_with_info(df_strict_filled, 'co2_dataset_09_strict_multiple.csv',
                              '严格清洗+多重填充版本 - 高质量数据集')

        stats = calculate_data_quality(df_strict_filled, 'co2_dataset_09_strict_multiple.csv')
        if stats:
            results.append(stats)

    # 生成宽松清洗版本 + 插值填充
    print(f"\n" + "="*80)
    print("第四步: 宽松清洗版本")
    print("="*80)

    df_loose = basic_clean(df_original, max_missing_ratio=0.95, min_data_columns=1, min_data_ratio=0.1)
    if df_loose is not None:
        df_loose_filled = fill_missing_interpolate(df_loose.copy())
        save_dataset_with_info(df_loose_filled, 'co2_dataset_10_loose_interpolate.csv',
                              '宽松清洗+插值填充版本 - 最大数据保留')

        stats = calculate_data_quality(df_loose_filled, 'co2_dataset_10_loose_interpolate.csv')
        if stats:
            results.append(stats)

    # 生成汇总报告
    generate_summary_report(results)

def generate_summary_report(results):
    """
    生成所有数据集的汇总报告
    """
    print(f"\n" + "="*80)
    print("数据集汇总报告")
    print("="*80)

    if not results:
        print("没有成功生成的数据集")
        return

    # 创建汇总DataFrame
    summary_df = pd.DataFrame(results)

    # 按完整性排序
    summary_df = summary_df.sort_values('completeness', ascending=False)

    print(f"\n共生成 {len(results)} 个数据集:")
    print(f"\n{'序号':<3} {'文件名':<40} {'行数':<8} {'列数':<4} {'实体数':<6} {'完整性':<8} {'缺失值':<8}")
    print("-" * 85)

    for i, row in summary_df.iterrows():
        print(f"{len(summary_df)-list(summary_df.index).index(i):<3} "
              f"{row['dataset_name']:<40} "
              f"{row['total_rows']:<8,} "
              f"{row['total_columns']:<4} "
              f"{row['total_entities']:<6} "
              f"{row['completeness']:<7.1f}% "
              f"{row['total_missing']:<8,}")

    print(f"\n推荐使用场景:")
    print(f"• 机器学习: co2_dataset_08_complete_cases.csv (无缺失值)")
    print(f"• 统计分析: co2_dataset_06_multiple_fill.csv (多重填充)")
    print(f"• 时间序列: co2_dataset_03_interpolate.csv (插值填充)")
    print(f"• 高质量分析: co2_dataset_09_strict_multiple.csv (严格清洗)")
    print(f"• 最大覆盖: co2_dataset_10_loose_interpolate.csv (宽松清洗)")

    # 保存汇总报告
    summary_report_file = 'refined_data/co2_datasets_summary_report.csv'
    Path(summary_report_file).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_report_file, index=False, encoding='utf-8')
    print(f"\n汇总报告已保存到: {summary_report_file}")

def main():
    """
    主函数
    """
    print("CO2数据集完整处理工具")
    print("将使用所有处理方法生成不同版本的数据集")
    print("="*80)

    # 检查输入文件
    input_file = 'merged_co2_dataset.csv'
    if not Path(input_file).exists():
        print(f"错误: 输入文件 {input_file} 不存在")
        print("请先运行数据合并脚本生成该文件")
        return

    # 执行完整处理
    process_all_methods(input_file)

    print(f"\n" + "="*80)
    print("所有处理完成！")
    print("="*80)
    print("\n生成的文件:")
    print("1. refined_data/co2_dataset_01_basic_cleaned.csv - 基础清洗版本")
    print("2. refined_data/co2_dataset_02_forward_fill.csv - 前向填充版本")
    print("3. refined_data/co2_dataset_03_interpolate.csv - 线性插值版本")
    print("4. refined_data/co2_dataset_04_median_fill.csv - 中位数填充版本")
    print("5. refined_data/co2_dataset_05_global_fill.csv - 全局统计填充版本")
    print("6. refined_data/co2_dataset_06_multiple_fill.csv - 多重填充版本 (推荐)")
    print("7. refined_data/co2_dataset_07_zero_fill.csv - 零值填充版本")
    print("8. refined_data/co2_dataset_08_complete_cases.csv - 完整案例版本")
    print("9. refined_data/co2_dataset_09_strict_multiple.csv - 严格清洗版本")
    print("10. refined_data/co2_dataset_10_loose_interpolate.csv - 宽松清洗版本")
    print("11. refined_data/co2_datasets_summary_report.csv - 汇总报告")

if __name__ == "__main__":
    main()
