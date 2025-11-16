# CO2数据集合并与清洗操作报告

## 项目概述

本项目将多个CO2排放相关的CSV文件合并成一个统一的数据集，并进行数据清洗以提高数据质量。

## 数据源文件

项目使用了以下6个CSV文件：

1. **annual-co2-emissions-per-country.csv** - 各国年度CO2排放量
2. **carbon-emission-intensity-vs-gdp-per-capita.csv** - 碳排放强度与人均GDP关系
3. **co2-by-source.csv** - 按来源分类的CO2排放
4. **co2-emissions-vs-gdp.csv** - CO2排放与GDP关系
5. **life-expectancy-at-birth-vs-co-emissions-per-capita.csv** - 出生时预期寿命与人均CO2排放关系
6. **share-co2-embedded-in-trade.csv** - 贸易中嵌入的CO2排放份额

## 数据合并操作

### 合并策略
- **主键**：使用 `Entity`（实体/国家）、`Code`（国家代码）、`Year`（年份）作为复合主键
- **连接方式**：外连接（outer join），保留所有记录，缺失数据用NaN填充
- **年份过滤**：只保留1700年及以后的数据

### 合并结果
- **原始合并数据集**：
  - 行数：70,802
  - 列数：22
  - 年份范围：1700-2024（过滤前为-10000-2024）
  - 唯一实体数：336
  - 文件大小：4.5 MB

### 合并后的列结构

#### 基础列（3列）
1. `Entity` - 实体/国家名称
2. `Code` - 国家代码
3. `Year` - 年份

#### 数据列（19列）
1. `Annual CO₂ emissions` - 年度CO2排放量
2. `Carbon intensity of GDP (kg CO2e per 2021 PPP $)` - GDP碳强度
3. `GDP per capita, PPP (constant 2021 international $)` - 人均GDP（PPP）
4. `World regions according to OWID` - 世界地区分类
5. `Life expectancy - Sex: all - Age: 0 - Variant: estimates` - 出生时预期寿命
6. `Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)` - 人均CO2排放量（排除LULUCF）
7. `World regions according to OWID_life_expectancy_at_birth_vs_co_emissions_per_capita` - 地区分类（重复列）
8. `Annual CO₂ emissions from other industry` - 其他工业年度CO2排放
9. `Annual CO₂ emissions from flaring` - 燃烧年度CO2排放
10. `Annual CO₂ emissions from cement` - 水泥年度CO2排放
11. `Annual CO₂ emissions from gas` - 天然气年度CO2排放
12. `Annual CO₂ emissions from oil` - 石油年度CO2排放
13. `Annual CO₂ emissions from coal` - 煤炭年度CO2排放
14. `Share of annual CO₂ emissions embedded in trade` - 贸易中嵌入的CO2排放份额
15. `Annual CO₂ emissions (per capita)` - 人均年度CO2排放
16. `GDP per capita` - 人均GDP
17. `900793-annotations` - 注释列
18. `Population (historical)` - 历史人口数据
19. `World regions according to OWID_co2_emissions_vs_gdp` - 地区分类（重复列）

## 数据清洗操作

### 清洗策略
采用**先清洗行，再清洗列**的策略，确保更准确的数据质量评估。

### 清洗参数设置
提供了三种清洗强度选项：

#### 1. 标准清洗（推荐）
- **行清洗**：保留至少2列有数据或数据比例>20%的行
- **列清洗**：删除缺失值>85%的列

#### 2. 严格清洗
- **行清洗**：保留至少3列有数据或数据比例>30%的行
- **列清洗**：删除缺失值>70%的列

#### 3. 宽松清洗
- **行清洗**：保留至少1列有数据或数据比例>10%的行
- **列清洗**：删除缺失值>95%的列

### 清洗步骤

#### 步骤1：行清洗
- 计算每行在数据列中的非空值数量
- 根据设定阈值删除数据过少的行
- 保护基础列（Entity, Code, Year）

#### 步骤2：列清洗
- 基于清洗后的行重新计算列的缺失比例
- 删除缺失值比例超过阈值的列
- 保护基础列不被删除

#### 步骤3：删除重复行
- 识别并删除完全重复的记录

#### 步骤4：排序和整理
- 按Entity和Year排序
- 重置索引

## 清洗结果对比

### 标准清洗结果
- **数据保留**：
  - 行保留率：46.4%（29,013/62,517）
  - 列保留率：77.3%（17/22）
  - 数据完整性：60.3%
- **删除统计**：
  - 删除行数：33,504
  - 删除列数：5
- **文件信息**：
  - 文件大小：2.8 MB
  - 唯一实体数：300
  - 年份范围：1750-2023

#### 保留的列（17列）
1. `Entity` - 实体/国家名称
2. `Code` - 国家代码  
3. `Year` - 年份
4. `Annual CO₂ emissions` - 年度CO2排放量（6.2% 缺失）
5. `Carbon intensity of GDP (kg CO2e per 2021 PPP $)` - GDP碳强度
6. `GDP per capita, PPP (constant 2021 international $)` - 人均GDP（PPP）
7. `Life expectancy - Sex: all - Age: 0 - Variant: estimates` - 出生时预期寿命（44.0% 缺失）
8. `Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)` - 人均CO2排放量
9. `Annual CO₂ emissions from flaring` - 燃烧年度CO2排放（45.6% 缺失）
10. `Annual CO₂ emissions from cement` - 水泥年度CO2排放（23.2% 缺失）
11. `Annual CO₂ emissions from gas` - 天然气年度CO2排放（37.9% 缺失）
12. `Annual CO₂ emissions from oil` - 石油年度CO2排放（13.1% 缺失）
13. `Annual CO₂ emissions from coal` - 煤炭年度CO2排放（25.0% 缺失）
14. `Share of annual CO₂ emissions embedded in trade` - 贸易中嵌入的CO2排放份额
15. `Annual CO₂ emissions (per capita)` - 人均年度CO2排放（9.4% 缺失）
16. `GDP per capita` - 人均GDP（45.3% 缺失）
17. `Population (historical)` - 历史人口数据（9.5% 缺失）

#### 删除的列（5列）
1. `World regions according to OWID`（99.1% 缺失）
2. `World regions according to OWID_life_expectancy_at_birth_vs_co_emissions_per_capita`（99.1% 缺失）
3. `Annual CO₂ emissions from other industry`（89.0% 缺失）
4. `900793-annotations`（100.0% 缺失）
5. `World regions according to OWID_co2_emissions_vs_gdp`（99.1% 缺失）

### 严格清洗结果
- **数据保留**：
  - 行保留率：42.1%（26,331/62,517）
  - 列保留率：63.6%（14/22）
  - 数据完整性：75.5%
- **文件信息**：
  - 文件大小：2.4 MB
  - 唯一实体数：257
  - 年份范围：1750-2023

### 宽松清洗结果
- **数据保留**：
  - 行保留率：100.0%（62,517/62,517）
  - 列保留率：81.8%（18/22）
  - 数据完整性：30.9%
- **文件信息**：
  - 文件大小：4.2 MB
  - 唯一实体数：336
  - 年份范围：1700-2024

## 数据质量评估

### 原始数据质量问题
1. **高缺失率列**：9列缺失率>80%，4列缺失率>95%
2. **空值分布不均**：25,496行缺失>90%数据，37,421行缺失>70%数据
3. **重复列问题**：存在多个地区分类的重复列
4. **注释列问题**：annotations列100%为空

### 清洗后数据质量改善
- **数据完整性显著提升**：从原始的低完整性提升到60.3%（标准清洗）
- **列质量优化**：保留的列中最优列缺失率仅6.2%
- **行质量统一**：确保每行都有足够的有效数据
- **数据结构简化**：删除重复和无效列，提高数据可用性

## 缺失值处理操作

在基础清洗之后，我们进一步实施了多种缺失值处理方法，以满足不同分析需求。

### 处理策略
采用**分步骤处理**的方法：
1. **基础清洗**：先删除质量不好的行，再删除质量不好的列
2. **缺失值填充**：应用多种填充方法
3. **质量优化**：根据具体需求进行最终处理

### 缺失值填充方法

#### 方法1：前向填充
- **原理**：按Entity分组，使用时间序列前值填充缺失值
- **适用场景**：时间序列数据，假设数据具有时间连续性
- **效果**：填充了8,118个缺失值，完整性提升到62.3%

#### 方法2：线性插值
- **原理**：按Entity分组进行线性插值，适用于数值型数据
- **适用场景**：数值变化相对平滑的时间序列
- **效果**：填充了108,989个缺失值，完整性提升到87.1%

#### 方法3：中位数填充
- **原理**：按年份分组，使用当年所有实体的中位数填充
- **适用场景**：需要保持统计特性的分析
- **效果**：填充了71,016个缺失值，完整性提升到77.8%

#### 方法4：全局统计填充
- **原理**：数值型用全局中位数，分类型用全局众数
- **适用场景**：需要完全无缺失值的数据集
- **效果**：填充所有161,374个缺失值，完整性达到100%

#### 方法5：多重填充（推荐）
- **原理**：组合前向填充、线性插值、中位数填充和全局填充
- **适用场景**：平衡各种填充方法优势的综合解决方案
- **效果**：填充所有161,374个缺失值，完整性达到100%

#### 方法6：零值填充
- **原理**：数值型用0填充，文本型用'Unknown'填充
- **适用场景**：特定分析需求，如将缺失视为无排放
- **效果**：填充所有161,374个缺失值，完整性达到100%

#### 方法7：删除缺失行
- **原理**：删除所有包含缺失值的行，保留完整记录
- **适用场景**：机器学习模型训练
- **效果**：删除26,420行，保留2,593完整记录

### 处理结果对比

| 处理方法 | 行数 | 列数 | 实体数 | 完整性 | 缺失值 | 文件大小 | 年份范围 |
|---------|------|------|--------|--------|--------|----------|----------|
| 基础清洗 | 29,013 | 17 | 300 | 60.3% | 161,374 | 2.8 MB | 1750-2023 |
| 前向填充 | 29,013 | 17 | 300 | 62.3% | 153,256 | 2.8 MB | 1750-2023 |
| 线性插值 | 29,013 | 17 | 300 | 87.1% | 52,385 | 3.6 MB | 1750-2023 |
| 中位数填充 | 29,013 | 17 | 300 | 77.8% | 90,358 | 3.3 MB | 1750-2023 |
| 全局统计填充 | 29,013 | 17 | 300 | 100.0% | 0 | 4.0 MB | 1750-2023 |
| 多重填充 | 29,013 | 17 | 300 | 100.0% | 0 | 4.0 MB | 1750-2023 |
| 零值填充 | 29,013 | 17 | 300 | 100.0% | 0 | 3.2 MB | 1750-2023 |
| 完整案例 | 2,593 | 17 | 90 | 100.0% | 0 | 0.4 MB | 1990-2022 |
| 严格清洗+多重填充 | 26,331 | 14 | 257 | 100.0% | 0 | 2.9 MB | 1750-2023 |
| 宽松清洗+插值 | 62,517 | 18 | 336 | 77.0% | 215,427 | 7.3 MB | 1700-2024 |

## 生成的文件

### 原始和基础清洗文件
1. **merged_co2_dataset.csv** - 原始合并数据集（4.5 MB）
2. **cleaned_co2_dataset_standard.csv** - 标准清洗结果（2.8 MB）
3. **cleaned_co2_dataset_strict.csv** - 严格清洗结果（2.4 MB）
4. **cleaned_co2_dataset_loose.csv** - 宽松清洗结果（4.2 MB）

### 缺失值处理后的文件
5. **co2_dataset_01_basic_cleaned.csv** - 基础清洗版本（2.8 MB）
6. **co2_dataset_02_forward_fill.csv** - 前向填充版本（2.8 MB）
7. **co2_dataset_03_interpolate.csv** - 线性插值版本（3.6 MB）
8. **co2_dataset_04_median_fill.csv** - 中位数填充版本（3.3 MB）
9. **co2_dataset_05_global_fill.csv** - 全局统计填充版本（4.0 MB）
10. **co2_dataset_06_multiple_fill.csv** - 多重填充版本（4.0 MB）**【推荐】**
11. **co2_dataset_07_zero_fill.csv** - 零值填充版本（3.2 MB）
12. **co2_dataset_08_complete_cases.csv** - 完整案例版本（0.4 MB）
13. **co2_dataset_09_strict_multiple.csv** - 严格清洗+多重填充版本（2.9 MB）
14. **co2_dataset_10_loose_interpolate.csv** - 宽松清洗+插值填充版本（7.3 MB）

### 报告文件
15. **co2_datasets_summary_report.csv** - 数据集汇总报告

## 使用建议

### 推荐使用场景

#### 按分析类型选择

1. **机器学习模型训练**：
   - 推荐：`co2_dataset_08_complete_cases.csv`
   - 特点：完全无缺失值，2,593行高质量数据
   - 适用：分类、回归、聚类等算法

2. **统计分析**：
   - 推荐：`co2_dataset_06_multiple_fill.csv`
   - 特点：多重填充方法，平衡了各种填充策略
   - 适用：描述性统计、相关性分析、假设检验

3. **时间序列分析**：
   - 推荐：`co2_dataset_03_interpolate.csv`
   - 特点：线性插值保持了时间序列的平滑性
   - 适用：趋势分析、预测模型、周期性分析

4. **高质量深度分析**：
   - 推荐：`co2_dataset_09_strict_multiple.csv`
   - 特点：严格清洗+多重填充，26,331行高质量数据
   - 适用：学术研究、政策分析、精确建模

5. **最大数据覆盖分析**：
   - 推荐：`co2_dataset_10_loose_interpolate.csv`
   - 特点：保留最多实体和时间跨度，62,517行数据
   - 适用：全球趋势分析、国家对比、长时间序列研究

6. **特殊分析需求**：
   - **零排放假设**：`co2_dataset_07_zero_fill.csv`
   - **保守估计**：`co2_dataset_04_median_fill.csv`
   - **时间连续性**：`co2_dataset_02_forward_fill.csv`

#### 按数据质量要求选择

- **完整性要求最高**：选择100%完整性的版本（全局填充、多重填充、零值填充、完整案例）
- **平衡质量和数量**：选择87.1%完整性的插值版本
- **保持原始特征**：选择62.3%完整性的前向填充版本

### 数据使用注意事项

1. **缺失值处理**：根据具体分析需求选择适当的缺失值处理方法
2. **时间序列分析**：注意不同指标的时间覆盖范围可能不同
3. **实体分类**：数据包含国家、地区和全球等不同层级的实体
4. **单位一致性**：注意不同指标的计量单位和基准年份

## 技术实现

### 使用的工具和库
- **Python 3.x**
- **pandas**：数据处理和分析
- **pathlib**：文件路径处理

### 主要脚本
1. **merge_co2_datasets.py** - 数据合并脚本
2. **data_cleaning.py** - 数据清洗脚本
3. **complete_data_processing.py** - 完整数据处理脚本（包含所有缺失值处理方法）

### 核心算法
- **外连接合并**：使用pandas.merge()实现多表外连接
- **缺失值分析**：基于比例和数量的双重阈值
- **质量评估**：多维度数据质量评估体系
- **前向填充**：groupby + ffill方法实现时间序列填充
- **线性插值**：interpolate方法实现平滑填充
- **统计填充**：median/mode方法实现统计特性保持
- **多重填充**：组合多种方法的渐进式填充策略

### 处理流程
```
原始数据 → 基础清洗 → 缺失值填充 → 质量验证 → 最终数据集
    ↓           ↓           ↓           ↓           ↓
6个CSV → 删除低质量 → 7种填充方法 → 完整性评估 → 11个版本
          行和列
```
