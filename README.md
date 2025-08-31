# ML Select - Intelligent Machine Learning Algorithm Recommendation System
# ML Select - 智能机器学习算法推荐系统

## English Documentation

### Overview
ML Select is an intelligent machine learning tool that automatically analyzes your data and recommends the most suitable algorithms. It provides both training and prediction capabilities through command-line interfaces.

### Features
- **Automatic Data Analysis**: Comprehensive data profiling and quality assessment
- **Smart Algorithm Recommendation**: AI-powered algorithm selection based on data characteristics
- **Model Training**: Automated training with the best-performing algorithms
- **Prediction Service**: Easy-to-use prediction interface for trained models
- **Multiple Input Formats**: Support for various data formats and column specifications

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install as package (optional)
pip install -e .
```

### Quick Start

#### 1. Training a Model
```bash
# Basic usage with column names
python mlselect.py -i data.csv -o results.json -xname "age,income" -yname "salary"

# Using column positions
python mlselect.py -i data.csv -o results.json -xloc "0,2" -yloc "3"

# Train top 3 algorithms
python mlselect.py -i data.csv -o results.json -xname "feature1,feature2" -yname "target" -n 3
```

#### 2. Making Predictions
```bash
# Batch prediction from CSV file
python predict.py --model model_name --input new_data.csv --output predictions.csv

# Interactive prediction
python predict.py --model model_name --interactive
```

### Command Line Arguments

#### mlselect.py
- `-i, --input`: Input CSV file path (required)
- `-o, --output`: Output JSON file path (required)
- `-xname`: Feature column names (comma-separated)
- `-xloc`: Feature column positions (comma-separated, 0-indexed)
- `-yname`: Target column name
- `-yloc`: Target column position (0-indexed)
- `-n, --num_algorithms`: Number of top algorithms to train (default: 1)

#### predict.py
- `--model`: Model name/path (required)
- `--input`: Input CSV file for batch prediction
- `--output`: Output CSV file for predictions
- `--interactive`: Enable interactive prediction mode

### Output Format
The training results are saved in JSON format containing:
- File information and data statistics
- Data quality analysis
- Feature analysis and correlations
- Algorithm recommendations with scores
- Training results and model performance metrics

### Model Management
- Models are automatically saved to `data/models/` directory
- Each model includes metadata and can be used for future predictions
- Model naming follows the pattern: `{algorithm}_{timestamp}`

---

## 中文文档

### 概述
ML Select 是一个智能机器学习工具，能够自动分析您的数据并推荐最适合的算法。它通过命令行界面提供训练和预测功能。

### 功能特点
- **自动数据分析**：全面的数据概况分析和质量评估
- **智能算法推荐**：基于数据特征的AI算法选择
- **模型训练**：使用最佳算法自动训练
- **预测服务**：易于使用的训练模型预测接口
- **多种输入格式**：支持各种数据格式和列规范

### 安装
```bash
# 安装依赖
pip install -r requirements.txt

# 安装为包（可选）
pip install -e .
```

### 快速开始

#### 1. 训练模型
```bash
# 使用列名的基本用法
python mlselect.py -i data.csv -o results.json -xname "age,income" -yname "salary"

# 使用列位置
python mlselect.py -i data.csv -o results.json -xloc "0,2" -yloc "3"

# 训练前3个算法
python mlselect.py -i data.csv -o results.json -xname "feature1,feature2" -yname "target" -n 3
```

#### 2. 进行预测
```bash
# 从CSV文件批量预测
python predict.py --model model_name --input new_data.csv --output predictions.csv

# 交互式预测
python predict.py --model model_name --interactive
```

### 命令行参数

#### mlselect.py
- `-i, --input`：输入CSV文件路径（必需）
- `-o, --output`：输出JSON文件路径（必需）
- `-xname`：特征列名（逗号分隔）
- `-xloc`：特征列位置（逗号分隔，从0开始）
- `-yname`：目标列名
- `-yloc`：目标列位置（从0开始）
- `-n, --num_algorithms`：训练的顶级算法数量（默认：1）

#### predict.py
- `--model`：模型名称/路径（必需）
- `--input`：批量预测的输入CSV文件
- `--output`：预测结果的输出CSV文件
- `--interactive`：启用交互式预测模式

### 输出格式
训练结果以JSON格式保存，包含：
- 文件信息和数据统计
- 数据质量分析
- 特征分析和相关性
- 算法推荐及评分
- 训练结果和模型性能指标

### 模型管理
- 模型自动保存到 `data/models/` 目录
- 每个模型包含元数据，可用于未来预测
- 模型命名遵循模式：`{algorithm}_{timestamp}`

### 项目结构
```
ML_select/
├── mlselect.py          # 主训练脚本
├── predict.py           # 预测脚本
├── setup.py            # 安装配置
├── ml_engine/          # 机器学习引擎
│   ├── algorithms/     # 算法实现
│   ├── feature_analysis/ # 特征分析
│   ├── model_selection/ # 模型选择
│   └── evaluation/     # 模型评估
├── data/               # 数据存储
│   └── models/         # 训练好的模型
└── README.md           # 本文档
```

### 示例

#### 训练示例
```bash
# 使用年龄和经验预测薪资
python mlselect.py -i employee_data.csv -o salary_model.json -xname "age,experience" -yname "salary"
```

#### 预测示例
```bash
# 批量预测
python predict.py --model salary_prediction_model --input new_employees.csv --output salary_predictions.csv

# 交互式预测
python predict.py --model salary_prediction_model --interactive
# 然后输入：25,3 (年龄25岁，3年经验)
```

### 支持的算法
- **回归**：线性回归、随机森林回归、支持向量回归、决策树回归
- **分类**：逻辑回归、随机森林分类、支持向量机、决策树分类
- **聚类**：K-Means、层次聚类

### 故障排除
1. **模块导入错误**：确保已安装所有依赖项
2. **文件路径错误**：使用绝对路径或确保文件在当前目录
3. **列名错误**：检查CSV文件的列名是否正确
4. **模型未找到**：确保模型已训练并保存在正确位置

### 贡献
欢迎提交问题和拉取请求来改进这个项目。

### 许可证
本项目采用MIT许可证。