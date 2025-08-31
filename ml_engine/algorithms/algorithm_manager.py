#!/usr/bin/env python3
"""
Algorithm Manager - Unified interface for all ML algorithms

This module provides a unified interface to access regression, classification,
and clustering algorithms in a modular way.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union

from .regression import RegressionAlgorithms
from .classification import ClassificationAlgorithms
from .clustering import ClusteringAlgorithms


class AlgorithmManager:
    """
    Unified manager for all ML algorithms.
    Provides a single interface to access regression, classification, and clustering algorithms.
    """
    
    def __init__(self):
        self.regression = RegressionAlgorithms()
        self.classification = ClassificationAlgorithms()
        self.clustering = ClusteringAlgorithms()
        
        # Algorithm type mapping
        self.algorithm_types = {
            # Regression algorithms
            'linear_regression': 'regression',
            'polynomial_regression': 'regression',
            'random_forest_regression': 'regression',
            'svr': 'regression',
            'decision_tree_regression': 'regression',
            
            # Classification algorithms
            'logistic_regression': 'classification',
            'random_forest_classification': 'classification',
            'svm': 'classification',
            'knn': 'classification',
            'naive_bayes': 'classification',
            'decision_tree_classification': 'classification',
            
            # Clustering algorithms
            'kmeans': 'clustering',
            'hierarchical_clustering': 'clustering'
        }
        
        # Algorithm detailed information with descriptions and use cases
        self.algorithm_info = {
            # Regression algorithms
            'linear_regression': {
                'name': '线性回归 (Linear Regression)',
                'description': '通过拟合线性关系来预测连续数值的基础回归算法。假设特征与目标变量之间存在线性关系。',
                'use_cases': [
                    '房价预测：根据房屋面积、位置等特征预测房价',
                    '销售预测：基于历史数据预测未来销售额',
                    '股票价格预测：分析影响因素预测股价走势',
                    '温度预测：根据气象数据预测温度变化'
                ],
                'advantages': ['简单易懂', '计算速度快', '可解释性强'],
                'limitations': ['只能处理线性关系', '对异常值敏感']
            },
            'polynomial_regression': {
                'name': '多项式回归 (Polynomial Regression)',
                'description': '线性回归的扩展，通过添加特征的高次项来捕捉非线性关系。',
                'use_cases': [
                    '生物增长模型：预测细菌或植物的非线性增长',
                    '经济增长预测：建模GDP等经济指标的非线性变化',
                    '物理现象建模：如抛物线运动轨迹预测',
                    '市场需求分析：捕捉价格与需求的非线性关系'
                ],
                'advantages': ['能处理非线性关系', '基于线性回归易于理解'],
                'limitations': ['容易过拟合', '高次项可能导致数值不稳定']
            },
            'random_forest_regression': {
                'name': '随机森林回归 (Random Forest Regression)',
                'description': '集成学习算法，通过构建多个决策树并平均其预测结果来提高准确性和稳定性。',
                'use_cases': [
                    '金融风险评估：综合多个因素评估贷款违约风险',
                    '医疗诊断：基于多项检查指标预测疾病严重程度',
                    '环境监测：预测空气质量指数或污染程度',
                    '电商推荐：预测用户对商品的评分'
                ],
                'advantages': ['抗过拟合能力强', '能处理缺失值', '提供特征重要性'],
                'limitations': ['模型较复杂', '内存消耗大']
            },
            'svr': {
                'name': '支持向量回归 (Support Vector Regression)',
                'description': '基于支持向量机的回归算法，通过寻找最优超平面来进行预测，对异常值具有较强的鲁棒性。',
                'use_cases': [
                    '时间序列预测：股票价格、汇率等金融时间序列',
                    '图像处理：图像质量评估或像素值预测',
                    '生物信息学：基因表达水平预测',
                    '工程优化：材料性能预测'
                ],
                'advantages': ['对异常值鲁棒', '适用于高维数据', '内存效率高'],
                'limitations': ['参数调优复杂', '对大数据集训练慢']
            },
            'decision_tree_regression': {
                'name': '决策树回归 (Decision Tree Regression)',
                'description': '通过构建决策树来进行预测，每个叶节点代表一个预测值。具有很好的可解释性。',
                'use_cases': [
                    '客户价值分析：根据客户特征预测生命周期价值',
                    '产品定价：基于成本和市场因素确定最优价格',
                    '资源分配：预测项目所需时间或资源',
                    '质量控制：预测产品质量评分'
                ],
                'advantages': ['易于理解和解释', '不需要数据预处理', '能处理非线性关系'],
                'limitations': ['容易过拟合', '对数据变化敏感']
            },
            
            # Classification algorithms
            'logistic_regression': {
                'name': '逻辑回归 (Logistic Regression)',
                'description': '用于二分类和多分类问题的线性分类算法，通过逻辑函数将线性组合映射到概率。',
                'use_cases': [
                    '邮件垃圾分类：判断邮件是否为垃圾邮件',
                    '医疗诊断：预测患者是否患有某种疾病',
                    '营销响应预测：预测客户是否会响应营销活动',
                    '信用评估：判断贷款申请是否会被批准'
                ],
                'advantages': ['输出概率值', '计算速度快', '不容易过拟合'],
                'limitations': ['假设线性关系', '对异常值敏感']
            },
            'random_forest_classification': {
                'name': '随机森林分类 (Random Forest Classification)',
                'description': '集成多个决策树的分类算法，通过投票机制确定最终分类结果。',
                'use_cases': [
                    '图像识别：识别图片中的物体类别',
                    '文本分类：新闻文章主题分类',
                    '生物识别：指纹或人脸识别',
                    '欺诈检测：识别信用卡欺诈交易'
                ],
                'advantages': ['准确率高', '能处理大数据集', '提供特征重要性'],
                'limitations': ['模型复杂', '训练时间长']
            },
            'svm': {
                'name': '支持向量机 (Support Vector Machine)',
                'description': '通过寻找最优分离超平面来进行分类，在高维空间中表现优异。',
                'use_cases': [
                    '文档分类：学术论文或新闻文章分类',
                    '基因分类：基于基因表达数据进行疾病分类',
                    '图像分类：手写数字识别',
                    '情感分析：文本情感倾向分类'
                ],
                'advantages': ['在高维空间有效', '内存效率高', '适用于非线性问题'],
                'limitations': ['对大数据集慢', '对特征缩放敏感']
            },
            'knn': {
                'name': 'K近邻算法 (K-Nearest Neighbors)',
                'description': '基于实例的学习算法，通过寻找K个最近邻居来进行分类预测。',
                'use_cases': [
                    '推荐系统：基于用户相似性推荐商品',
                    '模式识别：手写字符识别',
                    '异常检测：识别网络入侵或异常行为',
                    '地理信息系统：基于位置的服务分类'
                ],
                'advantages': ['简单易实现', '适用于非线性问题', '对局部结构敏感'],
                'limitations': ['计算成本高', '对维度诅咒敏感']
            },
            'naive_bayes': {
                'name': '朴素贝叶斯 (Naive Bayes)',
                'description': '基于贝叶斯定理的概率分类算法，假设特征之间相互独立。',
                'use_cases': [
                    '垃圾邮件过滤：基于邮件内容判断是否为垃圾邮件',
                    '情感分析：分析社交媒体评论的情感倾向',
                    '医疗诊断：基于症状预测疾病类型',
                    '新闻分类：自动将新闻归类到不同主题'
                ],
                'advantages': ['训练速度快', '对小数据集表现好', '对缺失数据不敏感'],
                'limitations': ['特征独立假设强', '对特征相关性敏感']
            },
            'decision_tree_classification': {
                'name': '决策树分类 (Decision Tree Classification)',
                'description': '通过构建决策树来进行分类，每个叶节点代表一个类别。',
                'use_cases': [
                    '客户细分：根据客户特征进行市场细分',
                    '风险评估：评估投资或保险风险等级',
                    '产品质量检测：判断产品是否合格',
                    '招聘筛选：根据简历信息筛选候选人'
                ],
                'advantages': ['易于理解', '不需要数据预处理', '能处理数值和类别特征'],
                'limitations': ['容易过拟合', '对噪声敏感']
            },
            
            # Clustering algorithms
            'kmeans': {
                'name': 'K均值聚类 (K-Means Clustering)',
                'description': '将数据分为K个簇的无监督学习算法，通过最小化簇内平方和来优化聚类结果。',
                'use_cases': [
                    '客户细分：根据购买行为将客户分组',
                    '市场研究：识别不同的消费者群体',
                    '图像分割：将图像分割为不同区域',
                    '基因分析：根据基因表达模式对基因分组'
                ],
                'advantages': ['简单高效', '适用于球形簇', '可扩展性好'],
                'limitations': ['需要预先指定K值', '对异常值敏感']
            },
            'hierarchical_clustering': {
                'name': '层次聚类 (Hierarchical Clustering)',
                'description': '构建数据的层次结构，可以自底向上或自顶向下进行聚类。',
                'use_cases': [
                    '生物分类学：构建物种进化树',
                    '社交网络分析：识别社区结构',
                    '文档聚类：按主题对文档进行层次分组',
                    '产品分类：构建产品类别层次结构'
                ],
                'advantages': ['不需要预先指定簇数', '提供层次结构', '结果稳定'],
                'limitations': ['计算复杂度高', '对噪声敏感']
            }
        }
    
    def get_algorithm_type(self, algorithm_name: str) -> str:
        """
        Get the type of algorithm (regression, classification, clustering)
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Algorithm type string
        """
        return self.algorithm_types.get(algorithm_name.lower(), 'unknown')
    
    def get_algorithm_engine(self, algorithm_name: str):
        """
        Get the appropriate algorithm engine based on algorithm name
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Algorithm engine instance
        """
        alg_type = self.get_algorithm_type(algorithm_name)
        
        if alg_type == 'regression':
            return self.regression
        elif alg_type == 'classification':
            return self.classification
        elif alg_type == 'clustering':
            return self.clustering
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, 
                    feature_columns: list, algorithm: str = None) -> Dict[str, Any]:
        """
        Prepare data for training using the appropriate algorithm engine
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            feature_columns: List of feature column names
            algorithm: Algorithm name (optional)
            
        Returns:
            Prepared data dictionary
        """
        if algorithm:
            engine = self.get_algorithm_engine(algorithm)
            return engine.prepare_data(df, target_column, feature_columns, algorithm)
        else:
            # Default to regression engine for backward compatibility
            return self.regression.prepare_data(df, target_column, feature_columns)
    
    def train_model(self, algorithm_name: str, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a model using the appropriate algorithm engine
        
        Args:
            algorithm_name: Name of the algorithm to use
            data_dict: Prepared data dictionary
            
        Returns:
            Training result dictionary
        """
        engine = self.get_algorithm_engine(algorithm_name)
        return engine.train_model(algorithm_name, data_dict)
    
    def predict(self, model_id: str, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Make predictions using a trained model
        
        Args:
            model_id: ID of the trained model
            data: Input data for prediction
            
        Returns:
            Prediction results
        """
        # Try to find the model in each engine
        for engine in [self.regression, self.classification, self.clustering]:
            try:
                result = engine.predict(model_id, data)
                if result and 'error' not in result:
                    return result
            except:
                continue
        
        return {'error': f'Model {model_id} not found in any engine'}
    
    def get_available_algorithms(self) -> Dict[str, list]:
        """
        Get all available algorithms grouped by type

        Returns:
            Dictionary with algorithm types as keys and algorithm lists as values
        """
        return {
            'regression': self.regression.get_supported_algorithms(),
            'classification': self.classification.get_supported_algorithms(),
            'clustering': self.clustering.get_supported_algorithms()
        }
    
    def get_all_algorithms(self) -> list:
        """
        Get a flat list of all available algorithms
        
        Returns:
            List of all algorithm names
        """
        all_algorithms = []
        available = self.get_available_algorithms()
        for alg_type, algorithms in available.items():
            all_algorithms.extend(algorithms)
        return all_algorithms
    
    def get_algorithm_help(self, algorithm_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific algorithm
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Dictionary containing algorithm information
        """
        algorithm_name = algorithm_name.lower()
        
        if algorithm_name not in self.algorithm_info:
            available_algorithms = self.get_all_algorithms()
            return {
                'error': f"算法 '{algorithm_name}' 不存在",
                'available_algorithms': available_algorithms,
                'suggestion': f"可用的算法包括: {', '.join(available_algorithms)}"
            }
        
        info = self.algorithm_info[algorithm_name]
        algorithm_type = self.get_algorithm_type(algorithm_name)
        
        return {
            'algorithm': algorithm_name,
            'type': algorithm_type,
            'name': info['name'],
            'description': info['description'],
            'use_cases': info['use_cases'],
            'advantages': info['advantages'],
            'limitations': info['limitations']
        }
    
    def print_algorithm_help(self, algorithm_name: str) -> None:
        """
        Print formatted help information for a specific algorithm
        
        Args:
            algorithm_name: Name of the algorithm
        """
        help_info = self.get_algorithm_help(algorithm_name)
        
        if 'error' in help_info:
            print(f"\n错误: {help_info['error']}")
            print(f"建议: {help_info['suggestion']}")
            return
        
        print(f"\n{'='*60}")
        print(f"算法帮助: {help_info['name']}")
        print(f"{'='*60}")
        print(f"\n类型: {help_info['type'].upper()}")
        print(f"\n描述:")
        print(f"  {help_info['description']}")
        
        print(f"\n应用场景:")
        for i, use_case in enumerate(help_info['use_cases'], 1):
            print(f"  {i}. {use_case}")
        
        print(f"\n优势:")
        for advantage in help_info['advantages']:
            print(f"  • {advantage}")
        
        print(f"\n局限性:")
        for limitation in help_info['limitations']:
            print(f"  • {limitation}")
        
        print(f"\n{'='*60}\n")