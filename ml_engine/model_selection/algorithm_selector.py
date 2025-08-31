import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AlgorithmSelector:
    """
    智能算法选择器 - 基于数据特征自动推荐最适合的机器学习算法
    """
    
    def __init__(self):
        self.regression_algorithms = {
            'linear_regression': {
                'name': '线性回归',
                'description': '适用于线性关系的回归问题',
                'conditions': ['linear_relationship', 'continuous_target', 'small_to_medium_data']
            },
            'polynomial_regression': {
                'name': '多项式回归',
                'description': '适用于非线性关系的回归问题',
                'conditions': ['nonlinear_relationship', 'continuous_target', 'small_to_medium_data']
            },
            'random_forest_regression': {
                'name': '随机森林回归',
                'description': '适用于复杂非线性关系，处理缺失值能力强',
                'conditions': ['complex_relationship', 'continuous_target', 'medium_to_large_data']
            },
            'svr': {
                'name': '支持向量回归',
                'description': '适用于高维数据和非线性关系',
                'conditions': ['high_dimensional', 'nonlinear_relationship', 'small_to_medium_data']
            },
            'decision_tree_regression': {
                'name': '决策树回归',
                'description': '适用于非线性关系，易于解释',
                'conditions': ['nonlinear_relationship', 'interpretability_required', 'small_to_medium_data']
            }
        }
        
        self.classification_algorithms = {
            'logistic_regression': {
                'name': '逻辑回归',
                'description': '适用于二分类和多分类问题',
                'conditions': ['binary_classification', 'linear_separable', 'small_to_medium_data']
            },
            'random_forest_classification': {
                'name': '随机森林分类',
                'description': '适用于复杂分类问题，处理缺失值能力强',
                'conditions': ['complex_relationship', 'categorical_target', 'medium_to_large_data']
            },
            'svm': {
                'name': '支持向量机',
                'description': '适用于高维数据分类',
                'conditions': ['high_dimensional', 'small_to_medium_data', 'binary_classification']
            },
            'knn': {
                'name': 'K近邻算法',
                'description': '适用于局部模式明显的分类问题',
                'conditions': ['local_patterns', 'small_to_medium_data', 'categorical_target']
            },
            'naive_bayes': {
                'name': '朴素贝叶斯',
                'description': '适用于文本分类和特征独立的问题',
                'conditions': ['feature_independence', 'categorical_target', 'text_data']
            },
            'decision_tree_classification': {
                'name': '决策树分类',
                'description': '适用于需要解释性的分类问题',
                'conditions': ['interpretability_required', 'categorical_target', 'small_to_medium_data']
            }
        }
        
        self.clustering_algorithms = {
            'kmeans': {
                'name': 'K均值聚类',
                'description': '适用于球形聚类',
                'conditions': ['spherical_clusters', 'known_cluster_number']
            },
            'hierarchical': {
                'name': '层次聚类',
                'description': '适用于不规则形状的聚类',
                'conditions': ['irregular_clusters', 'unknown_cluster_number']
            }
        }
    
    def analyze_data_characteristics(self, df: pd.DataFrame, target_column: str, 
                                   feature_columns: List[str]) -> Dict[str, Any]:
        """
        分析数据特征
        """
        characteristics = {}
        
        # 基本信息
        characteristics['n_samples'] = len(df)
        characteristics['n_features'] = len(feature_columns)
        characteristics['target_type'] = self._determine_target_type(df[target_column])
        
        # 数据大小分类
        if characteristics['n_samples'] < 1000:
            characteristics['data_size'] = 'small'
        elif characteristics['n_samples'] < 10000:
            characteristics['data_size'] = 'medium'
        else:
            characteristics['data_size'] = 'large'
        
        # 特征分析
        characteristics['feature_types'] = self._analyze_feature_types(df[feature_columns])
        characteristics['missing_values'] = df[feature_columns].isnull().sum().sum()
        characteristics['missing_percentage'] = characteristics['missing_values'] / (len(df) * len(feature_columns))
        
        # 目标变量分析
        if characteristics['target_type'] == 'continuous':
            characteristics['target_distribution'] = self._analyze_target_distribution(df[target_column])
        else:
            characteristics['n_classes'] = df[target_column].nunique()
            characteristics['class_balance'] = self._analyze_class_balance(df[target_column])
        
        # 关系分析
        if len(feature_columns) == 1 and characteristics['target_type'] == 'continuous':
            characteristics['relationship_type'] = self._analyze_relationship(
                df[feature_columns[0]], df[target_column]
            )
        
        # 维度分析
        characteristics['dimensionality'] = 'high' if len(feature_columns) > 20 else 'low_to_medium'
        
        return characteristics
    
    def _determine_target_type(self, target_series: pd.Series) -> str:
        """
        确定目标变量类型
        """
        if target_series.dtype in ['int64', 'float64']:
            unique_values = target_series.nunique()
            total_values = len(target_series)
            
            # 如果唯一值比例很高，或者数值范围很大，则认为是连续变量
            unique_ratio = unique_values / total_values
            value_range = target_series.max() - target_series.min()
            
            # 对于数值型数据，如果唯一值比例>0.5或者数值范围>100，认为是连续变量
            if unique_ratio > 0.5 or value_range > 100:
                return 'continuous'
            elif unique_values <= 10 and target_series.dtype == 'int64':
                return 'categorical'
            else:
                return 'continuous'
        else:
            return 'categorical'
    
    def _analyze_feature_types(self, features_df: pd.DataFrame) -> Dict[str, int]:
        """
        分析特征类型
        """
        types = {'numerical': 0, 'categorical': 0}
        for col in features_df.columns:
            if features_df[col].dtype in ['int64', 'float64']:
                types['numerical'] += 1
            else:
                types['categorical'] += 1
        return types
    
    def _analyze_target_distribution(self, target_series: pd.Series) -> Dict[str, float]:
        """
        分析目标变量分布
        """
        return {
            'mean': target_series.mean(),
            'std': target_series.std(),
            'skewness': stats.skew(target_series),
            'kurtosis': stats.kurtosis(target_series)
        }
    
    def _analyze_class_balance(self, target_series: pd.Series) -> Dict[str, float]:
        """
        分析类别平衡性
        """
        value_counts = target_series.value_counts()
        total = len(target_series)
        return {
            'most_frequent_ratio': value_counts.iloc[0] / total,
            'least_frequent_ratio': value_counts.iloc[-1] / total,
            'balance_ratio': value_counts.iloc[-1] / value_counts.iloc[0]
        }
    
    def _analyze_relationship(self, feature_series: pd.Series, target_series: pd.Series) -> str:
        """
        分析特征与目标变量的关系
        """
        correlation = np.corrcoef(feature_series, target_series)[0, 1]
        
        # 线性关系检验
        if abs(correlation) > 0.7:
            return 'linear'
        
        # 非线性关系检验（使用多项式拟合）
        try:
            # 二次多项式拟合
            poly_features = np.column_stack([feature_series, feature_series**2])
            poly_corr = np.corrcoef(poly_features[:, 1], target_series)[0, 1]
            
            if abs(poly_corr) > abs(correlation) + 0.1:
                return 'nonlinear'
        except:
            pass
        
        return 'complex'
    
    def recommend_algorithms(self, X: pd.DataFrame = None, y: pd.Series = None,
                           df: pd.DataFrame = None, target_column: str = None, 
                           feature_columns: List[str] = None, task_type: str = None) -> List[Dict[str, Any]]:
        """
        推荐算法
        支持两种调用方式：
        1. recommend_algorithms(X, y) - 直接传入特征和目标变量
        2. recommend_algorithms(df=df, target_column=target_column, feature_columns=feature_columns) - 传入DataFrame
        """
        # 兼容新的调用方式
        if df is None and X is not None:
            # 使用X和y构建临时DataFrame
            if isinstance(X, pd.DataFrame):
                df = X.copy()
                feature_columns = X.columns.tolist()
            else:
                df = pd.DataFrame(X)
                feature_columns = df.columns.tolist()
            
            if y is not None:
                target_column = 'target'
                df[target_column] = y
        elif df is None:
            # 没有提供足够的数据
            return [{
                'algorithm': 'error',
                'name': '错误',
                'description': '未提供足够的数据进行分析',
                'score': 0,
                'suitability': '不推荐'
            }]
        
        # 确保feature_columns存在
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        characteristics = self.analyze_data_characteristics(df, target_column, feature_columns)
        
        # 自动确定任务类型
        if task_type is None:
            if characteristics['target_type'] == 'continuous':
                task_type = 'regression'
            else:
                task_type = 'classification'
        
        # 获取候选算法
        if task_type == 'regression':
            algorithms = self.regression_algorithms
        elif task_type == 'classification':
            algorithms = self.classification_algorithms
        else:
            algorithms = self.clustering_algorithms
        
        # 算法评分和推荐
        recommendations = []
        for algo_key, algo_info in algorithms.items():
            score = self._calculate_algorithm_score(algo_key, characteristics, task_type)
            recommendations.append({
                'algorithm': algo_key,
                'name': algo_info['name'],
                'description': algo_info['description'],
                'score': score,
                'suitability': self._get_suitability_level(score)
            })
        
        # 按分数排序
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations
    
    def _calculate_algorithm_score(self, algorithm: str, characteristics: Dict[str, Any], 
                                 task_type: str) -> float:
        """
        计算算法适用性分数
        """
        score = 0.0
        
        # 基于数据大小的评分
        data_size = characteristics['data_size']
        if algorithm in ['linear_regression', 'logistic_regression', 'svm', 'svr']:
            if data_size in ['small', 'medium']:
                score += 0.3
        elif algorithm in ['random_forest_regression', 'random_forest_classification']:
            if data_size in ['medium', 'large']:
                score += 0.3
        
        # 基于关系类型的评分
        if 'relationship_type' in characteristics:
            rel_type = characteristics['relationship_type']
            if algorithm in ['linear_regression', 'logistic_regression'] and rel_type == 'linear':
                score += 0.4
            elif algorithm in ['polynomial_regression'] and rel_type == 'nonlinear':
                score += 0.4
            elif algorithm in ['random_forest_regression', 'random_forest_classification', 
                             'decision_tree_regression', 'decision_tree_classification'] and rel_type == 'complex':
                score += 0.3
        
        # 基于维度的评分
        if characteristics['dimensionality'] == 'high':
            if algorithm in ['svm', 'svr']:
                score += 0.2
        
        # 基于缺失值的评分
        if characteristics['missing_percentage'] > 0.1:
            if algorithm in ['random_forest_regression', 'random_forest_classification']:
                score += 0.1
        
        # 基于类别数量的评分（分类任务）
        if task_type == 'classification' and 'n_classes' in characteristics:
            n_classes = characteristics['n_classes']
            if n_classes == 2 and algorithm in ['logistic_regression', 'svm']:
                score += 0.2
            elif n_classes > 2 and algorithm in ['random_forest_classification', 'knn']:
                score += 0.2
        
        return min(score, 1.0)  # 限制最大分数为1.0
    
    def _get_suitability_level(self, score: float) -> str:
        """
        获取适用性等级
        """
        if score >= 0.7:
            return '高度推荐'
        elif score >= 0.5:
            return '推荐'
        elif score >= 0.3:
            return '可考虑'
        else:
            return '不推荐'
    
    def _get_recommendation_reasons(self, algorithm: str, characteristics: Dict[str, Any]) -> List[str]:
        """
        获取推荐理由
        """
        reasons = []
        
        # 基于数据大小的理由
        data_size = characteristics['data_size']
        if algorithm in ['linear_regression', 'logistic_regression', 'svm', 'svr']:
            if data_size in ['small', 'medium']:
                reasons.append(f'适合{data_size}规模数据集')
        elif algorithm in ['random_forest_regression', 'random_forest_classification']:
            if data_size in ['medium', 'large']:
                reasons.append(f'在{data_size}规模数据集上表现良好')
        
        # 基于关系类型的理由
        if 'relationship_type' in characteristics:
            rel_type = characteristics['relationship_type']
            if algorithm in ['linear_regression', 'logistic_regression'] and rel_type == 'linear':
                reasons.append('数据呈现线性关系')
            elif algorithm in ['polynomial_regression'] and rel_type == 'nonlinear':
                reasons.append('能够处理非线性关系')
            elif algorithm in ['random_forest_regression', 'random_forest_classification', 
                             'decision_tree_regression', 'decision_tree_classification'] and rel_type == 'complex':
                reasons.append('能够处理复杂的非线性关系')
        
        # 基于维度的理由
        if characteristics['dimensionality'] == 'high':
            if algorithm in ['svm', 'svr']:
                reasons.append('适合高维数据')
        
        # 基于缺失值的理由
        if characteristics['missing_percentage'] > 0.1:
            if algorithm in ['random_forest_regression', 'random_forest_classification']:
                reasons.append('能够处理缺失值')
        
        # 基于类别不平衡的理由
        if characteristics.get('class_balance', {}).get('is_balanced', True) == False:
            if algorithm in ['random_forest_classification', 'svm']:
                reasons.append('能够处理类别不平衡问题')
        
        # 如果没有特定理由，添加通用理由
        if not reasons:
            reasons.append('通用算法，适用于多种场景')
        
        return reasons
    
    def get_algorithm_details(self, algorithm: str) -> Dict[str, Any]:
        """
        获取算法详细信息
        """
        all_algorithms = {**self.regression_algorithms, **self.classification_algorithms, **self.clustering_algorithms}
        return all_algorithms.get(algorithm, {})