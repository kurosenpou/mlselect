import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class FeatureAnalyzer:
    """
    特征分析器 - 分析数据特征的统计属性、质量和重要性
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def analyze_features(self, X: pd.DataFrame, y: pd.DataFrame = None, 
                        df: pd.DataFrame = None, target_column: str = None, 
                        feature_columns: List[str] = None) -> Dict[str, Any]:
        """
        全面分析特征
        支持两种调用方式：
        1. analyze_features(X, y) - 直接传入特征和目标变量
        2. analyze_features(df=df, target_column=target_column, feature_columns=feature_columns) - 传入DataFrame
        """
        # 兼容新的调用方式
        if df is None:
            # 使用X和y构建临时DataFrame
            if isinstance(X, pd.DataFrame):
                df = X.copy()
                feature_columns = X.columns.tolist()
            else:
                df = pd.DataFrame(X)
                feature_columns = df.columns.tolist()
            
            if y is not None:
                if isinstance(y, pd.DataFrame):
                    target_column = y.columns[0] if len(y.columns) == 1 else 'target'
                    df[target_column] = y.iloc[:, 0] if len(y.columns) == 1 else y
                else:
                    target_column = 'target'
                    df[target_column] = y
        else:
            # 使用原有的DataFrame方式
            if feature_columns is None:
                feature_columns = [col for col in df.columns if col != target_column]
        
        analysis_result = {
            'basic_info': self._get_basic_info(df, feature_columns),
            'statistical_summary': self._get_statistical_summary(df, feature_columns),
            'data_quality': self._analyze_data_quality(df, feature_columns),
            'feature_types': self._analyze_feature_types(df, feature_columns),
            'correlations': self._analyze_correlations(df, feature_columns),
            'distributions': self._analyze_distributions(df, feature_columns)
        }
        
        # 如果有目标变量，分析特征重要性
        if target_column and target_column in df.columns:
            analysis_result['target_analysis'] = self._analyze_target_relationship(
                df, target_column, feature_columns
            )
            analysis_result['feature_importance'] = self._calculate_feature_importance(
                df, target_column, feature_columns
            )
        
        return analysis_result
    
    def _get_basic_info(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """
        获取基本信息
        """
        return {
            'n_samples': len(df),
            'n_features': len(feature_columns),
            'memory_usage': df[feature_columns].memory_usage(deep=True).sum(),
            'feature_names': feature_columns
        }
    
    def _get_statistical_summary(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """
        获取统计摘要
        """
        numerical_cols = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df[feature_columns].select_dtypes(exclude=[np.number]).columns.tolist()
        
        summary = {
            'numerical_features': len(numerical_cols),
            'categorical_features': len(categorical_cols)
        }
        
        if numerical_cols:
            numerical_summary = df[numerical_cols].describe()
            summary['numerical_summary'] = {
                col: {
                    'mean': numerical_summary.loc['mean', col],
                    'std': numerical_summary.loc['std', col],
                    'min': numerical_summary.loc['min', col],
                    'max': numerical_summary.loc['max', col],
                    'q25': numerical_summary.loc['25%', col],
                    'q50': numerical_summary.loc['50%', col],
                    'q75': numerical_summary.loc['75%', col]
                } for col in numerical_cols
            }
        
        if categorical_cols:
            summary['categorical_summary'] = {
                col: {
                    'unique_values': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    'most_frequent_count': df[col].value_counts().iloc[0] if not df[col].empty else 0
                } for col in categorical_cols
            }
        
        return summary
    
    def _analyze_data_quality(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """
        分析数据质量
        """
        quality_report = {}
        
        for col in feature_columns:
            col_data = df[col]
            
            # 缺失值分析
            missing_count = col_data.isnull().sum()
            missing_percentage = missing_count / len(col_data) * 100
            
            # 重复值分析
            duplicate_count = col_data.duplicated().sum()
            duplicate_percentage = duplicate_count / len(col_data) * 100
            
            # 唯一值分析
            unique_count = col_data.nunique()
            unique_percentage = unique_count / len(col_data) * 100
            
            quality_info = {
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'duplicate_count': duplicate_count,
                'duplicate_percentage': duplicate_percentage,
                'unique_count': unique_count,
                'unique_percentage': unique_percentage,
                'data_type': str(col_data.dtype)
            }
            
            # 数值型特征的额外质量检查
            if col_data.dtype in [np.number, 'int64', 'float64']:
                # 异常值检测（IQR方法）
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                quality_info['outlier_count'] = len(outliers)
                quality_info['outlier_percentage'] = len(outliers) / len(col_data) * 100
                
                # 零值检查
                zero_count = (col_data == 0).sum()
                quality_info['zero_count'] = zero_count
                quality_info['zero_percentage'] = zero_count / len(col_data) * 100
            
            quality_report[col] = quality_info
        
        return quality_report
    
    def _analyze_feature_types(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """
        分析特征类型
        """
        type_analysis = {
            'numerical': [],
            'categorical': [],
            'binary': [],
            'datetime': [],
            'text': []
        }
        
        for col in feature_columns:
            col_data = df[col]
            
            if col_data.dtype in ['int64', 'float64']:
                # 检查是否为二元特征
                if col_data.nunique() == 2:
                    type_analysis['binary'].append(col)
                else:
                    type_analysis['numerical'].append(col)
            elif col_data.dtype == 'datetime64[ns]':
                type_analysis['datetime'].append(col)
            elif col_data.dtype == 'object':
                # 检查是否为二元分类特征
                if col_data.nunique() == 2:
                    type_analysis['binary'].append(col)
                # 检查是否为文本特征（平均长度大于10）
                elif col_data.astype(str).str.len().mean() > 10:
                    type_analysis['text'].append(col)
                else:
                    type_analysis['categorical'].append(col)
            else:
                type_analysis['categorical'].append(col)
        
        return type_analysis
    
    def _analyze_correlations(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """
        分析特征间相关性
        """
        numerical_cols = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            return {'message': '数值特征少于2个，无法计算相关性'}
        
        # 计算相关性矩阵
        corr_matrix = df[numerical_cols].corr()
        
        # 找出高相关性特征对
        high_corr_pairs = []
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # 高相关性阈值
                    high_corr_pairs.append({
                        'feature1': numerical_cols[i],
                        'feature2': numerical_cols[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlation_pairs': high_corr_pairs,
            'max_correlation': corr_matrix.abs().max().max(),
            'mean_correlation': corr_matrix.abs().mean().mean()
        }
    
    def _analyze_distributions(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """
        分析特征分布
        """
        distribution_analysis = {}
        
        numerical_cols = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numerical_cols:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # 基本统计量
            skewness = stats.skew(col_data)
            kurtosis = stats.kurtosis(col_data)
            
            # 正态性检验
            try:
                _, p_value = stats.normaltest(col_data)
                is_normal = p_value > 0.05
            except:
                is_normal = False
                p_value = None
            
            distribution_analysis[col] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal': is_normal,
                'normality_p_value': p_value,
                'distribution_type': self._classify_distribution(skewness, kurtosis, is_normal)
            }
        
        return distribution_analysis
    
    def _classify_distribution(self, skewness: float, kurtosis: float, is_normal: bool) -> str:
        """
        分类分布类型
        """
        if is_normal:
            return '正态分布'
        elif abs(skewness) < 0.5:
            return '近似正态分布'
        elif skewness > 0.5:
            return '右偏分布'
        elif skewness < -0.5:
            return '左偏分布'
        else:
            return '未知分布'
    
    def _analyze_target_relationship(self, df: pd.DataFrame, target_column: str, 
                                   feature_columns: List[str]) -> Dict[str, Any]:
        """
        分析特征与目标变量的关系
        """
        target_data = df[target_column]
        target_type = 'continuous' if target_data.dtype in ['int64', 'float64'] and target_data.nunique() > 10 else 'categorical'
        
        relationship_analysis = {
            'target_type': target_type,
            'target_unique_values': target_data.nunique()
        }
        
        if target_type == 'continuous':
            relationship_analysis['target_stats'] = {
                'mean': target_data.mean(),
                'std': target_data.std(),
                'min': target_data.min(),
                'max': target_data.max()
            }
        else:
            relationship_analysis['target_distribution'] = target_data.value_counts().to_dict()
        
        # 计算每个特征与目标变量的关系强度
        feature_relationships = {}
        numerical_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        
        for feature in numerical_features:
            feature_data = df[feature].dropna()
            target_subset = target_data[feature_data.index]
            
            if target_type == 'continuous':
                # 计算皮尔逊相关系数
                correlation, p_value = stats.pearsonr(feature_data, target_subset)
                feature_relationships[feature] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'relationship_strength': self._classify_correlation_strength(abs(correlation))
                }
            else:
                # 对于分类目标，计算方差分析F值
                try:
                    groups = [feature_data[target_subset == class_] for class_ in target_subset.unique()]
                    f_stat, p_value = stats.f_oneway(*groups)
                    feature_relationships[feature] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'relationship_strength': 'significant' if p_value < 0.05 else 'not_significant'
                    }
                except:
                    feature_relationships[feature] = {
                        'f_statistic': None,
                        'p_value': None,
                        'relationship_strength': 'unknown'
                    }
        
        relationship_analysis['feature_relationships'] = feature_relationships
        return relationship_analysis
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """
        分类相关性强度
        """
        if correlation >= 0.7:
            return 'strong'
        elif correlation >= 0.3:
            return 'moderate'
        elif correlation >= 0.1:
            return 'weak'
        else:
            return 'very_weak'
    
    def _calculate_feature_importance(self, df: pd.DataFrame, target_column: str, 
                                    feature_columns: List[str]) -> Dict[str, Any]:
        """
        计算特征重要性
        """
        # 准备数据
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # 处理缺失值
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
        y = y.fillna(y.mean() if y.dtype in ['int64', 'float64'] else y.mode().iloc[0])
        
        # 编码分类特征
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # 确定任务类型
        task_type = 'regression' if y.dtype in ['int64', 'float64'] and y.nunique() > 10 else 'classification'
        
        try:
            if task_type == 'regression':
                # 使用互信息进行回归特征重要性计算
                importance_scores = mutual_info_regression(X, y, random_state=42)
            else:
                # 编码目标变量
                if y.dtype == 'object':
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(y)
                # 使用互信息进行分类特征重要性计算
                importance_scores = mutual_info_classif(X, y, random_state=42)
            
            # 创建特征重要性字典
            feature_importance = dict(zip(feature_columns, importance_scores))
            
            # 排序
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'task_type': task_type,
                'feature_importance': feature_importance,
                'sorted_importance': sorted_importance,
                'top_features': [item[0] for item in sorted_importance[:5]]
            }
        
        except Exception as e:
            return {
                'error': f'特征重要性计算失败: {str(e)}',
                'task_type': task_type
            }
    
    def generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        基于分析结果生成建议
        """
        recommendations = []
        
        # 数据质量建议
        if 'data_quality' in analysis_result:
            for feature, quality in analysis_result['data_quality'].items():
                if quality['missing_percentage'] > 10:
                    recommendations.append(f"特征 '{feature}' 有 {quality['missing_percentage']:.1f}% 的缺失值，建议进行缺失值处理")
                
                if quality.get('outlier_percentage', 0) > 5:
                    recommendations.append(f"特征 '{feature}' 有 {quality['outlier_percentage']:.1f}% 的异常值，建议进行异常值处理")
        
        # 相关性建议
        if 'correlations' in analysis_result and 'high_correlation_pairs' in analysis_result['correlations']:
            high_corr_pairs = analysis_result['correlations']['high_correlation_pairs']
            if high_corr_pairs:
                recommendations.append(f"发现 {len(high_corr_pairs)} 对高相关性特征，建议考虑特征选择或降维")
        
        # 分布建议
        if 'distributions' in analysis_result:
            skewed_features = []
            for feature, dist_info in analysis_result['distributions'].items():
                if abs(dist_info['skewness']) > 1:
                    skewed_features.append(feature)
            
            if skewed_features:
                recommendations.append(f"特征 {', '.join(skewed_features)} 分布偏斜，建议进行数据变换")
        
        # 特征重要性建议
        if 'feature_importance' in analysis_result and 'top_features' in analysis_result['feature_importance']:
            top_features = analysis_result['feature_importance']['top_features']
            recommendations.append(f"最重要的特征是: {', '.join(top_features[:3])}，建议重点关注这些特征")
        
        return recommendations