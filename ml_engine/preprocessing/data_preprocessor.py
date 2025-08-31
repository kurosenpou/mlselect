import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    数据预处理器 - 提供数据清洗、特征工程和数据转换功能
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}
        self.pca_transformers = {}
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean', 
                             columns: List[str] = None) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 输入数据框
            strategy: 填充策略 ('mean', 'median', 'mode', 'constant', 'knn')
            columns: 要处理的列，如果为None则处理所有列
        """
        try:
            df_processed = df.copy()
            
            if columns is None:
                columns = df.columns.tolist()
            
            for col in columns:
                if col not in df.columns:
                    continue
                    
                if df[col].isnull().sum() == 0:
                    continue
                
                if strategy == 'mean' and df[col].dtype in ['int64', 'float64']:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                elif strategy == 'median' and df[col].dtype in ['int64', 'float64']:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                elif strategy == 'mode':
                    mode_value = df_processed[col].mode()
                    if not mode_value.empty:
                        df_processed[col] = df_processed[col].fillna(mode_value[0])
                    else:
                        df_processed[col] = df_processed[col].fillna('Unknown')
                elif strategy == 'constant':
                    if df[col].dtype in ['int64', 'float64']:
                        df_processed[col] = df_processed[col].fillna(0)
                    else:
                        df_processed[col] = df_processed[col].fillna('Unknown')
                elif strategy == 'knn':
                    if df[col].dtype in ['int64', 'float64']:
                        imputer = KNNImputer(n_neighbors=5)
                        df_processed[[col]] = imputer.fit_transform(df_processed[[col]])
            
            return df_processed
            
        except Exception as e:
            raise Exception(f"处理缺失值失败: {str(e)}")
    
    def encode_categorical_features(self, df: pd.DataFrame, method: str = 'label', 
                                  columns: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        编码分类特征
        
        Args:
            df: 输入数据框
            method: 编码方法 ('label', 'onehot')
            columns: 要编码的列，如果为None则自动检测分类列
        """
        try:
            df_processed = df.copy()
            encoders = {}
            
            if columns is None:
                columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                if method == 'label':
                    encoder = LabelEncoder()
                    df_processed[col] = encoder.fit_transform(df_processed[col].astype(str))
                    encoders[col] = encoder
                    
                elif method == 'onehot':
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(df_processed[[col]])
                    feature_names = [f"{col}_{category}" for category in encoder.categories_[0]]
                    
                    # 创建编码后的数据框
                    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df_processed.index)
                    
                    # 删除原列并添加编码后的列
                    df_processed = df_processed.drop(columns=[col])
                    df_processed = pd.concat([df_processed, encoded_df], axis=1)
                    
                    encoders[col] = encoder
            
            return df_processed, encoders
            
        except Exception as e:
            raise Exception(f"分类特征编码失败: {str(e)}")
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard', 
                      columns: List[str] = None) -> Tuple[pd.DataFrame, Any]:
        """
        特征缩放
        
        Args:
            df: 输入数据框
            method: 缩放方法 ('standard', 'minmax', 'robust')
            columns: 要缩放的列，如果为None则缩放所有数值列
        """
        try:
            df_processed = df.copy()
            
            if columns is None:
                columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if not columns:
                return df_processed, None
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"不支持的缩放方法: {method}")
            
            df_processed[columns] = scaler.fit_transform(df_processed[columns])
            
            return df_processed, scaler
            
        except Exception as e:
            raise Exception(f"特征缩放失败: {str(e)}")
    
    def select_features(self, df: pd.DataFrame, target: pd.Series, method: str = 'k_best', 
                       k: int = 10, task_type: str = 'classification') -> Tuple[pd.DataFrame, Any]:
        """
        特征选择
        
        Args:
            df: 特征数据框
            target: 目标变量
            method: 选择方法 ('k_best', 'mutual_info')
            k: 选择的特征数量
            task_type: 任务类型 ('classification', 'regression')
        """
        try:
            if method == 'k_best':
                if task_type == 'classification':
                    selector = SelectKBest(score_func=f_classif, k=min(k, df.shape[1]))
                else:
                    selector = SelectKBest(score_func=f_regression, k=min(k, df.shape[1]))
                    
            elif method == 'mutual_info':
                if task_type == 'classification':
                    selector = SelectKBest(score_func=mutual_info_classif, k=min(k, df.shape[1]))
                else:
                    selector = SelectKBest(score_func=mutual_info_regression, k=min(k, df.shape[1]))
            else:
                raise ValueError(f"不支持的特征选择方法: {method}")
            
            X_selected = selector.fit_transform(df, target)
            selected_features = df.columns[selector.get_support()].tolist()
            df_selected = pd.DataFrame(X_selected, columns=selected_features, index=df.index)
            
            return df_selected, selector
            
        except Exception as e:
            raise Exception(f"特征选择失败: {str(e)}")
    
    def apply_pca(self, df: pd.DataFrame, n_components: Union[int, float] = 0.95) -> Tuple[pd.DataFrame, PCA]:
        """
        应用主成分分析(PCA)
        
        Args:
            df: 输入数据框
            n_components: 主成分数量或方差解释比例
        """
        try:
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(df)
            
            # 创建主成分列名
            n_components_actual = X_pca.shape[1]
            columns = [f'PC{i+1}' for i in range(n_components_actual)]
            
            df_pca = pd.DataFrame(X_pca, columns=columns, index=df.index)
            
            return df_pca, pca
            
        except Exception as e:
            raise Exception(f"PCA转换失败: {str(e)}")
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                       columns: List[str] = None) -> Dict[str, Any]:
        """
        检测异常值
        
        Args:
            df: 输入数据框
            method: 检测方法 ('iqr', 'zscore')
            columns: 要检测的列，如果为None则检测所有数值列
        """
        try:
            if columns is None:
                columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            outliers_info = {}
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                outliers_mask = pd.Series([False] * len(df), index=df.index)
                
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outliers_mask = z_scores > 3
                
                outliers_info[col] = {
                    'outliers_count': outliers_mask.sum(),
                    'outliers_percentage': (outliers_mask.sum() / len(df)) * 100,
                    'outliers_indices': df[outliers_mask].index.tolist()
                }
            
            return outliers_info
            
        except Exception as e:
            raise Exception(f"异常值检测失败: {str(e)}")
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                       columns: List[str] = None) -> pd.DataFrame:
        """
        移除异常值
        
        Args:
            df: 输入数据框
            method: 检测方法 ('iqr', 'zscore')
            columns: 要处理的列，如果为None则处理所有数值列
        """
        try:
            if columns is None:
                columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            df_cleaned = df.copy()
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                if method == 'iqr':
                    Q1 = df_cleaned[col].quantile(0.25)
                    Q3 = df_cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                    
                elif method == 'zscore':
                    z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
                    df_cleaned = df_cleaned[z_scores <= 3]
            
            return df_cleaned
            
        except Exception as e:
            raise Exception(f"移除异常值失败: {str(e)}")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        获取数据摘要信息
        """
        try:
            summary = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'duplicated_rows': df.duplicated().sum()
            }
            
            # 数值列统计
            if summary['numeric_columns']:
                summary['numeric_stats'] = df[summary['numeric_columns']].describe().to_dict()
            
            # 分类列统计
            if summary['categorical_columns']:
                categorical_stats = {}
                for col in summary['categorical_columns']:
                    categorical_stats[col] = {
                        'unique_count': df[col].nunique(),
                        'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                        'frequency': df[col].value_counts().head().to_dict()
                    }
                summary['categorical_stats'] = categorical_stats
            
            return summary
            
        except Exception as e:
            raise Exception(f"获取数据摘要失败: {str(e)}")
    
    def preprocess_pipeline(self, df: pd.DataFrame, target_column: str = None, 
                          config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        完整的数据预处理管道
        
        Args:
            df: 输入数据框
            target_column: 目标列名
            config: 预处理配置
        """
        try:
            if config is None:
                config = {
                    'handle_missing': True,
                    'missing_strategy': 'mean',
                    'encode_categorical': True,
                    'encoding_method': 'label',
                    'scale_features': True,
                    'scaling_method': 'standard',
                    'remove_outliers': False,
                    'outlier_method': 'iqr',
                    'feature_selection': False,
                    'selection_method': 'k_best',
                    'n_features': 10,
                    'apply_pca': False,
                    'pca_components': 0.95
                }
            
            df_processed = df.copy()
            preprocessing_info = {
                'original_shape': df.shape,
                'steps_applied': [],
                'transformers': {}
            }
            
            # 分离特征和目标
            if target_column and target_column in df.columns:
                X = df_processed.drop(columns=[target_column])
                y = df_processed[target_column]
            else:
                X = df_processed
                y = None
            
            # 1. 处理缺失值
            if config.get('handle_missing', True):
                X = self.handle_missing_values(X, strategy=config.get('missing_strategy', 'mean'))
                preprocessing_info['steps_applied'].append('handle_missing_values')
            
            # 2. 编码分类特征
            if config.get('encode_categorical', True):
                X, encoders = self.encode_categorical_features(X, method=config.get('encoding_method', 'label'))
                preprocessing_info['transformers']['encoders'] = encoders
                preprocessing_info['steps_applied'].append('encode_categorical_features')
            
            # 3. 移除异常值
            if config.get('remove_outliers', False):
                X = self.remove_outliers(X, method=config.get('outlier_method', 'iqr'))
                preprocessing_info['steps_applied'].append('remove_outliers')
            
            # 4. 特征缩放
            if config.get('scale_features', True):
                X, scaler = self.scale_features(X, method=config.get('scaling_method', 'standard'))
                preprocessing_info['transformers']['scaler'] = scaler
                preprocessing_info['steps_applied'].append('scale_features')
            
            # 5. 特征选择
            if config.get('feature_selection', False) and y is not None:
                task_type = 'classification' if y.dtype == 'object' or y.nunique() < 20 else 'regression'
                X, selector = self.select_features(X, y, 
                                                 method=config.get('selection_method', 'k_best'),
                                                 k=config.get('n_features', 10),
                                                 task_type=task_type)
                preprocessing_info['transformers']['feature_selector'] = selector
                preprocessing_info['steps_applied'].append('feature_selection')
            
            # 6. PCA降维
            if config.get('apply_pca', False):
                X, pca = self.apply_pca(X, n_components=config.get('pca_components', 0.95))
                preprocessing_info['transformers']['pca'] = pca
                preprocessing_info['steps_applied'].append('apply_pca')
            
            preprocessing_info['final_shape'] = X.shape
            preprocessing_info['feature_columns'] = X.columns.tolist()
            
            result = {
                'X': X,
                'y': y,
                'preprocessing_info': preprocessing_info
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"预处理管道执行失败: {str(e)}")