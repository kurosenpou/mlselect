import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.cluster import KMeans, AgglomerativeClustering
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ClusteringAlgorithms:
    """
    聚类算法实现类 - 专门处理聚类任务的机器学习算法
    """
    
    def __init__(self, model_save_path: str = "d:/ML_select/data/models"):
        self.model_save_path = model_save_path
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
        
        # 确保模型保存目录存在
        os.makedirs(model_save_path, exist_ok=True)
        
        # 聚类算法配置
        self.algorithm_configs = {
            'kmeans': {
                'class': KMeans,
                'params': {'n_clusters': 3, 'random_state': 42, 'n_init': 10},
                'type': 'clustering'
            },
            'hierarchical_clustering': {
                'class': AgglomerativeClustering,
                'params': {'n_clusters': 3, 'linkage': 'ward'},
                'type': 'clustering'
            }
        }
    
    def prepare_data(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """
        准备聚类数据
        """
        try:
            # 检查数据
            if df.empty:
                raise ValueError("数据集为空")
            
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                raise ValueError(f"特征列不存在: {missing_features}")
            
            # 提取特征
            X = df[feature_columns].copy()
            
            # 处理缺失值
            encoders = {}
            for col in X.columns:
                if X[col].dtype in ['object', 'category']:
                    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
                    # 编码分类特征
                    encoder = LabelEncoder()
                    X[col] = encoder.fit_transform(X[col].astype(str))
                    encoders[col] = encoder
                else:
                    X[col] = X[col].fillna(X[col].mean())
            
            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
            
            return {
                'X': X_scaled,
                'scaler': scaler,
                'encoders': encoders,
                'feature_columns': feature_columns,
                'original_X': X
            }
            
        except Exception as e:
            raise Exception(f"数据准备失败: {str(e)}")
    
    def train_model(self, algorithm: str, data_dict: Dict[str, Any], 
                   custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        训练聚类模型
        """
        try:
            if algorithm not in self.algorithm_configs:
                raise ValueError(f"不支持的聚类算法: {algorithm}")
            
            config = self.algorithm_configs[algorithm]
            params = config['params'].copy()
            if custom_params:
                params.update(custom_params)
            
            # 创建模型
            model = config['class'](**params)
            
            # 训练模型
            X = data_dict['X']
            cluster_labels = model.fit_predict(X)
            
            # 生成模型ID
            model_id = f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 保存模型信息
            self.models[model_id] = model
            self.scalers[model_id] = data_dict['scaler']
            self.encoders[model_id] = data_dict['encoders']
            self.feature_columns[model_id] = data_dict['feature_columns']
            
            return {
                'model_id': model_id,
                'algorithm': algorithm,
                'model': model,
                'cluster_labels': cluster_labels,
                'parameters': params,
                'n_clusters': len(np.unique(cluster_labels))
            }
            
        except Exception as e:
            raise Exception(f"模型训练失败: {str(e)}")
    
    def evaluate_model(self, model_id: str, data_dict: Dict[str, Any], 
                      cluster_labels: np.ndarray) -> Dict[str, Any]:
        """
        评估聚类模型
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"模型 {model_id} 不存在")
            
            X = data_dict['X']
            
            # 计算聚类评估指标
            n_clusters = len(np.unique(cluster_labels))
            
            metrics = {}
            
            # 轮廓系数
            if n_clusters > 1 and len(X) > n_clusters:
                silhouette_avg = silhouette_score(X, cluster_labels)
                metrics['silhouette_score'] = silhouette_avg
            
            # Calinski-Harabasz指数
            if n_clusters > 1:
                ch_score = calinski_harabasz_score(X, cluster_labels)
                metrics['calinski_harabasz_score'] = ch_score
            
            # 簇内平方和
            if hasattr(self.models[model_id], 'inertia_'):
                metrics['inertia'] = self.models[model_id].inertia_
            
            # 计算每个簇的统计信息
            cluster_stats = {}
            for cluster_id in np.unique(cluster_labels):
                cluster_mask = cluster_labels == cluster_id
                cluster_data = X[cluster_mask]
                cluster_stats[f'cluster_{cluster_id}'] = {
                    'size': int(np.sum(cluster_mask)),
                    'percentage': float(np.sum(cluster_mask) / len(cluster_labels) * 100),
                    'center': cluster_data.mean().tolist()
                }
            
            return {
                'model_id': model_id,
                'cluster_labels': cluster_labels.tolist(),
                'metrics': metrics,
                'cluster_statistics': cluster_stats,
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            raise Exception(f"模型评估失败: {str(e)}")
    
    def predict(self, model_id: str, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        使用聚类模型进行预测
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"模型 {model_id} 不存在")
            
            model = self.models[model_id]
            scaler = self.scalers[model_id]
            encoders = self.encoders[model_id]
            feature_columns = self.feature_columns[model_id]
            
            # 检查特征列
            missing_features = [col for col in feature_columns if col not in new_data.columns]
            if missing_features:
                raise ValueError(f"缺少特征列: {missing_features}")
            
            # 准备数据
            X_new = new_data[feature_columns].copy()
            
            # 处理缺失值和编码
            for col in X_new.columns:
                if X_new[col].dtype in ['object', 'category']:
                    X_new[col] = X_new[col].fillna('Unknown')
                    if col in encoders:
                        # 处理未见过的类别
                        X_new[col] = X_new[col].astype(str)
                        known_classes = set(encoders[col].classes_)
                        X_new[col] = X_new[col].apply(lambda x: x if x in known_classes else encoders[col].classes_[0])
                        X_new[col] = encoders[col].transform(X_new[col])
                else:
                    X_new[col] = X_new[col].fillna(X_new[col].mean())
            
            # 标准化
            X_new_scaled = scaler.transform(X_new)
            X_new_scaled = pd.DataFrame(X_new_scaled, columns=feature_columns, index=X_new.index)
            
            # 预测聚类标签
            cluster_labels = model.predict(X_new_scaled)
            
            # 计算到聚类中心的距离（如果支持）
            distances = None
            if hasattr(model, 'transform'):
                distances = model.transform(X_new_scaled)
            
            return {
                'model_id': model_id,
                'cluster_labels': cluster_labels.tolist(),
                'distances_to_centers': distances.tolist() if distances is not None else None,
                'input_data': new_data.to_dict('records'),
                'feature_columns': feature_columns
            }
            
        except Exception as e:
            raise Exception(f"预测失败: {str(e)}")
    
    def get_cluster_centers(self, model_id: str) -> Dict[str, Any]:
        """
        获取聚类中心
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"模型 {model_id} 不存在")
            
            model = self.models[model_id]
            feature_columns = self.feature_columns[model_id]
            
            centers = None
            if hasattr(model, 'cluster_centers_'):
                centers = model.cluster_centers_
            
            return {
                'model_id': model_id,
                'cluster_centers': centers.tolist() if centers is not None else None,
                'feature_columns': feature_columns,
                'n_clusters': len(centers) if centers is not None else None
            }
            
        except Exception as e:
            raise Exception(f"获取聚类中心失败: {str(e)}")
    
    def get_supported_algorithms(self) -> List[str]:
        """
        获取支持的聚类算法列表
        """
        return list(self.algorithm_configs.keys())
    
    def get_algorithm_info(self, algorithm: str) -> Dict[str, Any]:
        """
        获取算法信息
        """
        if algorithm not in self.algorithm_configs:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        config = self.algorithm_configs[algorithm]
        return {
            'name': algorithm,
            'type': config['type'],
            'class': config['class'].__name__,
            'default_params': config['params']
        }