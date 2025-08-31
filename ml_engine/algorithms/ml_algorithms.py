import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MLAlgorithms:
    """
    机器学习算法实现类 - 集成多种算法的训练、预测和评估功能
    """
    
    def __init__(self, model_save_path: str = "d:/ML_select/data/models"):
        self.model_save_path = model_save_path
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
        self.target_column = {}
        
        # 确保模型保存目录存在
        os.makedirs(model_save_path, exist_ok=True)
        
        # 算法配置
        self.algorithm_configs = {
            'linear_regression': {
                'class': LinearRegression,
                'params': {},
                'type': 'regression'
            },
            'polynomial_regression': {
                'class': LinearRegression,
                'params': {},
                'type': 'regression',
                'preprocessing': 'polynomial'
            },
            'random_forest_regression': {
                'class': RandomForestRegressor,
                'params': {'n_estimators': 100, 'random_state': 42},
                'type': 'regression'
            },
            'svr': {
                'class': SVR,
                'params': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
                'type': 'regression'
            },
            'decision_tree_regression': {
                'class': DecisionTreeRegressor,
                'params': {'random_state': 42},
                'type': 'regression'
            },
            'logistic_regression': {
                'class': LogisticRegression,
                'params': {'random_state': 42, 'max_iter': 1000},
                'type': 'classification'
            },
            'random_forest_classification': {
                'class': RandomForestClassifier,
                'params': {'n_estimators': 100, 'random_state': 42},
                'type': 'classification'
            },
            'svm': {
                'class': SVC,
                'params': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'random_state': 42},
                'type': 'classification'
            },
            'knn': {
                'class': KNeighborsClassifier,
                'params': {'n_neighbors': 5},
                'type': 'classification'
            },
            'naive_bayes': {
                'class': GaussianNB,
                'params': {},
                'type': 'classification'
            },
            'decision_tree_classification': {
                'class': DecisionTreeClassifier,
                'params': {'random_state': 42},
                'type': 'classification'
            },
            'kmeans': {
                'class': KMeans,
                'params': {'n_clusters': 3, 'random_state': 42},
                'type': 'clustering'
            },
            'hierarchical': {
                'class': AgglomerativeClustering,
                'params': {'n_clusters': 3},
                'type': 'clustering'
            }
        }
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, feature_columns: List[str], 
                    test_size: float = 0.2, algorithm: str = None) -> Dict[str, Any]:
        """
        准备训练数据
        """
        # 复制数据
        data = df.copy()
        
        # 处理缺失值
        for col in feature_columns:
            if data[col].dtype in ['int64', 'float64']:
                data[col].fillna(data[col].mean(), inplace=True)
            else:
                data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else 'unknown', inplace=True)
        
        if target_column in data.columns:
            if data[target_column].dtype in ['int64', 'float64']:
                data[target_column].fillna(data[target_column].mean(), inplace=True)
            else:
                data[target_column].fillna(data[target_column].mode().iloc[0] if not data[target_column].mode().empty else 'unknown', inplace=True)
        
        # 准备特征和目标变量
        X = data[feature_columns].copy()
        y = data[target_column].copy() if target_column in data.columns else None
        
        # 编码分类特征
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        
        # 编码目标变量（如果是分类任务）
        target_encoder = None
        if y is not None and y.dtype == 'object':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
        
        # 特殊预处理（多项式特征）
        if algorithm == 'polynomial_regression':
            poly_features = PolynomialFeatures(degree=2, include_bias=False)
            X = pd.DataFrame(poly_features.fit_transform(X))
            encoders['polynomial'] = poly_features
        
        # 标准化数值特征
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # 分割数据
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )
        else:
            X_train, X_test = train_test_split(X_scaled, test_size=test_size, random_state=42)
            y_train, y_test = None, None
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'encoders': encoders,
            'target_encoder': target_encoder,
            'feature_columns': feature_columns,
            'original_data': data
        }
    
    def train_model(self, algorithm: str, data_dict: Dict[str, Any], 
                   custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        训练模型
        """
        if algorithm not in self.algorithm_configs:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        config = self.algorithm_configs[algorithm]
        
        # 合并参数
        params = config['params'].copy()
        if custom_params:
            params.update(custom_params)
        
        # 创建模型
        model = config['class'](**params)
        
        # 训练模型
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        
        if config['type'] == 'clustering':
            # 聚类算法
            model.fit(X_train)
            predictions = model.labels_ if hasattr(model, 'labels_') else model.predict(X_train)
        else:
            # 监督学习算法
            model.fit(X_train, y_train)
            predictions = model.predict(X_train)
        
        # 保存模型和相关信息
        model_id = f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.models[model_id] = model
        self.scalers[model_id] = data_dict['scaler']
        self.encoders[model_id] = data_dict['encoders']
        self.feature_columns[model_id] = data_dict['feature_columns']
        
        if data_dict['target_encoder']:
            self.encoders[model_id]['target'] = data_dict['target_encoder']
        
        # 评估模型
        evaluation = self.evaluate_model(model_id, data_dict, algorithm)
        
        return {
            'model_id': model_id,
            'algorithm': algorithm,
            'model': model,
            'training_predictions': predictions,
            'evaluation': evaluation,
            'parameters': params
        }
    
    def evaluate_model(self, model_id: str, data_dict: Dict[str, Any], algorithm: str) -> Dict[str, Any]:
        """
        评估模型
        """
        model = self.models[model_id]
        config = self.algorithm_configs[algorithm]
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        evaluation = {
            'algorithm': algorithm,
            'model_type': config['type']
        }
        
        if config['type'] == 'regression':
            # 回归评估
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            evaluation.update({
                'train_mse': mean_squared_error(y_train, train_pred),
                'test_mse': mean_squared_error(y_test, test_pred),
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred))
            })
            
        elif config['type'] == 'classification':
            # 分类评估
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            evaluation.update({
                'train_accuracy': accuracy_score(y_train, train_pred),
                'test_accuracy': accuracy_score(y_test, test_pred),
                'classification_report': classification_report(y_test, test_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
            })
            
            # 交叉验证
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                evaluation['cv_accuracy_mean'] = cv_scores.mean()
                evaluation['cv_accuracy_std'] = cv_scores.std()
            except:
                pass
                
        elif config['type'] == 'clustering':
            # 聚类评估
            labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_train)
            
            # 计算轮廓系数
            try:
                from sklearn.metrics import silhouette_score
                silhouette_avg = silhouette_score(X_train, labels)
                evaluation['silhouette_score'] = silhouette_avg
            except:
                evaluation['silhouette_score'] = None
            
            # 计算惯性（对于KMeans）
            if hasattr(model, 'inertia_'):
                evaluation['inertia'] = model.inertia_
            
            evaluation['n_clusters'] = len(np.unique(labels))
            evaluation['cluster_sizes'] = np.bincount(labels).tolist()
        
        return evaluation
    
    def predict(self, model_id: str, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        使用训练好的模型进行预测
        """
        if model_id not in self.models:
            raise ValueError(f"模型 {model_id} 不存在")
        
        model = self.models[model_id]
        scaler = self.scalers[model_id]
        encoders = self.encoders[model_id]
        feature_columns = self.feature_columns[model_id]
        
        # 准备数据
        X = new_data[feature_columns].copy()
        
        # 处理缺失值
        for col in feature_columns:
            if X[col].dtype in ['int64', 'float64']:
                X[col].fillna(X[col].mean(), inplace=True)
            else:
                X[col].fillna('unknown', inplace=True)
        
        # 编码分类特征
        for col, encoder in encoders.items():
            if col in X.columns and col != 'target' and col != 'polynomial':
                try:
                    X[col] = encoder.transform(X[col].astype(str))
                except ValueError:
                    # 处理未见过的类别
                    X[col] = 0
        
        # 多项式特征处理
        if 'polynomial' in encoders:
            X = pd.DataFrame(encoders['polynomial'].transform(X))
        
        # 标准化
        X_scaled = scaler.transform(X)
        
        # 预测
        predictions = model.predict(X_scaled)
        
        # 如果有目标编码器，进行反向编码
        if 'target' in encoders:
            predictions = encoders['target'].inverse_transform(predictions.astype(int))
        
        # 预测概率（对于分类任务）
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X_scaled)
            except:
                pass
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'model_id': model_id,
            'n_samples': len(predictions)
        }
    
    def save_model(self, model_id: str, filename: str = None) -> str:
        """
        保存模型到磁盘
        """
        if model_id not in self.models:
            raise ValueError(f"模型 {model_id} 不存在")
        
        if filename is None:
            filename = f"{model_id}.joblib"
        
        filepath = os.path.join(self.model_save_path, filename)
        
        model_data = {
            'model': self.models[model_id],
            'scaler': self.scalers[model_id],
            'encoders': self.encoders[model_id],
            'feature_columns': self.feature_columns[model_id],
            'model_id': model_id
        }
        
        joblib.dump(model_data, filepath)
        return filepath
    
    def load_model(self, filepath: str) -> str:
        """
        从磁盘加载模型
        """
        model_data = joblib.load(filepath)
        
        model_id = model_data['model_id']
        self.models[model_id] = model_data['model']
        self.scalers[model_id] = model_data['scaler']
        self.encoders[model_id] = model_data['encoders']
        self.feature_columns[model_id] = model_data['feature_columns']
        
        return model_id
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        获取模型信息
        """
        if model_id not in self.models:
            raise ValueError(f"模型 {model_id} 不存在")
        
        model = self.models[model_id]
        
        info = {
            'model_id': model_id,
            'model_type': type(model).__name__,
            'feature_columns': self.feature_columns[model_id],
            'n_features': len(self.feature_columns[model_id])
        }
        
        # 添加模型特定信息
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(
                self.feature_columns[model_id], 
                model.feature_importances_
            ))
            info['feature_importance'] = feature_importance
        
        if hasattr(model, 'coef_'):
            coefficients = dict(zip(
                self.feature_columns[model_id], 
                model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
            ))
            info['coefficients'] = coefficients
        
        if hasattr(model, 'n_clusters'):
            info['n_clusters'] = model.n_clusters
        
        return info
    
    def list_models(self) -> List[str]:
        """
        列出所有已训练的模型
        """
        return list(self.models.keys())
    
    def delete_model(self, model_id: str) -> bool:
        """
        删除模型
        """
        if model_id not in self.models:
            return False
        
        del self.models[model_id]
        del self.scalers[model_id]
        del self.encoders[model_id]
        del self.feature_columns[model_id]
        
        return True
    
    def get_algorithm_info(self, algorithm: str) -> Dict[str, Any]:
        """
        获取算法信息
        """
        if algorithm not in self.algorithm_configs:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        config = self.algorithm_configs[algorithm]
        return {
            'algorithm': algorithm,
            'type': config['type'],
            'class': config['class'].__name__,
            'default_params': config['params'],
            'preprocessing': config.get('preprocessing', 'standard')
        }
    
    def get_supported_algorithms(self) -> Dict[str, List[str]]:
        """
        获取支持的算法列表
        """
        algorithms = {
            'regression': [],
            'classification': [],
            'clustering': []
        }
        
        for algo, config in self.algorithm_configs.items():
            algorithms[config['type']].append(algo)
        
        return algorithms