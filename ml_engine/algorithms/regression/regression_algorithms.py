import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RegressionAlgorithms:
    """
    回归算法实现类 - 专门处理回归任务的机器学习算法
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
        
        # 回归算法配置
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
            }
        }
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, feature_columns: List[str], 
                    test_size: float = 0.2, algorithm: str = None) -> Dict[str, Any]:
        """
        准备回归数据
        """
        try:
            # 检查数据
            if df.empty:
                raise ValueError("数据集为空")
            
            if target_column not in df.columns:
                raise ValueError(f"目标列 '{target_column}' 不存在")
            
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                raise ValueError(f"特征列不存在: {missing_features}")
            
            # 提取特征和目标
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # 处理缺失值
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # 数据预处理
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
            
            # 多项式特征（如果需要）
            if algorithm == 'polynomial_regression':
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly.fit_transform(X_scaled)
                feature_names = poly.get_feature_names_out(feature_columns)
                X_scaled = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'scaler': scaler,
                'feature_columns': feature_columns,
                'target_column': target_column,
                'original_X': X,
                'original_y': y
            }
            
        except Exception as e:
            raise Exception(f"数据准备失败: {str(e)}")
    
    def train_model(self, algorithm: str, data_dict: Dict[str, Any], 
                   custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        训练回归模型
        """
        try:
            if algorithm not in self.algorithm_configs:
                raise ValueError(f"不支持的回归算法: {algorithm}")
            
            config = self.algorithm_configs[algorithm]
            params = config['params'].copy()
            if custom_params:
                params.update(custom_params)
            
            # 创建模型
            model = config['class'](**params)
            
            # 训练模型
            X_train = data_dict['X_train']
            y_train = data_dict['y_train']
            model.fit(X_train, y_train)
            
            # 生成模型ID
            model_id = f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 保存模型信息
            self.models[model_id] = model
            self.scalers[model_id] = data_dict['scaler']
            self.feature_columns[model_id] = data_dict['feature_columns']
            self.target_column[model_id] = data_dict['target_column']
            
            # 预测训练集
            train_predictions = model.predict(X_train)
            
            # 评估
            train_mse = mean_squared_error(y_train, train_predictions)
            train_r2 = r2_score(y_train, train_predictions)
            
            return {
                'model_id': model_id,
                'algorithm': algorithm,
                'model': model,
                'training_predictions': train_predictions,
                'evaluation': {
                    'train_mse': train_mse,
                    'train_r2': train_r2
                },
                'parameters': params
            }
            
        except Exception as e:
            raise Exception(f"模型训练失败: {str(e)}")
    
    def evaluate_model(self, model_id: str, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估回归模型
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"模型 {model_id} 不存在")
            
            model = self.models[model_id]
            X_test = data_dict['X_test']
            y_test = data_dict['y_test']
            
            # 预测
            predictions = model.predict(X_test)
            
            # 计算指标
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            mae = np.mean(np.abs(y_test - predictions))
            
            return {
                'model_id': model_id,
                'predictions': predictions,
                'metrics': {
                    'mse': mse,
                    'rmse': rmse,
                    'r2_score': r2,
                    'mae': mae
                },
                'actual_values': y_test.tolist(),
                'predicted_values': predictions.tolist()
            }
            
        except Exception as e:
            raise Exception(f"模型评估失败: {str(e)}")
    
    def predict(self, model_id: str, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        使用回归模型进行预测
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"模型 {model_id} 不存在")
            
            model = self.models[model_id]
            scaler = self.scalers[model_id]
            feature_columns = self.feature_columns[model_id]
            
            # 检查特征列
            missing_features = [col for col in feature_columns if col not in new_data.columns]
            if missing_features:
                raise ValueError(f"缺少特征列: {missing_features}")
            
            # 准备数据
            X_new = new_data[feature_columns].copy()
            X_new = X_new.fillna(X_new.mean())
            
            # 标准化
            X_new_scaled = scaler.transform(X_new)
            X_new_scaled = pd.DataFrame(X_new_scaled, columns=feature_columns, index=X_new.index)
            
            # 预测
            predictions = model.predict(X_new_scaled)
            
            return {
                'model_id': model_id,
                'predictions': predictions.tolist(),
                'input_data': new_data.to_dict('records'),
                'feature_columns': feature_columns
            }
            
        except Exception as e:
            raise Exception(f"预测失败: {str(e)}")
    
    def get_supported_algorithms(self) -> List[str]:
        """
        获取支持的回归算法列表
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
            'default_params': config['params'],
            'preprocessing': config.get('preprocessing', 'standard')
        }