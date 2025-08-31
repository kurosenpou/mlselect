import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ClassificationAlgorithms:
    """
    分类算法实现类 - 专门处理分类任务的机器学习算法
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
        
        # 分类算法配置
        self.algorithm_configs = {
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
            }
        }
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, feature_columns: List[str], 
                    test_size: float = 0.2) -> Dict[str, Any]:
        """
        准备分类数据
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
            for col in X.columns:
                if X[col].dtype in ['object', 'category']:
                    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
                else:
                    X[col] = X[col].fillna(X[col].mean())
            
            # 编码分类特征
            encoders = {}
            for col in X.columns:
                if X[col].dtype in ['object', 'category']:
                    encoder = LabelEncoder()
                    X[col] = encoder.fit_transform(X[col].astype(str))
                    encoders[col] = encoder
            
            # 编码目标变量
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y.astype(str))
            
            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'scaler': scaler,
                'encoders': encoders,
                'target_encoder': target_encoder,
                'feature_columns': feature_columns,
                'target_column': target_column,
                'original_X': X,
                'original_y': y,
                'class_names': target_encoder.classes_
            }
            
        except Exception as e:
            raise Exception(f"数据准备失败: {str(e)}")
    
    def train_model(self, algorithm: str, data_dict: Dict[str, Any], 
                   custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        训练分类模型
        """
        try:
            if algorithm not in self.algorithm_configs:
                raise ValueError(f"不支持的分类算法: {algorithm}")
            
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
            self.encoders[model_id] = data_dict['encoders']
            self.feature_columns[model_id] = data_dict['feature_columns']
            self.target_column[model_id] = data_dict['target_column']
            
            # 预测训练集
            train_predictions = model.predict(X_train)
            
            # 评估
            train_accuracy = accuracy_score(y_train, train_predictions)
            train_f1 = f1_score(y_train, train_predictions, average='weighted')
            
            return {
                'model_id': model_id,
                'algorithm': algorithm,
                'model': model,
                'training_predictions': train_predictions,
                'evaluation': {
                    'train_accuracy': train_accuracy,
                    'train_f1_score': train_f1
                },
                'parameters': params,
                'class_names': data_dict['class_names']
            }
            
        except Exception as e:
            raise Exception(f"模型训练失败: {str(e)}")
    
    def evaluate_model(self, model_id: str, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估分类模型
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"模型 {model_id} 不存在")
            
            model = self.models[model_id]
            X_test = data_dict['X_test']
            y_test = data_dict['y_test']
            
            # 预测
            predictions = model.predict(X_test)
            probabilities = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_test)
            
            # 计算指标
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            
            # 分类报告
            class_report = classification_report(y_test, predictions, 
                                               target_names=data_dict['class_names'],
                                               output_dict=True)
            
            # 混淆矩阵
            conf_matrix = confusion_matrix(y_test, predictions)
            
            return {
                'model_id': model_id,
                'predictions': predictions,
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'metrics': {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall
                },
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'actual_values': y_test.tolist(),
                'predicted_values': predictions.tolist(),
                'class_names': data_dict['class_names']
            }
            
        except Exception as e:
            raise Exception(f"模型评估失败: {str(e)}")
    
    def predict(self, model_id: str, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        使用分类模型进行预测
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
            
            # 处理缺失值
            for col in X_new.columns:
                if X_new[col].dtype in ['object', 'category']:
                    X_new[col] = X_new[col].fillna('Unknown')
                else:
                    X_new[col] = X_new[col].fillna(X_new[col].mean())
            
            # 编码分类特征
            for col in X_new.columns:
                if col in encoders:
                    # 处理未见过的类别
                    X_new[col] = X_new[col].astype(str)
                    known_classes = set(encoders[col].classes_)
                    X_new[col] = X_new[col].apply(lambda x: x if x in known_classes else encoders[col].classes_[0])
                    X_new[col] = encoders[col].transform(X_new[col])
            
            # 标准化
            X_new_scaled = scaler.transform(X_new)
            X_new_scaled = pd.DataFrame(X_new_scaled, columns=feature_columns, index=X_new.index)
            
            # 预测
            predictions = model.predict(X_new_scaled)
            probabilities = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_new_scaled)
            
            return {
                'model_id': model_id,
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'input_data': new_data.to_dict('records'),
                'feature_columns': feature_columns
            }
            
        except Exception as e:
            raise Exception(f"预测失败: {str(e)}")
    
    def get_supported_algorithms(self) -> List[str]:
        """
        获取支持的分类算法列表
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