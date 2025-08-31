#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型预测工具
使用已训练的模型对新数据进行预测

使用方法:
  python predict.py --model model_file.joblib --data new_data.csv --output predictions.csv
  python predict.py --model model_file.joblib --interactive
"""

import argparse
import pandas as pd
import os
import sys
from ml_engine.algorithms.ml_algorithms import MLAlgorithms

def load_and_predict(model_path: str, data_path: str, output_path: str = None):
    """
    加载模型并对数据进行预测
    """
    try:
        # 初始化ML引擎
        ml_engine = MLAlgorithms()
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            return False
        
        # 加载模型
        print(f"正在加载模型: {model_path}")
        model_id = ml_engine.load_model(model_path)
        print(f"模型加载成功: {model_id}")
        
        # 获取模型信息
        model_info = ml_engine.get_model_info(model_id)
        print(f"模型类型: {model_info['model_type']}")
        print(f"特征列: {model_info['feature_columns']}")
        
        # 加载数据
        print(f"\n正在加载数据: {data_path}")
        if data_path.endswith('.csv'):
            new_data = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx'):
            new_data = pd.read_excel(data_path)
        else:
            print("错误: 不支持的文件格式，请使用 CSV 或 Excel 文件")
            return False
        
        print(f"数据形状: {new_data.shape}")
        print("数据预览:")
        print(new_data.head())
        
        # 检查特征列是否存在
        required_features = model_info['feature_columns']
        missing_features = [col for col in required_features if col not in new_data.columns]
        if missing_features:
            print(f"\n错误: 数据中缺少以下特征列: {missing_features}")
            print(f"需要的特征列: {required_features}")
            print(f"数据中的列: {list(new_data.columns)}")
            return False
        
        # 进行预测
        print("\n正在进行预测...")
        prediction_result = ml_engine.predict(model_id, new_data)
        predictions = prediction_result['predictions']
        
        # 创建结果DataFrame
        result_df = new_data.copy()
        result_df['prediction'] = predictions
        
        # 如果是分类模型且有概率预测
        if 'prediction_proba' in prediction_result:
            proba = prediction_result['prediction_proba']
            if proba is not None:
                # 添加概率列
                if len(proba.shape) == 2 and proba.shape[1] > 1:
                    for i in range(proba.shape[1]):
                        result_df[f'probability_class_{i}'] = proba[:, i]
                else:
                    result_df['probability'] = proba
        
        # 显示预测结果
        print("\n预测结果:")
        print(result_df)
        
        # 保存结果
        if output_path:
            if output_path.endswith('.csv'):
                result_df.to_csv(output_path, index=False)
            elif output_path.endswith('.xlsx'):
                result_df.to_excel(output_path, index=False)
            else:
                # 默认保存为CSV
                output_path += '.csv'
                result_df.to_csv(output_path, index=False)
            
            print(f"\n预测结果已保存到: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def interactive_predict(model_path: str):
    """
    交互式预测模式
    """
    try:
        # 初始化ML引擎
        ml_engine = MLAlgorithms()
        
        # 加载模型
        print(f"正在加载模型: {model_path}")
        model_id = ml_engine.load_model(model_path)
        print(f"模型加载成功: {model_id}")
        
        # 获取模型信息
        model_info = ml_engine.get_model_info(model_id)
        print(f"\n模型类型: {model_info['model_type']}")
        print(f"特征列: {model_info['feature_columns']}")
        
        required_features = model_info['feature_columns']
        
        print("\n=== 交互式预测模式 ===")
        print("输入 'quit' 退出程序")
        
        while True:
            print("\n请输入特征值:")
            
            # 收集特征值
            feature_values = {}
            for feature in required_features:
                while True:
                    try:
                        value = input(f"{feature}: ")
                        if value.lower() == 'quit':
                            print("退出程序")
                            return
                        
                        # 尝试转换为数值
                        try:
                            feature_values[feature] = float(value)
                            break
                        except ValueError:
                            # 如果不能转换为数值，保持字符串
                            feature_values[feature] = value
                            break
                    except KeyboardInterrupt:
                        print("\n退出程序")
                        return
            
            # 创建DataFrame
            new_data = pd.DataFrame([feature_values])
            
            # 进行预测
            try:
                prediction_result = ml_engine.predict(model_id, new_data)
                prediction = prediction_result['predictions'][0]
                
                print(f"\n预测结果: {prediction}")
                
                # 如果有概率预测
                if 'prediction_proba' in prediction_result:
                    proba = prediction_result['prediction_proba']
                    if proba is not None:
                        print(f"预测概率: {proba[0]}")
                
            except Exception as e:
                print(f"预测失败: {e}")
    
    except Exception as e:
        print(f"交互式预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description="使用已训练的模型进行预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 对CSV文件进行预测
  python predict.py --model models/my_model.joblib --data new_data.csv --output predictions.csv
  
  # 交互式预测
  python predict.py --model models/my_model.joblib --interactive
  
  # 预测并显示结果（不保存）
  python predict.py --model models/my_model.joblib --data new_data.csv
        """
    )
    
    parser.add_argument('--model', '-m', required=True,
                       help='模型文件路径 (.joblib 格式)')
    
    parser.add_argument('--data', '-d',
                       help='要预测的数据文件路径 (CSV 或 Excel 格式)')
    
    parser.add_argument('--output', '-o',
                       help='预测结果输出文件路径')
    
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='启用交互式预测模式')
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.interactive and not args.data:
        print("错误: 必须指定数据文件 (--data) 或启用交互式模式 (--interactive)")
        parser.print_help()
        sys.exit(1)
    
    if args.interactive and args.data:
        print("错误: 不能同时使用交互式模式和数据文件")
        parser.print_help()
        sys.exit(1)
    
    # 执行预测
    if args.interactive:
        interactive_predict(args.model)
    else:
        success = load_and_predict(args.model, args.data, args.output)
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()