#!/usr/bin/env python3
"""
MLSelect - Machine Learning Algorithm Selection Tool

Usage:
    python mlselect <input_file> <output_file> <-xloc|xname> <N> <x1> ... <xN> <-yloc|-yname> <N> <y1> ... <yN>

Arguments:
    input_file: Input data file for analysis
    output_file: Output JSON file containing analysis results
    -xloc/-xname: Specify X columns by location or name
    -yloc/-yname: Specify Y columns by location or name
    N: Number of columns
    x1...xN: X column specifications
    y1...yN: Y column specifications
"""

import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import os

# Import our ML engine modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_engine.feature_analysis.feature_analyzer import FeatureAnalyzer
from ml_engine.model_selection.algorithm_selector import AlgorithmSelector
from ml_engine.algorithms.ml_algorithms import MLAlgorithms


class MLSelectCLI:
    def __init__(self):
        self.input_file = None
        self.output_file = None
        self.x_columns = []
        self.y_columns = []
        self.x_mode = None  # 'loc' or 'name'
        self.y_mode = None  # 'loc' or 'name'
        self.run_training = False  # Whether to run training
        self.training_mode = None  # 'top' or 'bottom'
        self.training_count = 0  # Number of algorithms to train
        
    def parse_arguments(self, args):
        """Parse command line arguments"""
        if len(args) < 6:
            self.print_usage()
            return False
            
        self.input_file = args[0]
        self.output_file = args[1]
        
        # Parse X columns
        i = 2
        if args[i] == '-xloc':
            self.x_mode = 'loc'
        elif args[i] == '-xname':
            self.x_mode = 'name'
        else:
            print(f"Error: Expected -xloc or -xname, got {args[i]}")
            return False
            
        i += 1
        if i >= len(args):
            print("Error: Missing X column specification")
            return False
            
        # Support comma-separated column names or single column name
        if ',' in args[i]:
            self.x_columns = [col.strip() for col in args[i].split(',')]
            i += 1
        else:
            # Check if it's a number (original format) or a single column name
            try:
                x_count = int(args[i])
                # Original format: number followed by column names
                i += 1
                if i + x_count > len(args):
                    print("Error: Not enough X column specifications")
                    return False
                    
                self.x_columns = args[i:i+x_count]
                i += x_count
            except ValueError:
                # Single column name
                self.x_columns = [args[i]]
                i += 1
        
        # Parse Y columns
        if i >= len(args):
            print("Error: Missing Y column specification")
            return False
            
        if args[i] == '-yloc':
            self.y_mode = 'loc'
        elif args[i] == '-yname':
            self.y_mode = 'name'
        else:
            print(f"Error: Expected -yloc or -yname, got {args[i]}")
            return False
            
        i += 1
        if i >= len(args):
            print("Error: Missing Y column specification")
            return False
            
        # Support comma-separated column names or single column name
        if ',' in args[i]:
            self.y_columns = [col.strip() for col in args[i].split(',')]
            i += 1
        else:
            # Check if it's a number (original format) or a single column name
            try:
                y_count = int(args[i])
                # Original format: number followed by column names
                i += 1
                if i + y_count > len(args):
                    print("Error: Not enough Y column specifications")
                    return False
                    
                self.y_columns = args[i:i+y_count]
                i += y_count
            except ValueError:
                # Single column name
                self.y_columns = [args[i]]
                i += 1
        
        # Parse optional -run parameter
        if i < len(args) and args[i] == '-run':
            self.run_training = True
            i += 1
            
            if i >= len(args):
                print("Error: Missing training mode after -run")
                return False
                
            if args[i] not in ['top', 'bottom']:
                print(f"Error: Training mode must be 'top' or 'bottom', got {args[i]}")
                return False
                
            self.training_mode = args[i]
            i += 1
            
            if i >= len(args):
                print("Error: Missing training count after training mode")
                return False
                
            try:
                self.training_count = int(args[i])
                if self.training_count <= 0:
                    print("Error: Training count must be positive")
                    return False
            except ValueError:
                print(f"Error: Training count must be a number, got {args[i]}")
                return False
        
        return True
    
    def print_usage(self):
        """Print usage information"""
        print("Usage: python mlselect <input_file> <output_file> <-xloc|-xname> <N> <x1> ... <xN> <-yloc|-yname> <N> <y1> ... <yN> [-run] [top|bottom] N")
        print("")
        print("Arguments:")
        print("  input_file    Input data file for analysis")
        print("  output_file   Output JSON file containing analysis results")
        print("  -xloc/-xname  Specify X columns by location (0-based index) or name")
        print("  -yloc/-yname  Specify Y columns by location (0-based index) or name")
        print("  N             Number of columns")
        print("  x1...xN       X column specifications")
        print("  y1...yN       Y column specifications")
        print("  -run          Optional: Train models using selected algorithms")
        print("  top|bottom    Select top N or bottom N algorithms by score")
        print("  N             Number of algorithms to train")
        print("")
        print("Examples:")
        print("  python mlselect data.csv output.json -xname 2 feature1 feature2 -yname 1 target")
        print("  python mlselect data.csv output.json -xloc 2 0 1 -yloc 1 2")
        print("  python mlselect data.csv output.json -xname 2 feature1 feature2 -yname 1 target -run top 3")
        print("  python mlselect data.csv output.json -xloc 2 0 1 -yloc 1 2 -run bottom 2")
    
    def load_data(self):
        """Load input data file"""
        try:
            file_ext = Path(self.input_file).suffix.lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(self.input_file)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(self.input_file)
            elif file_ext == '.json':
                df = pd.read_json(self.input_file)
            else:
                print(f"Error: Unsupported file format {file_ext}")
                return None
                
            return df
        except Exception as e:
            print(f"Error loading file {self.input_file}: {str(e)}")
            return None
    
    def select_columns(self, df):
        """Select X and Y columns from dataframe"""
        try:
            # Select X columns
            if self.x_mode == 'loc':
                x_indices = [int(col) for col in self.x_columns]
                X = df.iloc[:, x_indices]
                x_column_names = df.columns[x_indices].tolist()
            else:  # name mode
                X = df[self.x_columns]
                x_column_names = self.x_columns
            
            # Select Y columns
            if self.y_mode == 'loc':
                y_indices = [int(col) for col in self.y_columns]
                y = df.iloc[:, y_indices]
                y_column_names = df.columns[y_indices].tolist()
            else:  # name mode
                y = df[self.y_columns]
                y_column_names = self.y_columns
            
            return X, y, x_column_names, y_column_names
            
        except Exception as e:
            print(f"Error selecting columns: {str(e)}")
            return None, None, None, None
    
    def analyze_data(self, df, X, y, x_column_names, y_column_names):
        """Analyze data and generate results"""
        # Basic file information
        file_info = {
            'filename': os.path.basename(self.input_file),
            'analysis_date': datetime.now().isoformat(),
            'x_columns_count': len(x_column_names),
            'x_columns_names': x_column_names,
            'x_rows_count': len(X),
            'y_columns_count': len(y_column_names),
            'y_columns_names': y_column_names,
            'y_rows_count': len(y),
            'total_rows': len(df),
            'total_columns': len(df.columns)
        }
        
        # DataFrame info
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info = buffer.getvalue()
        
        # Data quality analysis
        data_quality = {
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'duplicated_rows': int(df.duplicated().sum()),
            'memory_usage': df.memory_usage(deep=True).to_dict()
        }
        
        # Feature analysis
        try:
            feature_analyzer = FeatureAnalyzer()
            feature_analysis = feature_analyzer.analyze_features(X, y)
        except Exception as e:
            print(f"Warning: Feature analysis failed: {str(e)}")
            feature_analysis = {"error": str(e)}
        
        # Algorithm recommendation
        try:
            algorithm_selector = AlgorithmSelector()
            recommended_algorithms = algorithm_selector.recommend_algorithms(X, y)
        except Exception as e:
            print(f"Warning: Algorithm recommendation failed: {str(e)}")
            recommended_algorithms = {"error": str(e), "default_recommendations": [
                "Linear Regression", "Random Forest", "Support Vector Machine"
            ]}
        
        # Print console output
        self.print_console_output(file_info, data_quality, recommended_algorithms)
        
        # Train models if requested
        training_results = None
        if self.run_training:
            training_results = self.train_models(df, X, y, x_column_names, y_column_names, recommended_algorithms)
        
        results = {
            'file_info': file_info,
            'dataframe_info': df_info,
            'data_quality': data_quality,
            'feature_analysis': feature_analysis,
            'recommended_algorithms': recommended_algorithms
        }
        
        if training_results:
            results['training_results'] = training_results
            
        return results
    
    def train_models(self, df, X, y, x_column_names, y_column_names, recommended_algorithms):
        """Train models using selected algorithms"""
        print("\n" + "="*60)
        print("模型训练")
        print("="*60)
        
        try:
            # Get algorithm list from recommendations
            if isinstance(recommended_algorithms, list):
                algorithms = recommended_algorithms
            elif isinstance(recommended_algorithms, dict) and 'algorithms' in recommended_algorithms:
                algorithms = recommended_algorithms['algorithms']
            else:
                print("Error: Cannot extract algorithms from recommendations")
                return None
            
            # Sort algorithms by score
            if all(isinstance(alg, dict) and 'score' in alg for alg in algorithms):
                sorted_algorithms = sorted(algorithms, key=lambda x: x['score'], reverse=True)
            else:
                print("Warning: Algorithms don't have scores, using original order")
                sorted_algorithms = algorithms
            
            # Select algorithms based on training mode
            if self.training_mode == 'top':
                selected_algorithms = sorted_algorithms[:self.training_count]
                print(f"训练评分最高的 {self.training_count} 种算法:")
            else:  # bottom
                selected_algorithms = sorted_algorithms[-self.training_count:]
                print(f"训练评分最低的 {self.training_count} 种算法:")
            
            # Initialize ML algorithms engine
            ml_engine = MLAlgorithms()
            
            # Prepare target column name
            target_column = y_column_names[0] if y_column_names else 'target'
            
            # Create a combined dataframe for training
            train_df = pd.concat([X, y], axis=1)
            
            training_results = []
            
            for i, alg in enumerate(selected_algorithms, 1):
                alg_name = alg.get('algorithm', alg.get('name', f'algorithm_{i}'))
                alg_score = alg.get('score', 'N/A')
                
                print(f"\n{i}. 训练 {alg_name} (评分: {alg_score})")
                
                try:
                    # Prepare data for this algorithm
                    data_dict = ml_engine.prepare_data(
                        train_df, 
                        target_column, 
                        x_column_names,
                        algorithm=alg_name.lower().replace(' ', '_')
                    )
                    
                    # Train the model
                    train_result = ml_engine.train_model(
                        alg_name.lower().replace(' ', '_'),
                        data_dict
                    )
                    
                    if train_result and 'model_id' in train_result:
                        model_id = train_result['model_id']
                        evaluation = train_result.get('evaluation', {})
                        
                        print(f"   ✓ 训练成功 - 模型ID: {model_id}")
                        
                        # Display evaluation metrics
                        if evaluation and 'metrics' in evaluation:
                            metrics = evaluation['metrics']
                            if 'r2_score' in metrics:
                                print(f"   R² Score: {metrics['r2_score']:.4f}")
                            elif 'accuracy' in metrics:
                                print(f"   Accuracy: {metrics['accuracy']:.4f}")
                            if 'mse' in metrics:
                                print(f"   MSE: {metrics['mse']:.4f}")
                        
                        training_results.append({
                            'algorithm': alg_name,
                            'model_id': model_id,
                            'training_success': True,
                            'evaluation': evaluation,
                            'parameters': train_result.get('parameters', {})
                        })
                    else:
                        error_msg = train_result.get('error', 'Training failed') if train_result else 'Training returned None'
                        print(f"   ✗ 训练失败: {error_msg}")
                        training_results.append({
                            'algorithm': alg_name,
                            'training_success': False,
                            'error': error_msg
                        })
                        
                except Exception as e:
                    print(f"   ✗ 训练失败: {str(e)}")
                    training_results.append({
                        'algorithm': alg_name,
                        'training_success': False,
                        'error': str(e)
                    })
            
            print(f"\n训练完成! 成功训练了 {sum(1 for r in training_results if r['training_success'])} 个模型")
            
            return {
                'selected_algorithms': selected_algorithms,
                'training_mode': self.training_mode,
                'training_count': self.training_count,
                'results': training_results
            }
            
        except Exception as e:
            print(f"训练过程中发生错误: {str(e)}")
            return {
                'error': str(e),
                'training_mode': self.training_mode,
                'training_count': self.training_count
            }
    
    def print_console_output(self, file_info, data_quality, recommended_algorithms):
        """Print simplified console output"""
        print("\n" + "="*60)
        print("数据结构分析")
        print("="*60)
        print(f"文件名: {file_info['filename']}")
        print(f"数据维度: {file_info['total_rows']} 行 × {file_info['total_columns']} 列")
        print(f"特征列: {', '.join(file_info['x_columns_names'])} ({file_info['x_columns_count']} 个)")
        print(f"目标列: {', '.join(file_info['y_columns_names'])} ({file_info['y_columns_count']} 个)")
        
        # Data quality summary
        missing_count = sum(data_quality['missing_values'].values())
        duplicate_count = data_quality['duplicated_rows']
        print(f"缺失值: {missing_count} 个")
        print(f"重复行: {duplicate_count} 行")
        
        print("\n" + "="*60)
        print("算法推荐 (按评分排序)")
        print("="*60)
        
        if isinstance(recommended_algorithms, dict) and 'error' in recommended_algorithms:
            print("算法推荐失败，使用默认推荐:")
            for i, alg in enumerate(recommended_algorithms.get('default_recommendations', []), 1):
                print(f"{i}. {alg}")
        elif isinstance(recommended_algorithms, list) and recommended_algorithms:
            # Sort algorithms by score (descending)
            sorted_algorithms = sorted(recommended_algorithms, key=lambda x: x.get('score', 0), reverse=True)
            for i, alg in enumerate(sorted_algorithms, 1):
                score = alg.get('score', 0)
                name = alg.get('name', 'Unknown')
                description = alg.get('description', '')
                suitability = alg.get('suitability', '')
                print(f"{i}. {name} (评分: {score:.2f}) - {suitability}")
                if description:
                    print(f"   描述: {description}")
                print()
        else:
            print("未找到算法推荐")
        
        print("="*60)
    
    def save_results(self, results):
        """Save results to JSON file"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n详细结果已保存到: {self.output_file}")
            return True
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return False
    
    def run(self, args):
        """Main execution function"""
        # Parse arguments
        if not self.parse_arguments(args):
            return False
        
        # Validate input file
        if not os.path.exists(self.input_file):
            print(f"Error: Input file {self.input_file} does not exist")
            return False
        
        # Load data
        print(f"Loading data from {self.input_file}...")
        df = self.load_data()
        if df is None:
            return False
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Select columns
        print("Selecting columns...")
        X, y, x_column_names, y_column_names = self.select_columns(df)
        if X is None or y is None:
            return False
        
        print(f"Selected {len(x_column_names)} X columns and {len(y_column_names)} Y columns")
        
        # Analyze data
        print("Analyzing data...")
        results = self.analyze_data(df, X, y, x_column_names, y_column_names)
        
        # Save results
        print("Saving results...")
        if self.save_results(results):
            print("Analysis completed successfully!")
            return True
        else:
            return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        cli = MLSelectCLI()
        cli.print_usage()
        sys.exit(1)
    
    cli = MLSelectCLI()
    success = cli.run(sys.argv[1:])
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()