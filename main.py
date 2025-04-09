#!/usr/bin/env python3
"""
Neural Network Implementation and Analysis
Technical Assessment

This script runs a complete analysis of linear regression and perceptron models
on 2D and multidimensional datasets, comparing different activation functions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

def create_output_dir():
    """Create output directory for results if it doesn't exist"""
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Created 'results' directory")
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory")

def save_dataset(X, y, output_file):
    """Save dataset to file"""
    np.savez(output_file, X=X, y=y)
    print(f"Dataset saved to {output_file}")
    return output_file

def run_data_generation(dimensions=None, output_dir='data'):
    """Generate datasets and visualize them"""
    print("\n1. Data Generation and Visualization...")
    from data_generator import generate_2d_data, generate_multi_dimensional_data
    from data_generator import visualize_2d_data, visualize_3d_data

    # Determine which dimensions to generate
    if dimensions is None:
        dimensions = [2, 3, 4, 5, 10]
    elif isinstance(dimensions, int):
        dimensions = [dimensions]

    datasets = {}

    for dim in dimensions:
        if dim == 2:
            # Generate and visualize 2D data
            X, y = generate_2d_data(n_samples=300, n_clusters=2)
            file_path = os.path.join(output_dir, f"{dim}d_data.npz")
            save_dataset(X, y, file_path)
            
            # Only visualize 2D data if requested
            plt_2d = visualize_2d_data(X, y, f"{dim}D Dataset with 2 Clusters")
            plt_2d.savefig(os.path.join(output_dir, f"{dim}d_data.png"))
            datasets[dim] = (X, y)
        else:
            # Generate multi-dimensional data
            X, y = generate_multi_dimensional_data(n_samples=300, n_clusters=2, n_features=dim)
            file_path = os.path.join(output_dir, f"{dim}d_data.npz")
            save_dataset(X, y, file_path)
            
            # Visualize 3D data if possible
            if dim == 3:
                plt_3d = visualize_3d_data(X, y, f"{dim}D Dataset with 2 Clusters")
                plt_3d.savefig(os.path.join(output_dir, f"{dim}d_data.png"))
            
            datasets[dim] = (X, y)
            print(f"{dim}D dataset shape: {X.shape}, Labels: {np.unique(y)}")

    print("  ✓ Data generation and visualization completed")
    return datasets

def run_linear_regression_analysis(datasets, output_dir='results'):
    """Run linear regression analysis using LSM"""
    print("\n2. Linear Regression Separator Analysis...")
    from linear_regression import least_squares_method
    from linear_regression import visualize_linear_separator_2d, visualize_linear_separator_3d
    from linear_regression import visualize_multidim_projection

    lsm_weights = {}

    for dim, (X, y) in datasets.items():
        # Convert labels to binary (0, 1) for regression
        y_binary = (y == 1).astype(int)

        # Apply least squares method
        weights = least_squares_method(X, y_binary)
        lsm_weights[dim] = weights
        
        # Visualize based on dimension
        if dim == 2:
            plt_2d = visualize_linear_separator_2d(X, y, weights, f"{dim}D Linear Regression Separator")
            plt_2d.savefig(os.path.join(output_dir, f"{dim}d_linear_separator.png"))
        elif dim == 3:
            plt_3d = visualize_linear_separator_3d(X, y, weights, f"{dim}D Linear Regression Separator")
            plt_3d.savefig(os.path.join(output_dir, f"{dim}d_linear_separator.png"))
        else:
            plt_md = visualize_multidim_projection(X, y, weights, f"{dim}D Linear Regression Projection")
            plt_md.savefig(os.path.join(output_dir, f"{dim}d_linear_separator.png"))
        
        print(f"  {dim}D LSM Weights: {weights}")

    print("  ✓ Linear regression analysis completed")
    return lsm_weights

def run_perceptron_analysis(dimensions=None, activation_functions=None, output_dir='results'):
    """Run perceptron analysis with different activation functions"""
    print("\n3. Perceptron Analysis...")
    from perceptron_analysis import analyze_perceptron_2d, analyze_perceptron_multidimensional
    from perceptron_analysis import weight_vector_analysis
    
    results = {}
    
    # Default activation functions if not provided
    if activation_functions is None:
        activation_functions = ['step', 'sigmoid', 'relu', 'tanh']
    
    # Analyze 2D case separately
    if dimensions is None or 2 in dimensions:
        print("  Running 2D perceptron analysis...")
        results_2d, df_2d, _ = analyze_perceptron_2d(
            output_file=os.path.join(output_dir, "2d_perceptron_comparison.png")
        )
        
        # Analyze weight vectors
        plt_weights = weight_vector_analysis(
            results_2d, 
            output_file=os.path.join(output_dir, "weight_vector_analysis.png")
        )
        
        # Save detailed results to CSV
        df_2d.to_csv(os.path.join(output_dir, "2d_perceptron_results.csv"), index=False)
        results[2] = results_2d
    
    # Analyze multidimensional cases
    md_dimensions = [d for d in dimensions if d > 2] if dimensions else [3, 5, 10]
    
    if md_dimensions:
        print(f"  Running multidimensional analysis for dimensions: {md_dimensions}...")
        results_md, df_md, plt_md = analyze_perceptron_multidimensional(
            dimensions=md_dimensions,
            activation_functions=activation_functions,
            output_file=os.path.join(output_dir, "multidimensional_perceptron_comparison.png")
        )
        
        # Save detailed results to CSV
        df_md.to_csv(os.path.join(output_dir, "multidimensional_perceptron_results.csv"), index=False)
        
        # Add results to the overall results dictionary
        for dim in md_dimensions:
            if dim in results_md:
                results[dim] = results_md[dim]
    
    print("  ✓ Perceptron analysis completed")
    return results

def generate_report():
    """Generate HTML report"""
    print("\n4. Generating HTML Report...")
    
    # Import the report generation script
    try:
        from generate_report import generate_html_report
        generate_html_report()
        print("  ✓ HTML report generated")
    except ImportError:
        print("  ⨯ generate_report.py not found. Skipping report generation.")

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Neural Network Analysis')
    parser.add_argument('--dimensions', type=int, nargs='+', 
                        help='Dimensions to analyze (e.g., 2 3 4 5 10)')
    parser.add_argument('--activations', type=str, nargs='+',
                        choices=['step', 'sigmoid', 'relu', 'tanh'],
                        help='Activation functions to analyze')
    parser.add_argument('--data-only', action='store_true',
                        help='Generate only datasets without analysis')
    parser.add_argument('--lsm-only', action='store_true',
                        help='Run only linear regression analysis')
    parser.add_argument('--perceptron-only', action='store_true',
                        help='Run only perceptron analysis')
    parser.add_argument('--report', action='store_true',
                        help='Generate HTML report')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory to save or load datasets')
    return parser.parse_args()

def run_complete_analysis(args=None):
    """Run the complete analysis pipeline based on command-line arguments"""
    if args is None:
        args = parse_args()
    
    start_time = time.time()
    
    # Create output directories
    create_output_dir()
    
    # Determine what to run
    run_data = not (args.lsm_only or args.perceptron_only or args.report)
    run_lsm = not (args.data_only or args.perceptron_only or args.report)
    run_perceptron = not (args.data_only or args.lsm_only or args.report)
    run_report = args.report
    
    # If no specific task is selected, run everything
    if not any([run_data, run_lsm, run_perceptron, run_report]):
        run_data = run_lsm = run_perceptron = True
        run_report = True
    
    # Step 1: Data Generation
    if run_data:
        datasets = run_data_generation(args.dimensions, args.data_dir)
    else:
        # Load existing datasets if not generating new ones
        dimensions = args.dimensions if args.dimensions else [2, 3, 4, 5, 10]
        datasets = {}
        for dim in dimensions:
            data_file = os.path.join(args.data_dir, f"{dim}d_data.npz")
            if os.path.exists(data_file):
                data = np.load(data_file)
                datasets[dim] = (data['X'], data['y'])
                print(f"Loaded {dim}D dataset from {data_file}")
            else:
                print(f"Warning: {data_file} not found, generating new dataset for {dim}D")
                from data_generator import generate_2d_data, generate_multi_dimensional_data
                if dim == 2:
                    X, y = generate_2d_data(n_samples=300, n_clusters=2)
                else:
                    X, y = generate_multi_dimensional_data(n_samples=300, n_clusters=2, n_features=dim)
                datasets[dim] = (X, y)
                save_dataset(X, y, data_file)
    
    # Step 2: Linear Regression Analysis
    if run_lsm:
        lsm_weights = run_linear_regression_analysis(datasets, args.output_dir)
    
    # Step 3: Perceptron Analysis
    if run_perceptron:
        perceptron_results = run_perceptron_analysis(
            args.dimensions, args.activations, args.output_dir
        )
    
    # Step 4: Generate Report
    if run_report:
        generate_report()
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\nAnalysis completed in {total_time:.2f} seconds")
    print(f"Results saved to the '{args.output_dir}' directory")

if __name__ == "__main__":
    run_complete_analysis() 