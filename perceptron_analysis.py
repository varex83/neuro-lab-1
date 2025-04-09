import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
from matplotlib.gridspec import GridSpec
import argparse
import os
import sys

from data_generator import generate_2d_data, generate_multi_dimensional_data
from linear_regression import least_squares_method
from perceptron import Perceptron

def analyze_perceptron_2d(activation_functions=['step', 'sigmoid', 'relu', 'tanh'], 
                         data=None, output_file=None):
    """
    Analyze perceptron performance with different activation functions on 2D data
    
    Args:
        activation_functions: List of activation functions to test
        data: Tuple of (X, y) data, if None, will generate new data
        output_file: Path to save the visualization
    """
    # Generate or use provided 2D data
    if data is None:
        X, y = generate_2d_data(n_samples=200, n_clusters=2)
    else:
        X, y = data
    
    # Dictionary to store results
    results = {
        'activation': [],
        'accuracy': [],
        'iterations': [],
        'weights': [],
        'confusion_matrix': []
    }
    
    # First, apply least squares method for comparison
    y_binary = (y == 1).astype(int)
    lsm_weights = least_squares_method(X, y_binary)
    
    # Create a figure for all plots
    fig = plt.figure(figsize=(18, 12))
    
    # Create a grid layout that can accommodate all activation functions
    num_activations = len(activation_functions)
    gs = GridSpec(2, max(3, num_activations), figure=fig)
    
    # Plot the data and LSM separator
    ax_data = fig.add_subplot(gs[0, 0])
    unique_labels = np.unique(y)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        ax_data.scatter(
            X[y == label, 0],
            X[y == label, 1],
            color=colors[i],
            alpha=0.7,
            label=f'Class {label}'
        )
    
    # Draw the LSM decision boundary
    min_x1, max_x1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_values = np.array([min_x1, max_x1])
    x2_values = (-lsm_weights[0] - lsm_weights[1] * x1_values) / lsm_weights[2]
    
    ax_data.plot(x1_values, x2_values, 'k-', lw=2, label='LSM Separator')
    
    equation = f"LSM: {lsm_weights[0]:.2f} + {lsm_weights[1]:.2f}x₁ + {lsm_weights[2]:.2f}x₂ = 0"
    ax_data.text(0.05, 0.95, equation, transform=ax_data.transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    ax_data.set_title('2D Data with LSM Separator')
    ax_data.set_xlabel('Feature 1')
    ax_data.set_ylabel('Feature 2')
    ax_data.legend()
    ax_data.grid(True, alpha=0.3)
    
    # Plot for training curves
    ax_training = fig.add_subplot(gs[0, 1:])
    ax_training.set_title('Training Curves (Iterations vs. Errors)')
    ax_training.set_xlabel('Iterations')
    ax_training.set_ylabel('Number of Misclassifications')
    ax_training.grid(True, alpha=0.3)
    
    # Test each activation function
    for i, activation in enumerate(activation_functions):
        # Initialize and train perceptron
        perceptron = Perceptron(
            n_features=X.shape[1],
            learning_rate=0.01,
            max_iterations=1000,
            activation=activation
        )
        
        # Train the perceptron
        perceptron.fit(X, y, verbose=False)
        
        # Evaluate the perceptron
        accuracy, cm = perceptron.evaluate(X, y)
        
        # Store results
        results['activation'].append(activation)
        results['accuracy'].append(accuracy)
        results['iterations'].append(perceptron.iterations_performed)
        results['weights'].append(perceptron.weights)
        results['confusion_matrix'].append(cm)
        
        # Plot decision boundary
        ax_decision = fig.add_subplot(gs[1, i])
        
        # Create mesh grid for contour plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        
        # Make predictions on the mesh grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = perceptron.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        ax_decision.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
        ax_decision.contour(xx, yy, Z, colors='k', linewidths=0.5)
        
        # Plot the data points
        for j, label in enumerate(unique_labels):
            ax_decision.scatter(
                X[y == label, 0],
                X[y == label, 1],
                color=colors[j],
                alpha=0.7,
                s=20  # Smaller points for clearer visualization
            )
        
        # Draw the perceptron's linear boundary
        x1_values = np.array([x_min, x_max])
        x2_values = (-perceptron.weights[0] - perceptron.weights[1] * x1_values) / perceptron.weights[2]
        
        ax_decision.plot(x1_values, x2_values, 'r-', lw=2)
        
        # Add the equation as text
        equation = f"{perceptron.weights[0]:.2f} + {perceptron.weights[1]:.2f}x₁ + {perceptron.weights[2]:.2f}x₂ = 0"
        ax_decision.text(0.05, 0.95, equation, transform=ax_decision.transAxes, 
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        ax_decision.set_title(f'Activation: {activation} (Acc: {accuracy:.2f})')
        ax_decision.set_xlabel('Feature 1')
        ax_decision.set_ylabel('Feature 2')
        
        # Plot the training curve
        ax_training.plot(
            range(1, len(perceptron.errors) + 1),
            perceptron.errors,
            label=f'{activation} (Iter: {perceptron.iterations_performed})'
        )
    
    ax_training.legend()
    plt.tight_layout()
    
    # Save the visualization if output path is provided
    if output_file:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(output_file)
        print(f"2D perceptron comparison saved to {output_file}")
    
    # Create a DataFrame with the results
    df_results = pd.DataFrame({
        'Activation': results['activation'],
        'Accuracy': results['accuracy'],
        'Iterations': results['iterations']
    })
    
    print("2D Analysis Results:")
    print(df_results)
    
    return results, df_results, plt

def analyze_perceptron_multidimensional(dimensions=[3, 4, 5, 10], 
                                       activation_functions=['step', 'sigmoid', 'relu', 'tanh'],
                                       output_file=None):
    """
    Analyze perceptron performance with different activation functions and dimensions
    
    Args:
        dimensions: List of dimensions to test
        activation_functions: List of activation functions to test
        output_file: Path to save the visualization
    """
    # Initialize results dictionary
    results = {
        'dimensions': [],
        'activation': [],
        'accuracy': [],
        'iterations': [],
        'weights': []
    }
    
    # Test each dimension
    for dim in dimensions:
        print(f"\nAnalyzing {dim}-dimensional data:")
        
        # Generate data
        X, y = generate_multi_dimensional_data(n_samples=200, n_clusters=2, n_features=dim)
        
        # Test each activation function
        for activation in activation_functions:
            # Initialize and train perceptron
            perceptron = Perceptron(
                n_features=X.shape[1],
                learning_rate=0.01,
                max_iterations=1000,
                activation=activation
            )
            
            # Train the perceptron
            perceptron.fit(X, y, verbose=False)
            
            # Evaluate the perceptron
            accuracy, cm = perceptron.evaluate(X, y)
            
            # Store results
            results['dimensions'].append(dim)
            results['activation'].append(activation)
            results['accuracy'].append(accuracy)
            results['iterations'].append(perceptron.iterations_performed)
            results['weights'].append(perceptron.weights)
            
            print(f"  {activation} activation - Accuracy: {accuracy:.4f}, Iterations: {perceptron.iterations_performed}")
    
    # Create a DataFrame with the results
    df_results = pd.DataFrame({
        'Dimensions': results['dimensions'],
        'Activation': results['activation'],
        'Accuracy': results['accuracy'],
        'Iterations': results['iterations']
    })
    
    # Generate plot for comparison
    plt.figure(figsize=(12, 8))
    
    # Group by dimensions
    for dim in dimensions:
        # Get data for this dimension
        dim_data = df_results[df_results['Dimensions'] == dim]
        
        # Plot accuracy by activation function
        plt.subplot(2, 2, 1)
        plt.bar(
            dim_data['Activation'] + f" ({dim}D)",
            dim_data['Accuracy'],
            alpha=0.7,
            label=f"{dim}D"
        )
    
    plt.title('Accuracy by Activation Function and Dimension')
    plt.xlabel('Activation Function (Dimension)')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.05)
    plt.grid(True, alpha=0.3)
    
    # Group by activation function
    for activation in activation_functions:
        # Get data for this activation
        act_data = df_results[df_results['Activation'] == activation]
        
        # Sort by dimension
        act_data = act_data.sort_values('Dimensions')
        
        # Plot accuracy by dimension
        plt.subplot(2, 2, 2)
        plt.plot(
            act_data['Dimensions'],
            act_data['Accuracy'],
            'o-',
            label=activation
        )
        
        # Plot iterations by dimension
        plt.subplot(2, 2, 3)
        plt.plot(
            act_data['Dimensions'],
            act_data['Iterations'],
            'o-',
            label=activation
        )
    
    plt.subplot(2, 2, 2)
    plt.title('Accuracy vs Dimensions')
    plt.xlabel('Dimensions')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.title('Training Iterations vs Dimensions')
    plt.xlabel('Dimensions')
    plt.ylabel('Iterations')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add a summary table as text
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Create a summary table
    table_data = []
    table_columns = ['Activation', 'Avg Accuracy', 'Avg Iterations']
    
    for activation in activation_functions:
        act_data = df_results[df_results['Activation'] == activation]
        avg_acc = act_data['Accuracy'].mean()
        avg_iter = act_data['Iterations'].mean()
        table_data.append([activation, f"{avg_acc:.4f}", f"{avg_iter:.1f}"])
    
    plt.table(
        cellText=table_data,
        colLabels=table_columns,
        loc='center',
        cellLoc='center'
    )
    
    plt.tight_layout()
    
    # Save the visualization if output path is provided
    if output_file:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(output_file)
        print(f"Multidimensional comparison saved to {output_file}")
    
    # Save CSV results
    csv_output = output_file.replace('.png', '.csv') if output_file else "multidimensional_perceptron_results.csv"
    df_results.to_csv(csv_output, index=False)
    print(f"Multidimensional results saved to {csv_output}")
    
    return results, df_results, plt

def weight_vector_analysis(results_2d, output_file=None):
    """
    Analyze the weight vectors of different activation functions
    
    Args:
        results_2d: Results dictionary from analyze_perceptron_2d
        output_file: Path to save the visualization
    """
    plt.figure(figsize=(12, 6))
    
    # Plot the weight vectors
    activations = results_2d['activation']
    weights = results_2d['weights']
    
    # Bar chart for weight comparison
    bar_width = 0.2
    index = np.arange(3)  # 3 weights: bias, w1, w2
    
    for i, (activation, weight) in enumerate(zip(activations, weights)):
        plt.bar(
            index + i * bar_width,
            weight,
            bar_width,
            alpha=0.7,
            label=activation
        )
    
    plt.xlabel('Weight Component')
    plt.ylabel('Weight Value')
    plt.title('Weight Vector Comparison by Activation Function')
    plt.xticks(index + bar_width * (len(activations) - 1) / 2, ['Bias (w0)', 'w1', 'w2'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the visualization if output path is provided
    if output_file:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(output_file)
        print(f"Weight vector analysis saved to {output_file}")
    
    # Print weight interpretation
    print("\nWeight Vector Analysis:")
    print("=====================")
    
    for activation, weight in zip(activations, weights):
        print(f"\nActivation: {activation}")
        print(f"Weight vector: {weight}")
        print("Interpretation:")
        print(f"  Bias (w0) = {weight[0]:.4f} - The decision threshold")
        
        # Determine feature importance
        feature_importance = np.abs(weight[1:])
        most_important_idx = np.argmax(feature_importance) + 1
        
        print(f"  Feature {most_important_idx} has the largest magnitude ({np.abs(weight[most_important_idx]):.4f}), indicating it's the most influential in classification")
        
        # Check the sign of weights
        if weight[1] * weight[2] < 0:
            print("  Features have opposite signs, suggesting they have opposing effects on the classification")
        else:
            print("  Features have the same sign, suggesting they both push classification in the same direction")
            
        # Decision boundary equation
        print(f"  Decision boundary equation: {weight[0]:.4f} + {weight[1]:.4f}x₁ + {weight[2]:.4f}x₂ = 0")
    
    return plt

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Perceptron analysis with different activation functions and dimensions')
    parser.add_argument('--analysis', type=str, default='2d', 
                        choices=['2d', 'multidim', 'weight-analysis', 'activation-comparison'],
                        help='Type of analysis to perform')
    parser.add_argument('--dimensions', type=str, default='3,4,5,10', 
                        help='Comma-separated list of dimensions to analyze (for multidim analysis)')
    parser.add_argument('--activations', type=str, default='step,sigmoid,relu,tanh',
                        help='Comma-separated list of activation functions to test')
    parser.add_argument('--input', type=str, help='Input data file (.npz)')
    parser.add_argument('--output', type=str, help='Output visualization file (.png)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Parse dimensions and activations
    dimensions = [int(dim) for dim in args.dimensions.split(',')]
    activation_functions = args.activations.split(',')
    
    # Load data if input file is provided
    data = None
    if args.input and os.path.exists(args.input):
        data_loaded = np.load(args.input)
        data = (data_loaded['X'], data_loaded['y'])
    
    # Perform the requested analysis
    if args.analysis == '2d':
        results_2d, df_2d, _ = analyze_perceptron_2d(
            activation_functions=activation_functions,
            data=data,
            output_file=args.output
        )
        
        # Save CSV results
        csv_output = args.output.replace('.png', '.csv') if args.output else "2d_perceptron_results.csv"
        df_2d.to_csv(csv_output, index=False)
        
        # Also perform weight vector analysis
        weight_output = args.output.replace('.png', '_weights.png') if args.output else "weight_vector_analysis.png"
        weight_vector_analysis(results_2d, weight_output)
        
    elif args.analysis == 'multidim':
        results_md, df_md, _ = analyze_perceptron_multidimensional(
            dimensions=dimensions,
            activation_functions=activation_functions,
            output_file=args.output
        )
        
    elif args.analysis == 'weight-analysis':
        # First, run 2D analysis
        results_2d, _, _ = analyze_perceptron_2d(
            activation_functions=activation_functions,
            data=data
        )
        
        # Then analyze weights
        weight_vector_analysis(results_2d, args.output)
        
    elif args.analysis == 'activation-comparison':
        # Compare activation functions across dimensions
        results_md, df_md, _ = analyze_perceptron_multidimensional(
            dimensions=dimensions,
            activation_functions=activation_functions,
            output_file=args.output
        )
    
    print("\nAnalysis completed and saved to files.") 