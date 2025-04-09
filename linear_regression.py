import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import argparse
import os
import sys

def add_bias_term(X):
    """
    Add a bias term (column of ones) to the input features
    
    Args:
        X: Input features of shape (n_samples, n_features)
        
    Returns:
        X_with_bias: Features with bias term of shape (n_samples, n_features+1)
    """
    return np.hstack([np.ones((X.shape[0], 1)), X])

def least_squares_method(X, y):
    """
    Implement the Least Squares Method for linear regression
    
    Args:
        X: Input features of shape (n_samples, n_features)
        y: Target labels of shape (n_samples,)
        
    Returns:
        weights: Weight vector including the bias term [w0, w1, w2, ...]
    """
    # Add bias term to X
    X_with_bias = add_bias_term(X)
    
    # Calculate weights using the normal equation: w = (X^T X)^(-1) X^T y
    # Using pseudoinverse to handle potentially singular matrices
    weights = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    
    return weights

def visualize_linear_separator_2d(X, y, weights, title="Linear Regression Separator"):
    """
    Visualize the linear separator in 2D space
    
    Args:
        X: Input features of shape (n_samples, 2)
        y: Target labels of shape (n_samples,)
        weights: Weight vector [w0, w1, w2]
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Plot each cluster with a different color
    unique_labels = np.unique(y)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        plt.scatter(
            X[y == label, 0],
            X[y == label, 1],
            color=colors[i],
            alpha=0.7,
            label=f'Cluster {label}'
        )
    
    # Draw the linear decision boundary
    # For a 2D feature space, the equation is: w0 + w1*x1 + w2*x2 = 0
    # Solving for x2: x2 = (-w0 - w1*x1) / w2
    min_x1, max_x1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_values = np.array([min_x1, max_x1])
    x2_values = (-weights[0] - weights[1] * x1_values) / weights[2]
    
    plt.plot(x1_values, x2_values, 'r-', lw=2, label='Linear Separator')
    
    # Add the equation as text
    equation = f"Equation: {weights[0]:.2f} + {weights[1]:.2f}x₁ + {weights[2]:.2f}x₂ = 0"
    plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def visualize_linear_separator_3d(X, y, weights, title="Linear Regression Hyperplane"):
    """
    Visualize the linear separator (hyperplane) in 3D space
    
    Args:
        X: Input features of shape (n_samples, 3)
        y: Target labels of shape (n_samples,)
        weights: Weight vector [w0, w1, w2, w3]
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each cluster with a different color
    unique_labels = np.unique(y)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        ax.scatter(
            X[y == label, 0],
            X[y == label, 1],
            X[y == label, 2],
            color=colors[i],
            alpha=0.7,
            label=f'Cluster {label}'
        )
    
    # Create a meshgrid to plot the hyperplane
    min_x1, max_x1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    min_x2, max_x2 = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    x1_grid, x2_grid = np.meshgrid(
        np.linspace(min_x1, max_x1, 20),
        np.linspace(min_x2, max_x2, 20)
    )
    
    # Calculate x3 from the hyperplane equation: w0 + w1*x1 + w2*x2 + w3*x3 = 0
    # Solving for x3: x3 = (-w0 - w1*x1 - w2*x2) / w3
    x3_grid = (-weights[0] - weights[1] * x1_grid - weights[2] * x2_grid) / weights[3]
    
    # Plot the hyperplane
    surf = ax.plot_surface(
        x1_grid, x2_grid, x3_grid,
        alpha=0.4,
        color='red'
    )
    
    # Manually add surface to legend
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', alpha=0.4, label='Hyperplane')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(red_patch)
    
    # Add the equation as text
    equation = f"Equation: {weights[0]:.2f} + {weights[1]:.2f}x₁ + {weights[2]:.2f}x₂ + {weights[3]:.2f}x₃ = 0"
    ax.text2D(0.05, 0.95, equation, transform=plt.gca().transAxes, 
              fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.legend(handles=handles)
    
    return plt

def visualize_multidim_projection(X, y, weights, title="Multidimensional Projection"):
    """
    For dimensions > 3, visualize by projecting onto the most important dimensions
    
    Args:
        X: Input features of shape (n_samples, n_features)
        y: Target labels of shape (n_samples,)
        weights: Weight vector [w0, w1, w2, ..., wn]
        title: Plot title
    """
    # Get feature importance based on weight magnitudes (excluding bias)
    feature_importance = np.abs(weights[1:])
    
    # Find the indices of the top 2 most important features
    top_feature_indices = np.argsort(feature_importance)[-2:]
    
    plt.figure(figsize=(10, 6))
    
    # Plot each cluster with a different color
    unique_labels = np.unique(y)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        plt.scatter(
            X[y == label, top_feature_indices[0]],
            X[y == label, top_feature_indices[1]],
            color=colors[i],
            alpha=0.7,
            label=f'Cluster {label}'
        )
    
    # Draw the linear decision boundary projected onto the 2 most important features
    # For a 2D subspace, the equation is: w0 + w_i*x_i + w_j*x_j = 0 (where i,j are the important features)
    # Solving for x_j: x_j = (-w0 - w_i*x_i) / w_j
    min_xi, max_xi = X[:, top_feature_indices[0]].min() - 1, X[:, top_feature_indices[0]].max() + 1
    xi_values = np.array([min_xi, max_xi])
    
    # Get the weights for the top features
    w_i = weights[1 + top_feature_indices[0]]
    w_j = weights[1 + top_feature_indices[1]]
    
    # Calculate the line
    xj_values = (-weights[0] - w_i * xi_values) / w_j
    
    plt.plot(xi_values, xj_values, 'r-', lw=2, label='Projected Separator')
    
    # Add the equation as text
    equation = f"Projected Equation: {weights[0]:.2f} + {w_i:.2f}x_{top_feature_indices[0]} + {w_j:.2f}x_{top_feature_indices[1]} = 0"
    plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    full_equation = "Full Equation: " + " + ".join([f"{w:.2f}x_{i}" if i > 0 else f"{w:.2f}" 
                                                 for i, w in enumerate(weights)])
    plt.text(0.05, 0.88, full_equation, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.title(title)
    plt.xlabel(f'Feature {top_feature_indices[0]}')
    plt.ylabel(f'Feature {top_feature_indices[1]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Linear regression analysis using the Least Squares Method')
    parser.add_argument('--input', type=str, required=True, help='Input data file (.npz)')
    parser.add_argument('--output', type=str, help='Output visualization file (.png)')
    parser.add_argument('--dimensions', type=int, help='Specify dimensions if different from input data')
    parser.add_argument('--title', type=str, help='Title for the visualization')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Load input data
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    data = np.load(args.input)
    X = data['X']
    y = data['y']
    
    # Determine dimensions if not specified
    dimensions = args.dimensions if args.dimensions is not None else X.shape[1]
    
    # Convert labels to binary (0, 1) for regression
    y_binary = (y == 1).astype(int)
    
    # Apply least squares method
    weights = least_squares_method(X, y_binary)
    
    # Set default title if not provided
    title = args.title if args.title else f"{dimensions}D Linear Regression Separator"
    
    # Visualize based on dimensions
    if dimensions == 2:
        plt_viz = visualize_linear_separator_2d(X, y, weights, title)
    elif dimensions == 3:
        plt_viz = visualize_linear_separator_3d(X, y, weights, title)
    else:
        plt_viz = visualize_multidim_projection(X, y, weights, f"{dimensions}D Linear Separator Projection")
    
    # Save visualization if output path is provided
    if args.output:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt_viz.savefig(args.output)
        print(f"Visualization saved to {args.output}")
    else:
        plt_viz.show()
    
    print(f"{dimensions}D LSM Weights: {weights}") 