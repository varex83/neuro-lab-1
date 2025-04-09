import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import argparse
import os
import sys

def generate_2d_data(n_samples=200, n_clusters=2, random_state=42):
    """
    Generate 2D data with clear clusters
    
    Args:
        n_samples: Total number of data points
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
    
    Returns:
        X: Feature array of shape (n_samples, 2)
        y: Label array of shape (n_samples,)
    """
    # Create cluster centers with good separation
    centers = np.array([
        [-3.0, -3.0],  # Cluster 1 center
        [3.0, 3.0]     # Cluster 2 center
    ])
    
    if n_clusters > 2:
        # Add more centers if needed
        additional_centers = np.array([
            [3.0, -3.0],  # Cluster 3 center
            [-3.0, 3.0],  # Cluster 4 center
        ])
        centers = np.vstack([centers, additional_centers[:n_clusters-2]])
    
    # Generate the blobs
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=centers,
        cluster_std=1.0,
        random_state=random_state
    )
    
    return X, y

def generate_multi_dimensional_data(n_samples=200, n_clusters=2, n_features=3, random_state=42):
    """
    Generate multi-dimensional data with clear clusters
    
    Args:
        n_samples: Total number of data points
        n_clusters: Number of clusters
        n_features: Number of dimensions (features)
        random_state: Random seed for reproducibility
    
    Returns:
        X: Feature array of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
    """
    # Generate the blobs in higher dimensions
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=random_state
    )
    
    return X, y

def visualize_2d_data(X, y, title="2D Data Visualization"):
    """
    Visualize 2D data points with their cluster labels
    
    Args:
        X: Feature array of shape (n_samples, 2)
        y: Label array of shape (n_samples,)
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
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def visualize_3d_data(X, y, title="3D Data Visualization"):
    """
    Visualize 3D data points with their cluster labels
    
    Args:
        X: Feature array of shape (n_samples, 3)
        y: Label array of shape (n_samples,)
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
    
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.legend()
    
    return plt

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate data for neural network analysis')
    parser.add_argument('--dimensions', type=int, default=2, help='Number of dimensions (2, 3, 4, etc.)')
    parser.add_argument('--samples', type=int, default=300, help='Number of data points to generate')
    parser.add_argument('--clusters', type=int, default=2, help='Number of clusters')
    parser.add_argument('--output', type=str, default=None, help='Output file path (.npz)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Generate data based on dimensions
    if args.dimensions == 2:
        X, y = generate_2d_data(n_samples=args.samples, n_clusters=args.clusters, random_state=args.random_state)
        
        # Visualize if requested
        if args.visualize:
            plt_2d = visualize_2d_data(X, y, f"2D Dataset with {args.clusters} Clusters")
            output_img = args.output.replace('.npz', '.png') if args.output else "2d_data_visualization.png"
            plt_2d.savefig(output_img)
            print(f"Visualization saved to {output_img}")
    else:
        X, y = generate_multi_dimensional_data(
            n_samples=args.samples, 
            n_clusters=args.clusters, 
            n_features=args.dimensions,
            random_state=args.random_state
        )
        
        # Visualize if requested and if it's 3D
        if args.visualize and args.dimensions == 3:
            plt_3d = visualize_3d_data(X, y, f"{args.dimensions}D Dataset with {args.clusters} Clusters")
            output_img = args.output.replace('.npz', '.png') if args.output else "3d_data_visualization.png"
            plt_3d.savefig(output_img)
            print(f"Visualization saved to {output_img}")
    
    # Save data if output path is provided
    if args.output:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save the dataset
        np.savez(args.output, X=X, y=y)
        print(f"Dataset saved to {args.output}")
        
        # Also save visualizations in the same directory
        if args.dimensions == 2:
            plt_2d = visualize_2d_data(X, y, f"2D Dataset with {args.clusters} Clusters")
            output_img = args.output.replace('.npz', '.png')
            plt_2d.savefig(output_img)
            print(f"Visualization saved to {output_img}")
        elif args.dimensions == 3:
            plt_3d = visualize_3d_data(X, y, f"3D Dataset with {args.clusters} Clusters")
            output_img = args.output.replace('.npz', '.png')
            plt_3d.savefig(output_img)
            print(f"Visualization saved to {output_img}")
    
    print(f"{args.dimensions}D dataset shape: {X.shape}, Labels: {np.unique(y)}") 