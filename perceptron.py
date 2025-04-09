import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import time
import argparse
import os
import sys

class Perceptron:
    """
    A single-layer perceptron implementation with configurable activation function
    """
    
    def __init__(self, n_features, learning_rate=0.01, max_iterations=1000, 
                 activation='step', random_state=42):
        """
        Initialize the perceptron
        
        Args:
            n_features: Number of input features
            learning_rate: Learning rate for weight updates
            max_iterations: Maximum number of training iterations
            activation: Activation function to use ('step', 'sigmoid', 'relu', 'tanh')
            random_state: Random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.activation_type = activation
        
        # Initialize weights and bias with small random values
        np.random.seed(random_state)
        # +1 for the bias term
        self.weights = np.random.randn(n_features + 1) * 0.1
        
        # Track the error during training
        self.errors = []
        self.iterations_performed = 0
        
    def activation(self, x):
        """
        Apply the activation function
        
        Args:
            x: Input value
            
        Returns:
            Activated value
        """
        if self.activation_type == 'step':
            return 1 if x >= 0 else 0
        elif self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_type == 'relu':
            return max(0, x)
        elif self.activation_type == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_type}")
    
    def activation_batch(self, X):
        """
        Apply activation function to a batch of inputs
        
        Args:
            X: Input array
            
        Returns:
            Activated values
        """
        if self.activation_type == 'step':
            return np.where(X >= 0, 1, 0)
        elif self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-X))
        elif self.activation_type == 'relu':
            return np.maximum(0, X)
        elif self.activation_type == 'tanh':
            return np.tanh(X)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_type}")
    
    def predict_single(self, x):
        """
        Make a prediction for a single sample
        
        Args:
            x: Input features (without bias)
            
        Returns:
            Predicted class (0 or 1)
        """
        # Add bias term
        x_with_bias = np.insert(x, 0, 1)
        # Calculate the net input
        net_input = np.dot(x_with_bias, self.weights)
        # Apply activation function
        prediction = self.activation(net_input)
        
        if self.activation_type == 'step':
            return prediction
        else:
            # For non-step functions, apply thresholding
            return 1 if prediction >= 0.5 else 0
    
    def predict(self, X):
        """
        Make predictions for multiple samples
        
        Args:
            X: Input features array of shape (n_samples, n_features)
            
        Returns:
            Predicted classes array of shape (n_samples,)
        """
        # Add bias term
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        # Calculate the net input
        net_input = np.dot(X_with_bias, self.weights)
        # Apply activation function
        predictions = self.activation_batch(net_input)
        
        if self.activation_type != 'step':
            # For non-step functions, apply thresholding
            predictions = np.where(predictions >= 0.5, 1, 0)
            
        return predictions
    
    def fit(self, X, y, verbose=True):
        """
        Train the perceptron on the given data
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            verbose: Whether to print progress information
            
        Returns:
            self: Trained perceptron
        """
        start_time = time.time()
        self.errors = []
        
        for iteration in range(self.max_iterations):
            error_count = 0
            
            # Process each sample one by one (online learning)
            for i in range(len(X)):
                # Make prediction
                x_i = X[i]
                y_i = y[i]
                y_pred = self.predict_single(x_i)
                
                # Update weights if prediction is wrong
                if y_i != y_pred:
                    error_count += 1
                    
                    # Add bias term to the input
                    x_i_with_bias = np.insert(x_i, 0, 1)
                    
                    # Update weights: w = w + learning_rate * (y - y_pred) * x
                    self.weights += self.learning_rate * (y_i - y_pred) * x_i_with_bias
            
            # Record the number of errors
            self.errors.append(error_count)
            
            # Check if converged (no errors)
            if error_count == 0:
                if verbose:
                    print(f"Converged after {iteration+1} iterations!")
                self.iterations_performed = iteration + 1
                break
                
            # Print progress
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration+1}/{self.max_iterations}, Errors: {error_count}/{len(X)}")
        
        else:
            # This block runs if the loop completes without a break
            self.iterations_performed = self.max_iterations
            if verbose:
                print(f"Reached maximum iterations ({self.max_iterations}) without convergence.")
        
        training_time = time.time() - start_time
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds")
        
        return self
    
    def evaluate(self, X, y):
        """
        Evaluate the perceptron performance
        
        Args:
            X: Test features of shape (n_samples, n_features)
            y: True labels of shape (n_samples,)
            
        Returns:
            accuracy: Classification accuracy
            cm: Confusion matrix
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        
        return accuracy, cm
    
    def plot_decision_boundary_2d(self, X, y, title=None, output_file=None):
        """
        Plot the decision boundary for 2D input
        
        Args:
            X: Input features of shape (n_samples, 2)
            y: Target labels of shape (n_samples,)
            title: Plot title
            output_file: Path to save the plot
            
        Returns:
            plt: Matplotlib figure
        """
        if X.shape[1] != 2:
            raise ValueError("This method only works for 2D input data.")
        
        plt.figure(figsize=(10, 6))
        
        # Plot each class with a different color
        unique_labels = np.unique(y)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            plt.scatter(
                X[y == label, 0],
                X[y == label, 1],
                color=colors[i],
                alpha=0.7,
                label=f'Class {label}'
            )
        
        # Create a mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        
        # Make predictions on the mesh grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
        plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
        
        # Draw the perceptron's linear boundary
        # For a 2D feature space with bias, the equation is: w0 + w1*x1 + w2*x2 = 0
        # Solving for x2: x2 = (-w0 - w1*x1) / w2
        x1_values = np.array([x_min, x_max])
        x2_values = (-self.weights[0] - self.weights[1] * x1_values) / self.weights[2]
        
        plt.plot(x1_values, x2_values, 'k-', lw=2, label='Decision Boundary')
        
        # Add the equation as text
        equation = f"Equation: {self.weights[0]:.2f} + {self.weights[1]:.2f}x₁ + {self.weights[2]:.2f}x₂ = 0"
        plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, 
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.title(title or f"Perceptron Decision Boundary (Activation: {self.activation_type})")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot if output path is provided
        if output_file:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            plt.savefig(output_file)
            print(f"Decision boundary plot saved to {output_file}")
        
        return plt
    
    def plot_training_curve(self, output_file=None):
        """
        Plot the training error curve
        
        Args:
            output_file: Path to save the plot
            
        Returns:
            plt: Matplotlib figure
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.errors) + 1), self.errors, marker='o')
        plt.title(f'Perceptron Training Curve (Activation: {self.activation_type})')
        plt.xlabel('Iterations')
        plt.ylabel('Number of Misclassifications')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot if output path is provided
        if output_file:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            plt.savefig(output_file)
            print(f"Training curve plot saved to {output_file}")
        
        return plt

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train and analyze a perceptron with configurable activation function')
    parser.add_argument('--input', type=str, required=True, help='Input data file (.npz)')
    parser.add_argument('--output', type=str, help='Output visualization file (.png)')
    parser.add_argument('--activation', type=str, default='step', 
                        choices=['step', 'sigmoid', 'relu', 'tanh'],
                        help='Activation function to use')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate for weight updates')
    parser.add_argument('--max-iterations', type=int, default=1000, help='Maximum number of training iterations')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--plot-training-curve', action='store_true', help='Also plot the training curve')
    
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
    
    # Initialize and train perceptron
    perceptron = Perceptron(
        n_features=X.shape[1],
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
        activation=args.activation,
        random_state=args.random_state
    )
    
    # Train the perceptron
    perceptron.fit(X, y)
    
    # Evaluate the perceptron
    accuracy, cm = perceptron.evaluate(X, y)
    print(f"{args.activation} Activation - Accuracy: {accuracy:.4f}")
    print(f"{args.activation} Activation - Confusion Matrix:\n{cm}")
    
    # Plot the decision boundary if it's 2D data
    if X.shape[1] == 2:
        decision_output = args.output
        plt_decision = perceptron.plot_decision_boundary_2d(X, y, output_file=decision_output)
    
    # Plot the training curve if requested
    if args.plot_training_curve:
        curve_output = args.output.replace('.png', '_training_curve.png') if args.output else f"perceptron_{args.activation}_training_curve.png"
        plt_training = perceptron.plot_training_curve(output_file=curve_output)
    
    print(f"{args.activation} Activation - Weights: {perceptron.weights}")
    print(f"{args.activation} Activation - Iterations performed: {perceptron.iterations_performed}") 