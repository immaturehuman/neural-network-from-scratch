import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize a simple neural network with one hidden layer.
        
        Parameters:
        - input_size: number of input features
        - hidden_size: number of neurons in hidden layer
        - output_size: number of output neurons
        """
        # Initialize weights with random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        # Store gradients for visualization
        self.gradients = {}
        self.cache = {}
        
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        return x * (1 - x)
    
    def forward_propagation(self, X):
        """
        Perform forward propagation through the network.
        
        Parameters:
        - X: input data (m samples, n features)
        
        Returns:
        - A2: output of the network
        """
        # Layer 1: input to hidden
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        
        # Layer 2: hidden to output
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        
        # Cache values for backpropagation
        self.cache['Z1'] = Z1
        self.cache['A1'] = A1
        self.cache['Z2'] = Z2
        self.cache['A2'] = A2
        
        return A2
    
    def compute_loss(self, Y, Y_hat):
        """
        Compute binary cross-entropy loss.
        
        Parameters:
        - Y: true labels
        - Y_hat: predicted labels
        
        Returns:
        - loss: computed loss
        """
        m = Y.shape[0]
        # Avoid log(0) errors
        Y_hat = np.clip(Y_hat, 1e-15, 1 - 1e-15)
        loss = -np.mean(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
        return loss
    
    def backward_propagation(self, X, Y):
        """
        Perform backpropagation to compute gradients.
        
        Parameters:
        - X: input data
        - Y: true labels
        
        Returns:
        - gradients dictionary
        """
        m = X.shape[0]
        
        # Get cached values from forward propagation
        A1 = self.cache['A1']
        A2 = self.cache['A2']
        
        # Output layer gradients
        dZ2 = A2 - Y  # Derivative of loss with respect to Z2
        dW2 = (1/m) * np.dot(A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(A1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Store gradients
        self.gradients = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2
        }
        
        return self.gradients
    
    def update_parameters(self, learning_rate=0.1):
        """
        Update network parameters using gradient descent.
        
        Parameters:
        - learning_rate: step size for gradient descent
        """
        self.W1 -= learning_rate * self.gradients['dW1']
        self.b1 -= learning_rate * self.gradients['db1']
        self.W2 -= learning_rate * self.gradients['dW2']
        self.b2 -= learning_rate * self.gradients['db2']
    
    def train(self, X, Y, epochs=1000, learning_rate=0.1, verbose=True):
        """
        Train the neural network.
        
        Parameters:
        - X: input data
        - Y: true labels
        - epochs: number of training iterations
        - learning_rate: step size for gradient descent
        - verbose: whether to print progress
        
        Returns:
        - losses: list of loss values over training
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward propagation
            Y_hat = self.forward_propagation(X)
            
            # Compute loss
            loss = self.compute_loss(Y, Y_hat)
            losses.append(loss)
            
            # Backward propagation
            self.backward_propagation(X, Y)
            
            # Update parameters
            self.update_parameters(learning_rate)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """
        Make predictions using the trained network.
        
        Parameters:
        - X: input data
        
        Returns:
        - predictions: binary predictions (0 or 1)
        """
        Y_hat = self.forward_propagation(X)
        predictions = (Y_hat > 0.5).astype(int)
        return predictions
    
    def accuracy(self, Y_true, Y_pred):
        """
        Compute accuracy of predictions.
        
        Parameters:
        - Y_true: true labels
        - Y_pred: predicted labels
        
        Returns:
        - accuracy: percentage of correct predictions
        """
        return np.mean(Y_true == Y_pred) * 100

# Create a simple XOR dataset
def create_xor_dataset():
    """Create XOR dataset for binary classification."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])  # XOR function
    return X, Y

# Visualize the training process
def visualize_training(losses, network, X, Y):
    """Create visualizations of the training process."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot loss curve
    axes[0, 0].plot(losses)
    axes[0, 0].set_title('Training Loss Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Plot decision boundary
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = network.forward_propagation(grid_points)
    Z = Z.reshape(xx.shape)
    
    axes[0, 1].contourf(xx, yy, Z, levels=20, cmap=plt.cm.RdYlBu, alpha=0.8)
    axes[0, 1].scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap=plt.cm.RdYlBu, 
                       edgecolors='k', s=100)
    axes[0, 1].set_title('Decision Boundary')
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    
    # Visualize network architecture
    axes[1, 0].axis('off')
    network_text = f"""
    Network Architecture:
    - Input Layer: 2 neurons
    - Hidden Layer: 4 neurons
    - Output Layer: 1 neuron
    
    Parameters:
    - W1 shape: {network.W1.shape}
    - b1 shape: {network.b1.shape}
    - W2 shape: {network.W2.shape}
    - b2 shape: {network.b2.shape}
    
    Final Loss: {losses[-1]:.4f}
    """
    axes[1, 0].text(0.1, 0.5, network_text, fontsize=12, 
                    verticalalignment='center', fontfamily='monospace')
    axes[1, 0].set_title('Network Information')
    
    # Show predictions
    predictions = network.predict(X)
    accuracy = network.accuracy(Y, predictions)
    
    axes[1, 1].axis('off')
    results_text = f"""
    XOR Predictions:
    Input [0, 0] -> Predicted: {predictions[0, 0]}, Actual: {Y[0, 0]}
    Input [0, 1] -> Predicted: {predictions[1, 0]}, Actual: {Y[1, 0]}
    Input [1, 0] -> Predicted: {predictions[2, 0]}, Actual: {Y[2, 0]}
    Input [1, 1] -> Predicted: {predictions[3, 0]}, Actual: {Y[3, 0]}
    
    Accuracy: {accuracy:.1f}%
    """
    axes[1, 1].text(0.1, 0.5, results_text, fontsize=12, 
                    verticalalignment='center', fontfamily='monospace')
    axes[1, 1].set_title('Predictions')
    
    plt.tight_layout()
    plt.show()

# Step-by-step backpropagation demonstration
def demonstrate_backpropagation_step_by_step():
    """Show step-by-step backpropagation calculations for one training example."""
    print("="*60)
    print("STEP-BY-STEP BACKPROPAGATION DEMONSTRATION")
    print("="*60)
    
    # Simple example with one training sample
    X_single = np.array([[0, 1]])  # Input: [0, 1]
    Y_single = np.array([[1]])     # Expected output: 1 (for XOR)
    
    # Create a tiny network
    nn_simple = NeuralNetwork(2, 2, 1)
    
    print("\n1. INITIAL PARAMETERS:")
    print(f"   W1 (input to hidden):\n{nn_simple.W1}")
    print(f"   b1 (hidden bias):\n{nn_simple.b1}")
    print(f"   W2 (hidden to output):\n{nn_simple.W2}")
    print(f"   b2 (output bias):\n{nn_simple.b2}")
    
    print("\n2. FORWARD PROPAGATION:")
    # Manual forward propagation
    Z1 = np.dot(X_single, nn_simple.W1) + nn_simple.b1
    A1 = nn_simple.sigmoid(Z1)
    Z2 = np.dot(A1, nn_simple.W2) + nn_simple.b2
    A2 = nn_simple.sigmoid(Z2)
    
    print(f"   Z1 = X·W1 + b1 = {Z1}")
    print(f"   A1 = sigmoid(Z1) = {A1}")
    print(f"   Z2 = A1·W2 + b2 = {Z2}")
    print(f"   A2 = sigmoid(Z2) = {A2} (Prediction)")
    print(f"   Expected output: {Y_single[0, 0]}")
    
    print("\n3. LOSS CALCULATION:")
    loss = - (Y_single * np.log(A2) + (1 - Y_single) * np.log(1 - A2))
    print(f"   Loss = -[y·log(ŷ) + (1-y)·log(1-ŷ)] = {loss[0, 0]:.6f}")
    
    print("\n4. BACKPROPAGATION:")
    # Output layer gradients
    dZ2 = A2 - Y_single
    print(f"   dZ2 = A2 - Y = {dZ2[0, 0]:.6f}")
    
    dW2 = np.dot(A1.T, dZ2)
    print(f"   dW2 = A1ᵀ · dZ2 = {dW2}")
    
    db2 = dZ2
    print(f"   db2 = dZ2 = {db2}")
    
    # Hidden layer gradients
    dA1 = np.dot(dZ2, nn_simple.W2.T)
    print(f"   dA1 = dZ2 · W2ᵀ = {dA1}")
    
    dZ1 = dA1 * nn_simple.sigmoid_derivative(A1)
    print(f"   dZ1 = dA1 * A1*(1-A1) = {dZ1}")
    
    dW1 = np.dot(X_single.T, dZ1)
    print(f"   dW1 = Xᵀ · dZ1 = {dW1}")
    
    db1 = dZ1
    print(f"   db1 = dZ1 = {db1}")
    
    print("\n5. PARAMETER UPDATES (learning rate = 0.1):")
    print(f"   New W1 = W1 - 0.1*dW1 =\n{nn_simple.W1 - 0.1*dW1}")
    print(f"   New W2 = W2 - 0.1*dW2 =\n{nn_simple.W2 - 0.1*dW2}")
    
    print("\n" + "="*60)
    print("This completes one iteration of forward/backward propagation!")
    print("="*60)

# Main execution
if __name__ == "__main__":
    # Create dataset
    X, Y = create_xor_dataset()
    
    print("XOR Dataset:")
    print(f"Input features (X):\n{X}")
    print(f"Labels (Y):\n{Y}")
    print()
    
    # Demonstrate step-by-step backpropagation
    demonstrate_backpropagation_step_by_step()
    
    print("\n" + "="*60)
    print("FULL TRAINING EXAMPLE")
    print("="*60)
    
    # Create and train neural network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    
    print(f"\nTraining neural network on XOR problem...")
    print(f"Architecture: 2 input → 4 hidden → 1 output")
    print(f"Training for 1000 epochs...")
    
    # Train the network
    losses = nn.train(X, Y, epochs=1000, learning_rate=0.5, verbose=True)
    
    # Make predictions
    predictions = nn.predict(X)
    accuracy = nn.accuracy(Y, predictions)
    
    print(f"\nFinal accuracy on training set: {accuracy:.1f}%")
    
    # Visualize results
    visualize_training(losses, nn, X, Y)
