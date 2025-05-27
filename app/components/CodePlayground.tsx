'use client';

import { Check, Copy, Download, Play, RotateCcw } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

interface CodeExample {
  id: string;
  title: string;
  description: string;
  code: string;
  category: 'numpy' | 'pandas' | 'sklearn' | 'tensorflow' | 'pytorch' | 'visualization';
}

const codeExamples: CodeExample[] = [
  {
    id: 'linear-regression',
    title: 'Linear Regression from Scratch',
    description: 'Implement linear regression using only NumPy',
    category: 'numpy',
    code: `import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X.squeeze() + 1 + 0.1 * np.random.randn(100)

# Add bias term
X_with_bias = np.column_stack([np.ones(X.shape[0]), X])

# Analytical solution: theta = (X^T X)^(-1) X^T y
theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

print(f"Learned parameters: bias={theta[0]:.3f}, weight={theta[1]:.3f}")

# Make predictions
y_pred = X_with_bias @ theta

# Calculate MSE
mse = np.mean((y - y_pred) ** 2)
print(f"Mean Squared Error: {mse:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, label='Data points')
plt.plot(X, y_pred, 'r-', label=f'Fitted line: y = {theta[1]:.2f}x + {theta[0]:.2f}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression from Scratch')
plt.grid(True, alpha=0.3)
plt.show()`
  },
  {
    id: 'neural-network',
    title: 'Neural Network from Scratch',
    description: 'Build a simple neural network with backpropagation',
    category: 'numpy',
    code: `import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Calculate gradients
        dZ2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)
            losses.append(loss)
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses

# Example usage: XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(2, 4, 1, learning_rate=1.0)
losses = nn.train(X, y, 1000)

print("\\nFinal predictions:")
predictions = nn.forward(X)
for i in range(len(X)):
    print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")`
  },
  {
    id: 'cnn-pytorch',
    title: 'CNN with PyTorch',
    description: 'Convolutional Neural Network for image classification',
    category: 'pytorch',
    code: `import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset (commented out for demo)
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
print(f"Using device: {device}")

# Training loop (simplified)
def train_model(model, dataloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}')
                running_loss = 0.0

# Model summary
print("\\nModel Architecture:")
print(model)`
  },
  {
    id: 'transformer',
    title: 'Transformer Attention Mechanism',
    description: 'Implement multi-head attention from scratch',
    category: 'tensorflow',
    code: `import numpy as np
import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calculate the attention weights."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Add the mask to the scaled tensor
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax is normalized on the last axis
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output, attention_weights

# Example usage
d_model = 512
num_heads = 8
seq_len = 10
batch_size = 2

# Create sample input
sample_input = tf.random.normal((batch_size, seq_len, d_model))

# Initialize multi-head attention
mha = MultiHeadAttention(d_model, num_heads)

# Forward pass
output, attention_weights = mha(sample_input, sample_input, sample_input)

print(f"Input shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")

# Visualize attention pattern (simplified)
import matplotlib.pyplot as plt

# Take first head of first batch
attention_head = attention_weights[0, 0].numpy()

plt.figure(figsize=(8, 6))
plt.imshow(attention_head, cmap='Blues')
plt.colorbar()
plt.title('Attention Weights (Head 1)')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.show()`
  }
];

export default function CodePlayground() {
  const [selectedExample, setSelectedExample] = useState<CodeExample>(codeExamples[0]);
  const [code, setCode] = useState(selectedExample.code);
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [copied, setCopied] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setCode(selectedExample.code);
  }, [selectedExample]);

  const runCode = async () => {
    setIsRunning(true);
    setOutput('Running code...\n\n');
    
    // Simulate code execution (in a real implementation, you'd use Pyodide or similar)
    setTimeout(() => {
      const simulatedOutput = `Code executed successfully!

Example output for: ${selectedExample.title}

${selectedExample.category === 'numpy' ? 
  'NumPy arrays created and operations performed.\nResults computed and displayed.' :
  selectedExample.category === 'pytorch' ?
  'PyTorch model initialized.\nTensor operations completed.\nModel ready for training.' :
  selectedExample.category === 'tensorflow' ?
  'TensorFlow model created.\nLayers configured successfully.\nReady for training and inference.' :
  'Code execution completed successfully.'
}

Note: This is a simulated environment. In a full implementation, 
this would execute real Python code using Pyodide or a backend service.`;
      
      setOutput(simulatedOutput);
      setIsRunning(false);
    }, 2000);
  };

  const copyCode = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const resetCode = () => {
    setCode(selectedExample.code);
    setOutput('');
  };

  const downloadCode = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedExample.id}.py`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const categoryColors = {
    numpy: 'bg-blue-100 text-blue-800',
    pandas: 'bg-green-100 text-green-800',
    sklearn: 'bg-purple-100 text-purple-800',
    tensorflow: 'bg-orange-100 text-orange-800',
    pytorch: 'bg-red-100 text-red-800',
    visualization: 'bg-yellow-100 text-yellow-800'
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-12">
      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">
          Interactive Code Playground 💻
        </h2>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Learn by doing! Experiment with machine learning and deep learning code examples.
          Modify the code, run it, and see the results instantly.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Example Selection */}
        <div className="lg:col-span-1">
          <h3 className="text-lg font-semibold mb-4">Code Examples</h3>
          <div className="space-y-2">
            {codeExamples.map((example) => (
              <button
                key={example.id}
                onClick={() => setSelectedExample(example)}
                className={`w-full text-left p-3 rounded-lg border transition-all ${
                  selectedExample.id === example.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <h4 className="font-medium text-sm">{example.title}</h4>
                  <span className={`px-2 py-1 rounded-full text-xs ${categoryColors[example.category]}`}>
                    {example.category}
                  </span>
                </div>
                <p className="text-xs text-gray-600">{example.description}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Code Editor and Output */}
        <div className="lg:col-span-3">
          <div className="bg-white rounded-lg shadow-lg overflow-hidden">
            {/* Header */}
            <div className="bg-gray-50 px-4 py-3 border-b flex items-center justify-between">
              <div>
                <h3 className="font-semibold">{selectedExample.title}</h3>
                <p className="text-sm text-gray-600">{selectedExample.description}</p>
              </div>
              <div className="flex space-x-2">
                <button
                  onClick={copyCode}
                  className="p-2 text-gray-600 hover:text-gray-800 transition-colors"
                  title="Copy code"
                >
                  {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                </button>
                <button
                  onClick={resetCode}
                  className="p-2 text-gray-600 hover:text-gray-800 transition-colors"
                  title="Reset code"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
                <button
                  onClick={downloadCode}
                  className="p-2 text-gray-600 hover:text-gray-800 transition-colors"
                  title="Download code"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2">
              {/* Code Editor */}
              <div className="border-r">
                <div className="bg-gray-100 px-4 py-2 text-sm font-medium">
                  Code Editor
                </div>
                <div className="relative">
                  <textarea
                    ref={textareaRef}
                    value={code}
                    onChange={(e) => setCode(e.target.value)}
                    className="w-full h-96 p-4 font-mono text-sm border-none resize-none focus:outline-none"
                    placeholder="Write your Python code here..."
                  />
                </div>
                <div className="bg-gray-50 px-4 py-2 border-t">
                  <button
                    onClick={runCode}
                    disabled={isRunning}
                    className="flex items-center space-x-2 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <Play className="w-4 h-4" />
                    <span>{isRunning ? 'Running...' : 'Run Code'}</span>
                  </button>
                </div>
              </div>

              {/* Output */}
              <div>
                <div className="bg-gray-100 px-4 py-2 text-sm font-medium">
                  Output
                </div>
                <div className="h-96 p-4 bg-gray-900 text-green-400 font-mono text-sm overflow-auto">
                  <pre className="whitespace-pre-wrap">
                    {output || 'Click "Run Code" to see the output here...'}
                  </pre>
                </div>
              </div>
            </div>
          </div>

          {/* Tips */}
          <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-semibold text-blue-900 mb-2">💡 Tips for Learning</h4>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Modify the code to experiment with different parameters</li>
              <li>• Try changing the model architecture or hyperparameters</li>
              <li>• Add print statements to understand the data flow</li>
              <li>• Copy the code to your local environment for full execution</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
} 