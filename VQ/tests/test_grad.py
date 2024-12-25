import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network with three fully connected layers
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)  # First fully connected layer
        self.fc2 = nn.Linear(20, 10)  # Second fully connected layer
        self.fc3 = nn.Linear(10, 5)   # Third fully connected layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the network
model = SimpleNet()

# Freeze parameters of the second layer by setting requires_grad to False
for param in model.fc2.parameters():
    param.requires_grad = False

# Print which parameters require gradients
print("Layer requires_grad status:")
old_params = {}
for name, param in model.named_parameters():
    old_params[name] = param
    print(f"{name}: {param.requires_grad}")

# Generate a random input tensor and a dummy target for loss computation
input_data = torch.randn(1, 10)
target = torch.randn(1, 5)

# Define a simple mean squared error loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Forward pass
output = model(input_data)
loss = criterion(output, target)

# Backward pass
optimizer.zero_grad()  # Zero out gradients
loss.backward()

# Print gradients of each layer's parameters
print("\nGradients after backward pass:")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} gradient: {param.grad}")
    else:
        print(f"{name} gradient: None (requires_grad is False)")
