import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from ddmo.equations import plugflow_equation, goodwin_eqs
from scipy.integrate import odeint
import ddmo.differentiation.differentiation as diff
import matplotlib
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.integrate import odeint

matplotlib.use('Qt5Agg')
# Assume X_train and Y_train are your input and output data for training
ReactorModel = plugflow_equation.PlugFlow()
time_array = np.linspace(0, 1, 200)

# Instantiate the model
x_0 = np.column_stack((np.random.rand(len(time_array), 2), np.zeros(len(time_array))))
# Define loss function and optimizer
training_samples = 300
x_dot_trace = np.zeros([len(time_array), len(x_0[0])])

time_series = []
derivatives =[]

# Generate data and derivatves
for i in range(training_samples):
    x_0 = np.random.rand(3)  # Example initial condition, adjust as needed
    x_dot_trace = np.zeros((len(time_array), len(x_0)))

    x = odeint(ReactorModel.dynamics, x_0, time_array)
    x_noise = diff.add_noise(x, time_array, 0.1)
    padded_signal = np.zeros((len(x_noise) * 2, len(x_noise[0, :])))
    x_wein = np.zeros_like(padded_signal)

    time_series.append(x_noise)

    for j in range(len(time_array)):
        x_dot_trace[j, :] = ReactorModel.dynamics(x[j, :], time_array[j])

    derivatives.append(x_dot_trace)

time_series = np.asarray(time_series)
derivatives = np.asarray(derivatives)
class TimeSeriesDataset(Dataset):
    def __init__(self, time_series, derivatives):
        self.time_series = torch.tensor(time_series, dtype=torch.float32)
        self.derivatives = torch.tensor(derivatives, dtype=torch.float32)

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, idx):
        return self.time_series[idx], self.derivatives[idx]

# Neural network model
class DerivativePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        super(DerivativePredictor, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)
        ])

        # Output layer
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))

        # Forward through each hidden layer
        for layer in self.hidden_layers:
            x = self.relu(layer(x))

        x = self.fc2(x)

        return x
# Parameters
input_size = 3  # Assuming 3 time series
hidden_size = 80
output_size = 3  # Assuming 3 derivatives

# Instantiate the model, loss function, and optimizer
model = DerivativePredictor(input_size=3, hidden_size=64, num_hidden_layers=2, output_size=3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Generate training data
# time_series and derivatives are generated as described in your routine

# Create dataset and dataloader
dataset = TimeSeriesDataset(time_series, derivatives)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in dataloader:

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Now you can use the trained model to predict derivatives for new time series
x_0_test = [0, 1, 1]  # Example initial condition for test data
time_array_test = np.linspace(0, 100, 1500)  # Adjust as needed

Goodwin = goodwin_eqs.GoodwinModel()

x_test = odeint(Goodwin.dynamics, x_0_test, time_array_test)
x_noise_test = diff.add_noise(x_test, time_array_test, 0.1)
x_dot_trace = np.zeros_like(x_test)

for j in range(len(time_array_test)):
    x_dot_trace[j, :] = Goodwin.dynamics(x_noise_test[j, :], time_array_test[j])

# Convert the test data to PyTorch tensor
input_tensor = torch.tensor(x_noise_test, dtype=torch.float32)

# Make predictions
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    predictions = model(input_tensor)

# Convert predictions back to NumPy array
predictions_np = predictions.numpy()

plt.plot(time_array_test, predictions[:, 2])
plt.plot(time_array_test, x_dot_trace[:, 2], alpha = 0.5)
plt.show()