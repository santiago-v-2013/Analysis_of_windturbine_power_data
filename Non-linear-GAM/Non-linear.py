import pandas as pd
import numpy as np
from pygam import LinearGAM, s

import matplotlib.pyplot as plt

# --- 1. Data Loading and Preparation ---
print("Loading data from Data/Location1.csv")
try:
    # Load the dataset from the specified CSV file
    data = pd.read_csv('Data/Location1.csv')
    print(f"Data loaded successfully. Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Check if the 'Power' column exists, as it is the target variable
    if 'Power' not in data.columns:
        print("Error: 'Power' column not found in the data. Please check the CSV file.")
        exit()
except FileNotFoundError:
    # Handle the case where the file is not found
    print("Error: Data file not found. Make sure '../Data/Location1.csv' exists.")
    exit()

print("Data ready.")

# Define target and features based on the actual CSV structure
TARGET = 'Power'
FEATURES = [
    "temperature_2m", "relativehumidity_2m", "dewpoint_2m", 
    "windspeed_10m", "windspeed_100m", "windgusts_10m"
]

y = data[TARGET]
X = data[FEATURES]

# Handle any missing values
if X.isnull().any().any():
    print("Warning: Missing values found in features. Dropping rows with missing values.")
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    print(f"Cleaned data shape: {X.shape}")

# --- 2. Build and Fit the GAM ---
# The model is a sum of smooth functions s() for each feature.
# s(i) corresponds to the i-th column in the feature matrix X.
gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5)).fit(X, y)

# --- 3. Display Model Summary ---
# This provides statistics for each feature's contribution to the model.
print("\n--- GAM Model Summary ---")
gam.summary()


# --- 4. Visualize Partial Dependencies ---
# This is the main strength of GAMs, allowing you to see the effect
# of each variable on the target variable independently.
print("\nGenerating individual partial dependence plots...")

for i, feature in enumerate(FEATURES):
    # Create a new figure for each feature
    plt.figure(figsize=(10, 6))
    
    # Generate the partial dependence data
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
    
    # Plot the partial dependence and confidence intervals
    plt.plot(XX[:, i], pdep, linewidth=2, label='Partial Dependence', color='blue')
    plt.fill_between(XX[:, i], confi[:, 0], confi[:, 1], alpha=0.3, color='red', label='95% Confidence Interval')
    
    # Customize the plot
    plt.title(f'Partial Dependence of {feature} on Power Output', fontsize=14, fontweight='bold')
    plt.xlabel(f'{feature}', fontsize=12)
    plt.ylabel('Partial Dependence on Power', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot as PNG in the Non-linear folder
    filename = f'partial_dependence_{feature}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    
    # Close the figure to free memory
    plt.close()

print("\nScript finished.")