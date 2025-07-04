import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Load the data
data = pd.read_csv('Data/Location1.csv')

# Data exploration and preprocessing
print("=== DATA EXPLORATION ===")
print(f"Dataset shape: {data.shape}")
print(f"Missing values in temperature_2m: {data['temperature_2m'].isnull().sum()}")
print(f"Missing values in Power: {data['Power'].isnull().sum()}")
print(f"Temperature range: {data['temperature_2m'].min():.2f} to {data['temperature_2m'].max():.2f}")
print(f"Power range: {data['Power'].min():.2f} to {data['Power'].max():.2f}")

# Handle missing values if any
if data.isnull().sum().sum() > 0:
    print("Warning: Missing values detected. Removing rows with missing data.")
    data = data.dropna()
    print(f"Dataset shape after removing missing values: {data.shape}")

print("-" * 60)

# Define predictor and target variables
X = data[['temperature_2m']]
y = data['Power']

# Check for outliers using IQR method
def detect_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

temp_outliers = detect_outliers(data, 'temperature_2m')
power_outliers = detect_outliers(data, 'Power')

print(f"Temperature outliers: {len(temp_outliers)} ({len(temp_outliers)/len(data)*100:.1f}%)")
print(f"Power outliers: {len(power_outliers)} ({len(power_outliers)/len(data)*100:.1f}%)")

# Calculate correlation
correlation = data['temperature_2m'].corr(data['Power'])
print(f"Correlation between temperature and power: {correlation:.4f}")

# Split the data into training and testing sets (80/20 rule)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print("-" * 60)

# Fit linear regression model on training data
model = LinearRegression()
model.fit(X_train, y_train)

# Print model coefficients
print("=== MODEL COEFFICIENTS ===")
print(f'Intercept: {model.intercept_:.4f}')
print(f'Coefficient for temperature: {model.coef_[0]:.4f}')
print(f'Equation: Power = {model.coef_[0]:.4f} * temperature + {model.intercept_:.4f}')
print("-" * 60)

# Make predictions on both training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate residuals
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

# Evaluate model performance
print("=== MODEL PERFORMANCE ===")
print("Training Set Performance:")
print(f"Mean Squared Error: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_train, y_train_pred):.4f}")
print(f"R² Score: {r2_score(y_train, y_train_pred):.4f}")
print()

print("Testing Set Performance:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_test_pred):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"R² Score: {r2_score(y_test, y_test_pred):.4f}")

# Calculate and display performance difference
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"\nR² difference (train - test): {train_r2 - test_r2:.4f}")
if abs(train_r2 - test_r2) > 0.1:
    print("⚠️  Warning: Large difference between training and testing R² may indicate overfitting")
else:
    print("✅ Model appears to generalize well to unseen data")
    
# Statistical significance test for the coefficient
# Perform t-test for the slope coefficient
n = len(X_train)
p = 1  # number of predictors
df = n - p - 1  # degrees of freedom
mse = mean_squared_error(y_train, y_train_pred)
x_mean = X_train.mean().values[0]
ss_x = ((X_train - x_mean) ** 2).sum().values[0]
se_slope = np.sqrt(mse / ss_x)
t_stat = model.coef_[0] / se_slope
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

print(f"\nStatistical Significance:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Coefficient is {'significant' if p_value < 0.05 else 'not significant'} at α=0.05")
print("-" * 60)

# Create a standalone scatter plot with regression line for better visualization
plt.figure(figsize=(12, 8))

# Plot training and testing data
plt.scatter(X_train, y_train, alpha=0.6, color='blue', label='Training Data', s=20)
plt.scatter(X_test, y_test, alpha=0.6, color='red', label='Testing Data', s=20)

# Create regression line
x_range = np.linspace(X.min().values[0], X.max().values[0], 100)
x_range_df = pd.DataFrame(x_range.reshape(-1, 1), columns=['temperature_2m'])
y_range = model.predict(x_range_df)
plt.plot(x_range, y_range, 'g-', linewidth=3, label='Regression Line')

# Add equation to the plot
equation_text = f'Power = {model.coef_[0]:.4f} × Temperature + {model.intercept_:.4f}'
plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Add R² value
r2_text = f'R² = {test_r2:.4f}'
plt.text(0.05, 0.88, r2_text, transform=plt.gca().transAxes, 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

# Formatting
plt.xlabel('Temperature (°C)', fontsize=14)
plt.ylabel('Power', fontsize=14)
plt.title('Linear Regression: Temperature vs Power\nStandalone Visualization', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Add correlation info
correlation_text = f'Correlation: {correlation:.4f}'
plt.text(0.05, 0.81, correlation_text, transform=plt.gca().transAxes, 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

plt.tight_layout()
plt.savefig('Linear/scatter_plot_with_regression.png', dpi=300, bbox_inches='tight')
plt.close()

print("📊 Standalone scatter plot with regression line saved as:")
print("   - Linear/scatter_plot_with_regression.png")
print("-" * 60)

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Linear Regression Analysis: Temperature vs Power', fontsize=16, fontweight='bold')

# 1. Scatter plot with regression line
axes[0, 0].scatter(X_train, y_train, alpha=0.6, color='blue', label='Training Data')
axes[0, 0].scatter(X_test, y_test, alpha=0.6, color='red', label='Testing Data')
x_range = np.linspace(X.min().values[0], X.max().values[0], 100)
x_range_df = pd.DataFrame(x_range.reshape(-1, 1), columns=['temperature_2m'])
y_range = model.predict(x_range_df)
axes[0, 0].plot(x_range, y_range, 'g-', linewidth=2, label='Regression Line')
axes[0, 0].set_xlabel('Temperature (°C)')
axes[0, 0].set_ylabel('Power')
axes[0, 0].set_title('Data Points and Regression Line')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals plot
axes[0, 1].scatter(y_train_pred, train_residuals, alpha=0.6, color='blue', label='Training')
axes[0, 1].scatter(y_test_pred, test_residuals, alpha=0.6, color='red', label='Testing')
axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Predicted Power')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residuals vs Predicted Values')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Q-Q plot for residuals normality
stats.probplot(train_residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Training Residuals)')
axes[1, 0].grid(True, alpha=0.3)

# 4. Predicted vs Actual
axes[1, 1].scatter(y_train, y_train_pred, alpha=0.6, color='blue', label='Training')
axes[1, 1].scatter(y_test, y_test_pred, alpha=0.6, color='red', label='Testing')
min_val = min(y.min(), y_train_pred.min(), y_test_pred.min())
max_val = max(y.max(), y_train_pred.max(), y_test_pred.max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
axes[1, 1].set_xlabel('Actual Power')
axes[1, 1].set_ylabel('Predicted Power')
axes[1, 1].set_title('Predicted vs Actual Values')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Linear/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("📊 Comprehensive analysis plots saved as:")
print("   - Linear/comprehensive_analysis.png")
print("-" * 60)

# Create individual plots for detailed analysis
print("=== CREATING INDIVIDUAL PLOTS ===")

# 1. Individual scatter plot with regression line
plt.figure(figsize=(10, 8))
plt.scatter(X_train, y_train, alpha=0.6, color='blue', label='Training Data', s=20)
plt.scatter(X_test, y_test, alpha=0.6, color='red', label='Testing Data', s=20)
x_range = np.linspace(X.min().values[0], X.max().values[0], 100)
x_range_df = pd.DataFrame(x_range.reshape(-1, 1), columns=['temperature_2m'])
y_range = model.predict(x_range_df)
plt.plot(x_range, y_range, 'g-', linewidth=2, label='Regression Line')
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.title('Temperature vs Power - Training and Testing Data', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Linear/individual_scatter_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Individual residuals plot
plt.figure(figsize=(10, 8))
plt.scatter(y_train_pred, train_residuals, alpha=0.6, color='blue', label='Training', s=20)
plt.scatter(y_test_pred, test_residuals, alpha=0.6, color='red', label='Testing', s=20)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Predicted Power', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Linear/residuals_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Individual Q-Q plot
plt.figure(figsize=(10, 8))
stats.probplot(train_residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot - Normality of Residuals', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Linear/qq_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Individual predicted vs actual plot
plt.figure(figsize=(10, 8))
plt.scatter(y_train, y_train_pred, alpha=0.6, color='blue', label='Training', s=20)
plt.scatter(y_test, y_test_pred, alpha=0.6, color='red', label='Testing', s=20)
min_val = min(y.min(), y_train_pred.min(), y_test_pred.min())
max_val = max(y.max(), y_train_pred.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
plt.xlabel('Actual Power', fontsize=12)
plt.ylabel('Predicted Power', fontsize=12)
plt.title('Predicted vs Actual Values', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Linear/predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()

print("📊 Individual plots saved:")
print("   - Linear/individual_scatter_plot.png")
print("   - Linear/residuals_plot.png")
print("   - Linear/qq_plot.png")
print("   - Linear/predicted_vs_actual.png")
print("-" * 60)

# Model assumptions check
print("=== MODEL ASSUMPTIONS CHECK ===")

# 1. Linearity check
print("1. Linearity: Check the scatter plot above")

# 2. Independence of residuals
print("2. Independence: Durbin-Watson test would be needed for time series")

# 3. Homoscedasticity (constant variance)
print("3. Homoscedasticity: Check residuals plot - should show random scatter")

# 4. Normality of residuals
# Use a sample for large datasets to avoid Shapiro-Wilk warning
if len(train_residuals) > 5000:
    sample_size = 5000
    sample_residuals = np.random.choice(train_residuals, size=sample_size, replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(sample_residuals)
    print(f"4. Normality of residuals (sample of {sample_size} from {len(train_residuals)} observations):")
else:
    shapiro_stat, shapiro_p = stats.shapiro(train_residuals)
    print(f"4. Normality of residuals:")
    
print(f"   Shapiro-Wilk test: W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")
print(f"   Residuals are {'normally distributed' if shapiro_p > 0.05 else 'not normally distributed'} (α=0.05)")

# Additional normality tests for large datasets
if len(train_residuals) > 5000:
    jb_stat, jb_p = jarque_bera(train_residuals)
    print(f"   Jarque-Bera test: JB = {jb_stat:.4f}, p = {jb_p:.4f}")
    print(f"   Residuals are {'normally distributed' if jb_p > 0.05 else 'not normally distributed'} (α=0.05, JB test)")

print("-" * 60)

# Summary and recommendations
print("=== SUMMARY AND RECOMMENDATIONS ===")
print(f"Model Performance: R² = {test_r2:.4f} ({test_r2*100:.1f}% of variance explained)")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")

if test_r2 > 0.7:
    print("Strong relationship between temperature and power")
elif test_r2 > 0.5:
    print("Moderate relationship - consider additional features")
else:
    print("Weak relationship - linear model may not be appropriate")

print("\nRecommendations for improvement:")
print("1. Feature Engineering: Consider polynomial features, interactions")
print("2. Additional Variables: Include other relevant predictors")
print("3. Model Validation: Use cross-validation for robust evaluation")
print("4. Advanced Models: Try Ridge/Lasso regression for regularization")
print("5. Non-linear Models: Consider tree-based models if assumptions are violated")

print("\n" + "=" * 60)
print("=== SAVED FILES SUMMARY ===")
print("All plots have been saved in the 'Linear' folder:")
print("Main visualizations:")
print("   • scatter_plot_with_regression.png - Standalone scatter plot with regression line")
print("   • comprehensive_analysis.png - 4-subplot comprehensive analysis")
print("Individual analysis plots:")
print("   • individual_scatter_plot.png - Clean scatter plot with regression line")
print("   • residuals_plot.png - Residuals analysis")
print("   • qq_plot.png - Q-Q plot for normality testing")
print("   • predicted_vs_actual.png - Model prediction accuracy")
print("\n All files are high-resolution PNG images (300 DPI)")
print("=" * 60)