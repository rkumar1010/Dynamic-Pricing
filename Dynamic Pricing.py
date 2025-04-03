import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Generate synthetic dataset
def generate_data(n_samples=1000):
    np.random.seed(42)
    demand = np.random.randint(50, 500, n_samples)
    competition_price = np.random.uniform(10, 100, n_samples)
    seasonality = np.random.randint(1, 5, n_samples)  # 1: Low, 5: High
    base_price = np.random.uniform(20, 80, n_samples)
    optimal_price = base_price + (demand * 0.05) - (competition_price * 0.3) + (seasonality * 5)
    
    df = pd.DataFrame({
        'demand': demand,
        'competition_price': competition_price,
        'seasonality': seasonality,
        'base_price': base_price,
        'optimal_price': optimal_price
    })
    return df

data = generate_data()

# Split data
X = data[['demand', 'competition_price', 'seasonality', 'base_price']]
y = data['optimal_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Actual Optimal Price")
plt.ylabel("Predicted Optimal Price")
plt.title("Actual vs Predicted Prices")
plt.show()
