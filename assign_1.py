import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('./SOCR-HeightWeight.csv')
data = data.sort_values('Height(Inches)').reset_index(drop=True)
data.head()
data.describe()
model = LinearRegression()
model.fit(data[['Height(Inches)']],data['Weight(Pounds)'])
# Extracting the features (X) and target variable (y)
X = data[['Height(Inches)']]
y = data['Weight(Pounds)']
X.dropna()
y_pred=model.predict(X)
# Plot the actual data
plt.scatter(X, y, label='Actual Data')
# Plot the predictions
plt.plot(X, y_pred, 'r-', label='Linear Regression Prediction')
# Add labels and legend
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.legend()
# Show the plot
plt.show()