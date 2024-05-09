from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_excel("./housing.xls", engine="xlrd")  # or engine="openpyxl"
data.shape
data.head()
X = data[['area', 'bedrooms']] # Features (X)
y = data['price'] # Target variable (y)
(X_train,X_test, y_train,y_test)=train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
# Train the model on the training set
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred
y_test
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
# Calculate R-squared (R2) score
r2 = r2_score(y_test, y_pred)
print("R-squared (R2) Score:",r2)

# Scatter plot of actual values
plt.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted', marker='o')
# Plotting the perfect fit line
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Fit')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.legend()
plt.show()