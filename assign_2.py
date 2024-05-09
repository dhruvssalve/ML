import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
data = pd.read_csv("./mammals.csv")
X=data['brain_wt']
y=data['body_wt']
X = np.array(X).reshape(-1, 1)
data.head()
(X_train,X_test, y_train,y_test)=train_test_split(X, y, test_size=0.2, random_state=40)
# Create a linear regression model
model = LinearRegression()
# Train the model on the training set
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred
# Assuming you have trained your model and made predictions
mse = mean_squared_error(X_test, y_pred)
r2 = r2_score(X_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
plt.scatter(X_test, y_test, color='black') # Actual data points
plt.plot(X_test, y_pred, color='blue', linewidth=3) # Linear regression line
plt.xlabel('Body Weight')
plt.ylabel('Brain Weight')
plt.title('Linear Regression Model')
plt.show()