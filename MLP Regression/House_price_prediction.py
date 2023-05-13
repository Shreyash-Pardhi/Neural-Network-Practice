#  required functions are imported from sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading the King County housing dataset
df = pd.read_csv('kc_house_data.csv')

### Exploratory Data Analysis

df.info()
df.head()
df.columns

#Checking for missing Data
df.isnull().sum()

#describe the statistics for each row of the  DataFrame
df.describe().transpose()

# some visualizations using sns
import seaborn as sb # statistical data visualization library based on matplotlib
#sb.countplot(x=df['bedrooms'])
#sb.scatterplot(x=df['sqft_living'],y= df['price'])
#sb.heatmap(df.corr(), cmap="YlGnBu", annot=True)

#droping some features
df = df.drop(['id', 'date', 'zipcode', 'lat', 'long','sqft_above'], axis=1)

# Preparing input and output data
X = df.drop('price', axis =1).values #drop output varibale
y = df.price.values

## Train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

## scaling the data
from sklearn.preprocessing import StandardScaler,MinMaxScaler, RobustScaler
#scaler = MinMaxScaler()
#scaler=RobustScaler()
scaler = StandardScaler()
X_train_stdsc = scaler.fit_transform(X_train)
X_test_stdsc = scaler.fit_transform(X_test)

from sklearn.neural_network import MLPRegressor #loss function is squared error
# Here alpha is a factor for L2 regularization
# mlp_reg = MLPRegressor(hidden_layer_sizes=(150,150,150),max_iter = 600,activation = 'relu',alpha=0.05,verbose=3) #R2=.77
mlp_reg = MLPRegressor(hidden_layer_sizes=(150,150,150),max_iter = 1000,activation = 'relu',alpha=0.5,early_stopping=True, verbose=3) #R2=.7725

print("\nWait, the model is being trained...")
mlp_reg.fit(X_train_stdsc,y_train)
y_pred = mlp_reg.predict(X_test_stdsc)
df_temp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_temp.head()

# evaluating the model
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print("The R2 Score is for the test data is", r2)

from sklearn.metrics import mean_squared_error,  mean_absolute_percentage_error
print('mean_squared_error ' ,np.sqrt(mean_squared_error(y_test,y_pred)))
print('mean_absolute_percentage_error ',mean_absolute_percentage_error(y_test,y_pred))

#Plotting loss curve(cross entropy loss, as it is a classification model)
#sklearn MLP model optimizes the log-loss function (cross entropy function)
loss_values = mlp_reg.loss_curve_
epochs = range(1, len(loss_values)+1)
import matplotlib.pyplot as plt
plt.plot(epochs, loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#scatter plot of y_test vs y_pred
label=f'House cost prediction R2 value is {r2}'
plt.scatter(y_test,y_pred, label=label)
plt.xlabel('Actual cost')
plt.ylabel('Predicted cost')
plt.legend()
plt.show()