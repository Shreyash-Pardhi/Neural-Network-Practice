from sklearn.neural_network import MLPClassifier
#Import scikit-learn dataset library
from sklearn import datasets

#Load dataset
winedata = datasets.load_wine()

#printing class distribution of wine dataset
import numpy as np
print(f'Classes: {np.unique(winedata.target)}')
print(f'Class distribution of the entire dataset {np.bincount(winedata.target)}')

#from sklearn.cross_validation import train_test_split(Hold out method)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(winedata.data, winedata.target, test_size=.25,stratify=winedata.target)

#printing class distribution of train and test dataset
print(f'Class distribution for test data: {np.bincount(y_train)}')
print(f'Class distribution for test data: {np.bincount(y_test)}')

##-----------Understanding the dataset---------------------
# Print the information contained within the dataset
print(winedata.keys(),"\n")
#print(winedata.DESCR) #gives exhaustive information of dadaset

#Print the feature names
count=0
print("Feature names:")
for f in winedata.feature_names:
 count+=1
 print(count,"-",f)

#Print the classes
print(f'Class names: {winedata.target_names}\n')

#Printing the Initial Five Rows of data
print(winedata.data[0:5], "\n")

#Print the class values of first 5 datapoints
print(winedata.target[0:5], "\n")

#Print the dimensions of data
print(f'Shape of dataset:{winedata.data.shape}\n')


##MLP is sensitive to feature scaling, hence performing scaling
#Options: MinmaxScaler and Standardscaler
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = StandardScaler()
#scaler = MinMaxScaler()

X_train_stdsc = scaler.fit_transform(X_train)
X_test_stdsc = scaler.fit_transform(X_test)

#Setting of hyperparameters of the network
# visit this link for details of hyper parameters https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

#----------- creating initial model --------------
mlp = MLPClassifier(hidden_layer_sizes=(2,),max_iter=1000,verbose=1)
#mlp = MLPClassifier(hidden_layer_sizes=(10,),batch_size=4,solver='adam', learning_rate='0.1',momentum=0.8,activation='logistic',max_iter=1000,verbose=1)
# You do the above settings to fintune the model as explained in the class

#Calculating Training Time : more neurons, more time
import time
t1=time.time()

#Train the model using the scaled training sets
mlp.fit(X_train_stdsc, y_train)

t2=time.time()
print("Training Time:",t2-t1)

#Predict the response for test dataset
y_pred = mlp.predict(X_test_stdsc) #scaled

#Import scikit-learn metrics module for evaluating model performance
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# Model Accuracy, how often is the classifier correct?
print("\nAccuracy:",accuracy_score(y_test, y_pred))

#printing class distribution of test dataset
print(f'Classes: {np.unique(y_test)}')
print(f'Class distribution for test data{np.bincount(y_test)}')

#display the confusion matrix
print('\nConfusion Matrix is:\n',confusion_matrix(y_test, y_pred))
print('\nClassification Report:\n',classification_report(y_test, y_pred))

#Plotting loss curve(cross entropy loss, as it is a classification model)
#sklearn MLP model optimizes the log-loss function (cross entropy function)
loss_values = mlp.loss_curve_
epochs = range(1, len(loss_values)+1)
import matplotlib.pyplot as plt
plt.plot(epochs, loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#saving the MLP model for future use
#There are two packages, pickle and joblib that can be used to save and load scikit-learn models.
import joblib
filename = "mlp_wine.joblib"

# save model
joblib.dump(mlp, filename)

# you can use saved model to compute accuracy by lading it
# load model
loaded_model = joblib.load(filename)

# Predict the response for test dataset
y_pred =loaded_model.predict(X_test_stdsc)  # scaled
print("\nAccuracy from the saved model:\n",accuracy_score(y_test, y_pred))