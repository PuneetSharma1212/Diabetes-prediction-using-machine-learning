# %%
"""
# Diabetese prediction using Machine learning algorithms.
"""

# %%
"""
##IDA(Initial Data analysis)
"""

# %%
# Importing required libraries
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')
from joblib import dump, load

# %%
# Reading the dataset using Pandas library
df_diabetes = pd.read_csv('https://raw.githubusercontent.com/PuneetSharma1212/dataset/main/diabetes.csv')

# %%
# Identifying the number of rows and number of features of dataset using shape method.
df_diabetes.shape

# %%
# Printing top five rows of dataset.
df_diabetes.head()

# %%
# Printing bottom five rows of dataset.
df_diabetes.tail()

# %%
# Identifying the data types of every feature.
df_diabetes.dtypes

# %%
"""
#### Identifying the presense of null values
"""

# %%
df_diabetes.isnull().sum()

# %%
# plotting all missing values in a heatmap.
sns.heatmap(df_diabetes.isnull().T, cbar=True, cmap="vlag")
plt.xlabel("Heatmap of missing Values")

# %%
"""
The dataframe does not have any null values
"""

# %%
"""
#### Identifying the presence of 0 values in dataset
"""

# %%
(df_diabetes == 0).sum(axis=0)

# %%
"""
As from above output we can see that dataset consist several 0 values for Glucose, BloodPressure, Pregnancies, SkinThickness, Insulin and BMI. This appears to be an issue in the dataset and hense needs to be corrected. I have replaced these 0 values with the mean value of the respective column values in next step.
"""

# %%
"""
#### Handeling 0 values:
Replacing 0 values available in Glucose, BloodPressure, Pregnancies, SkinThickness, Insulin and BMI features with the mean value of all the rows in the respective column.
"""

# %%
df_diabetes_before = df_diabetes.copy()
Column_to_modify = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for column in Column_to_modify:
  df_diabetes[column] = df_diabetes[column].replace(0, np.NaN)
  mean = int(df_diabetes[column].mean(skipna = True))
  df_diabetes[column] = df_diabetes[column].replace(np.NaN, mean)
  df_diabetes.round(4)
df_diabetes.head()

# %%
df_diabetes.head()

# %%
df_diabetes = df_diabetes.astype({'Pregnancies': 'int64','Glucose': 'int64', 'BloodPressure': 'int64', 'SkinThickness': 'int64', 'Insulin': 'int64' })

# %%
df_diabetes.dtypes

# %%
# printing top five rows of dataset before replacing the zero values with mean.
df_diabetes_before.head()

# %%
# printing top five rows of dataset after replacing the zero values with mean.
df_diabetes.head()

# %%
"""
From above output we can see that the 0 values in columns 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',  have been replaced with the mean values.
"""

# %%
"""
#### Histograms to understand the skewness of values for each feature.
"""

# %%
# Plotting the distribution of all the variables
df_diabetes1 = df_diabetes.drop('Outcome', axis=1)
ax = df_diabetes1.hist(figsize = (10,10))
plt.show()

# %%
"""
#### Outliers detection using box plots
"""

# %%
# set a = 1 to increment
a=1
# set figure size
plt.figure(figsize=(20,12))
# iterate through numerical measures
for attr in df_diabetes1:
    # create subplots
    plt.subplot(3,3,a)
    # plot boxplot
    ax=sns.boxplot(x=attr, data=df_diabetes1, color='y')
    # set label
    plt.xlabel(attr, fontsize=14)
    # increment a
    a+=1
# show plot
plt.show()

# %%
"""
#### Scale the features using power transformed from of sklearn.
"""

# %%
# scale the features using PowerTransformer scaler of sklearn.
X = df_diabetes.drop('Outcome', axis=1)
pt = PowerTransformer()
# scaler = StandardScaler()
# scaler = scaler.fit(X)
# X = scaler.transform(X)
# features = pd.DataFrame(X, columns=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"])
# features

# %%
dump(pt, 'pt.joblib')

# %%
pt = pt.fit_transform(X)
X= pd.DataFrame(pt, columns=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"])

# %%
"""
#### Histogram of features after performing scaling on dataset.
"""

# %%
ax = X.hist(figsize = (10,10))

# %%
"""
## EDA(Exploratory Data Analysis) 
"""

# %%
"""
###Descriptive statistics of Dataset
"""

# %%
stats = df_diabetes.describe().T
print("Statistical mesures of the data:\n")
stats

# %%
"""
Method describe() of Pandas DataFrame is used to list statistical properties of all the features.

Statistical properties such as Count, Mean, Standard Deviation, Minimum Value, 25th Percentile, 50th Percentile (Median), 75th Percentile, Maximum Value are obtained.
"""

# %%
"""
###Identifying the number of people with and without diabetes as per dataset.
"""

# %%
# Seaborne library to plot the countplot of target variable which is 'Outcome'.
ax1 = sns.countplot(df_diabetes['Outcome'], palette='Set2')
ax1.yaxis.grid()
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.spines['bottom'].set_linewidth(True)
ax1.spines['left'].set_linewidth(True)

# %%
# The number of people with diabetes is about half of the number of those without.

# %%
# We can use the KDE plot to separately visualize the distribution of features for people with and without diabetes


ax = sns.kdeplot(df_diabetes.Glucose[(df_diabetes["Outcome"] == 0) ],
                color="Blue", shade = True)
ax = sns.kdeplot(df_diabetes.Glucose[(df_diabetes["Outcome"] == 1) ],
                ax =ax, color="Red", shade= True)
ax.legend(["Diabetes = No","Diabetes = Yes"], loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Glucose Concentration')
ax.set_title('Distribution of Glucose by diabetes status')

ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_linewidth(True)
ax.spines['left'].set_linewidth(False)


# figure size in inches
sns.set(rc={'figure.figsize':(7,5)})

# %%
# Pregnancies


ax = sns.kdeplot(df_diabetes.Pregnancies[(df_diabetes["Outcome"] == 0) ],
                color="Blue", shade = True)
ax = sns.kdeplot(df_diabetes.Pregnancies[(df_diabetes["Outcome"] == 1) ],
                ax =ax, color="Red", shade= True)
ax.legend(["Diabetes = No","Diabetes = Yes"], loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Pragenencies')
ax.set_title('Distribution of Pregnancies by diabetes status')

ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_linewidth(True)
ax.spines['left'].set_linewidth(False)


# figure size in inches
sns.set(rc={'figure.figsize':(7,5)})

# %%
# Pair Plot allows us to plot the KDE - such as the one above - for all the combination of variables
# It also yields scatter plots

ax = sns.pairplot(df_diabetes,hue= "Outcome")

# %%
# We can conclude that older age, high glucose concentration, and highe number pregnancies are associated with diabetes.
# However, the association doesn't seem to be strong.

# %%
df_diabetes.tail()

# %%
"""
Using correlation matrix to understand the correlation between variables

We can exclude the vraibales, from the model, that have strong correlation with the other variables
"""

# %%
df_corr = df_diabetes.corr()
sns.set(style="white")
mask = np.zeros_like(df_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 6))

cmap = sns.diverging_palette(255, 133, as_cmap=True)
sns.heatmap(df_corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=0.5, annot=True)

plt.yticks(rotation=0, ha="right")
plt.xticks(rotation=90, ha="center")

plt.show()

# %%
"""
As per above correlation matrix, none of the variables has a strong correlation with any other, we can't eliminate any of the variables
"""

# %%
"""
## Implementation of Machine learning algorithms


#### Five individual machine learning models developed are:

 * Logistic Regression
 * Decision Tree
 * KNN
 * Random Forest
 * AdaBoosting

#### Two ensemble machine learning models developed are:

 * Voting Classifier
"""

# %%
# We will split the data set into train and test (30% of the original dataframe)
y= df_diabetes['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# %%
# Logistic Regression
logreg = LogisticRegression(solver='newton-cg', class_weight='balanced')
logreg.fit(X_train,y_train)

# %%
y_pred_logreg = logreg.predict(X_test)
print('Logistic regression Accuracy score:')
print(accuracy_score(y_test, y_pred_logreg))
print('\n')

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_logreg))

fig1 = plot_confusion_matrix(logreg, X_test, y_test, cmap= plt.cm.Oranges)

# %%
print(classification_report(y_test, y_pred_logreg))

# %%
logreg_cv = cross_val_score(estimator = logreg, X = X_train, y = y_train, cv = 10)
print(logreg_cv)
print('The average accuracy of the 10 K-fold', logreg_cv.mean()*100)

# %%
# Decision Tree
dectree = DecisionTreeClassifier(class_weight={0: 0.35, 1: 0.65})
dectree.fit(X_train,y_train)

# %%
y_pred_dectree = dectree.predict(X_test)
print('Accuracy score of decision tree model:')
print(accuracy_score(y_test, y_pred_dectree))
print('\n')
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_dectree))

fig1 = plot_confusion_matrix(dectree, X_test, y_test, cmap= plt.cm.Oranges)

# %%
print(classification_report(y_test, y_pred_dectree))

# %%
dectree_cv = cross_val_score(estimator = dectree, X = X_train, y = y_train, cv = 10)
print(dectree_cv)
print('The average accuracy of the 10 K-fold', dectree_cv.mean()*100)

# %%
# KNN
# We will check the accuracy with different number of neighbors

neighbors = np.arange(1,15)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 
    
    
plt.title('k-NN Accuracy for different number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

# %%
# The accuracy on the test dataset is maximum with 5 neighbors

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# %%
y_pred_knn = knn.predict(X_test)
print('Accuracy')
print(accuracy_score(y_test, y_pred_knn))
print('\n')

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_knn))

fig1 = plot_confusion_matrix(knn, X_test, y_test, cmap= plt.cm.Oranges)

# %%
print(classification_report(y_test, y_pred_knn))

# %%
knn_cv = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = 10)
print(knn_cv)
print('The average accuracy of the 10 K-fold', knn_cv.mean()*100)

# %%
ranfor = RandomForestClassifier(n_estimators=300, bootstrap = True, max_features = 'sqrt')
ranfor.fit(X_train, y_train)

# %%
y_pred_ranfor = ranfor.predict(X_test)
print('Accuracy')
print(accuracy_score(y_test, y_pred_ranfor))
print('\n')
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_ranfor))

# %%
fig1 = plot_confusion_matrix(ranfor, X_test, y_test, cmap= plt.cm.Oranges)

# %%
print(classification_report(y_test, y_pred_ranfor))

# %%
ranfor_cv = cross_val_score(estimator = ranfor, X = X_train, y = y_train, cv = 10)
print(ranfor_cv)
print('The average accuracy of the 10 K-fold', ranfor_cv.mean()*100)

# %%
abc = AdaBoostClassifier(n_estimators=1000)
abc.fit(X_train, y_train)

# %%
y_pred_abc = abc.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_abc))
print('\n')
print('Accuracy')
print(accuracy_score(y_test, y_pred_abc))

# %%
fig1 = plot_confusion_matrix(abc, X_test, y_test, cmap= plt.cm.Oranges)

# %%
print(classification_report(y_test,y_pred_abc))

# %%
abc_cv = cross_val_score(estimator = abc, X = X_train, y = y_train, cv = 10)
print(abc_cv)
print('The average accuracy of the 10 K-fold', abc_cv.mean()*100)

# %%
# We have already used two ensemble methods - Random Forests (Averaging) and Adaptive Boosting (Boosting) 
# To improve accuracy, we will combine different classifiers using Voting Classifier, which is also an ensemble method. 

# %%
# Voting Classifier without weights



vc = VotingClassifier(estimators=[('logreg',logreg),('dectree',dectree),('ranfor',ranfor),('knn',knn),('abc',abc)], 
                      voting='soft')
vc.fit(X_train, y_train)

# %%
y_pred_vc = vc.predict(X_test)
print('Accuracy')
print(accuracy_score(y_test, y_pred_vc))
print('\n')
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_vc))

# %%
fig= plot_confusion_matrix(vc, X_test ,y_test,cmap= plt.cm.Oranges)

# %%
print(classification_report(y_test, y_pred_vc))

# %%
vc_cv = cross_val_score(estimator = vc, X = X_train, y = y_train, cv = 10)
print(vc_cv)
print('The average accuracy of the 10 K-fold', vc_cv.mean()*100)

# %%
# The accuracy of Voting Calssifier is more than any of the other individual classsifiers

# %%
# Now, we will use Voting classifier with weights
# We will assign more weight to the classifiers with better accuracy

# %%
# Voting Classifier with weights

vc1 = VotingClassifier(estimators=[('logreg',logreg),('dectree',dectree),('ranfor',ranfor),('knn',knn),('abc',abc)], 
                      voting='soft', weights=[2,2,2,2,2])
vc1.fit(X_train, y_train)

# %%
dump(vc1, 'vc1.joblib')

# %%
y_pred_vc1 = vc1.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_vc1))
print('\n')
print('Accuracy')
print(accuracy_score(y_test, y_pred_vc1))
fig= plot_confusion_matrix(vc1, X_test ,y_test,cmap= plt.cm.Oranges)

# %%
print(classification_report(y_test, y_pred_vc1))

# %%
vc1_cv = cross_val_score(estimator = vc1, X = X_train, y = y_train, cv = 10)
print(vc1_cv)
print('The average accuracy of the 10 K-fold', vc1_cv.mean()*100)

# %%
print('Model Accuracy')
print('\n')
print('Logistic Regression: '+str(round(accuracy_score(y_test, y_pred_logreg)*100,2))+'%')
print('Decision Tree: '+str(round(accuracy_score(y_test, y_pred_dectree)*100,2))+'%')
print('KNN: '+str(round(accuracy_score(y_test, y_pred_knn)*100,2))+'%')
print('\n')
print('Averaging Method')
print('Random Forest: '+str(round(accuracy_score(y_test, y_pred_ranfor)*100,2))+'%')
print('\n')
print('Boosting Method')
print('AdaBoost: '+str(round(accuracy_score(y_test, y_pred_abc)*100,2))+'%')
print('\n')
print('Voting Classifiers')
print('Voting Classifier without Weights: '+str(round(accuracy_score(y_test, y_pred_vc)*100,2))+'%')
print('Voting Classifier with Weights: '+str(round(accuracy_score(y_test, y_pred_vc1)*100,2))+'%')

# %%
"""
To Conclude, ensemble machine learning model developed withough weights have shown highest accuracy and thus this will be integrated with the developed web application.
"""