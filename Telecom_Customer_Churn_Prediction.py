# -*- coding: utf-8 -*-
# Auto-converted from Jupyter Notebook (.ipynb)
# Source: Telecom Customer Churn Prediction.ipynb

# %% [markdown] (cell 1)
# # Telecom Customer Churn Prediction

# %% [markdown] (cell 2)
# The aim of this project is to analyze customer demographics, services, tenure and other variables to predict whether a particular customer will churn or not. 
# 
# ### Data Dictionary
# 
# | Variable | Description |
# | --- | --- |
# |CustomerID| Unique customer ID|
# |Gender|Customer's gender|
# |SeniorCitizen|Whether the customer is a senior citizen or not (1, 0)|
# |Partner|Whether the customer has a partner or not (Yes, No)|
# |Dependents|Whether the customer has dependents or not (Yes, No)|
# |Tenure|Number of months the customer has stayed with the company|
# |PhoneService|Whether the customer has a phone service or not (Yes, No)|
# |MultipleLines|Whether the customer has multiple lines or not (Yes, No, No phone service)|
# |InternetService|Customer’s internet service provider (DSL, Fiber optic, No)|
# |OnlineSecurity|Whether the customer has online security or not (Yes, No, No internet service)|
# |OnlineBackup|Whether the customer has online backup or not (Yes, No, No internet service)|
# |DeviceProtection|Whether the customer has device protection or not (Yes, No, No internet service)|
# |TechSupport|Whether the customer has tech support or not (Yes, No, No internet service)|
# |StreamingTV|Whether the customer has streaming TV or not (Yes, No, No internet service)|
# |StreamingMovies|Whether the customer has streaming movies or not (Yes, No, No internet service)|
# |Contract|The contract term of the customer (Month-to-month, One year, Two year)|
# |PaperlessBilling|Whether the customer has paperless billing or not (Yes, No)|
# |PaymentMethod|The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))|
# |MonthlyCharges|The amount charged to the customer monthly|
# |TotalCharges|The total amount charged to the customer|
# |Churn|Whether the customer churned or not (Yes or No)|

# %% (cell 3)
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% (cell 4)
#Loading the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()

# %% [markdown] (cell 5)
# ## Data Preprocessing Part 1 

# %% (cell 6)
#Checking the shape of the dataset
df.shape

# %% (cell 7)
#Removing the customerID column
df.drop(columns='customerID', inplace=True)

# %% (cell 8)
#Checking for duplicate values
df.duplicated().sum()

# %% (cell 9)
#Removing the duplicate values
df.drop_duplicates(inplace=True)

# %% (cell 10)
#Checking the datatypes of the columns
df.dtypes

# %% [markdown] (cell 11)
# Type casting column Total Charges

# %% (cell 12)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# %% (cell 13)
#Checking for null values
df.isnull().sum()

# %% (cell 14)
#Droping the null values
df.dropna(inplace=True)

# %% (cell 15)
#checking number of unique values in each column
df.nunique()

# %% (cell 16)
#Checking the unique values in each column
cols = df.columns
for i in cols:
    print(i, df[i].unique(), '\n')

# %% [markdown] (cell 17)
# Descriptive Statistics

# %% (cell 18)
df.describe()

# %% (cell 19)
df.head()

# %% [markdown] (cell 20)
# ## Exploratory Data Analysis

# %% [markdown] (cell 21)
# In the exploratory data analysis, I will be visualizing the data to get a better understanding of the data and to see if there are any trends or pattern in the data. First I will begin with looking at the distribution of the data and then I will look at the relationship between the independent variables and the target variable.

# %% [markdown] (cell 22)
# ### Customer Demographics

# %% (cell 23)
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

#Gender Distribution
ax[0,0].pie(df['gender'].value_counts(), labels  = ['Male', 'Female'], autopct = '%1.2f%%')
ax[0,0].set_title('Gender Distribution')

#Senior Citizen Distribution
sns.barplot(y = df['SeniorCitizen'].value_counts(), x = df['SeniorCitizen'].unique(), ax=ax[0,1]).set_title('Senior Citizen')

#Partner Distribution
sns.barplot(y = df['Partner'].value_counts(), x = df['Partner'].unique(), ax=ax[1,0]).set_title('Partner')

#Dependents Distribution
sns.barplot(y = df['Dependents'].value_counts(), x = df['Dependents'].unique(), ax=ax[1,1]).set_title('Dependents')

# %% [markdown] (cell 24)
# These graphs shows the customer demographics. The number of males and females is almost same, with few more males than females in the dataset. Majority of them are not senior citizen. Nearly 3500, customers have a partner and similar number of cutomers don't. Majority of the customers don't have dependents, but still a significant number does have dependents. 
# 
# From these graphs, we get know about the customers demographics, which help us to get an idea of their psychology based on their age, relationship status, and dependents.

# %% [markdown] (cell 25)
# ### Services

# %% (cell 26)
fig, ax = plt.subplots(3, 3, figsize=(20, 20))
#phone service
sns.countplot(x = df['PhoneService'], ax=ax[0,0]).set_title('Phone Service')
ax[0,0].set_title('Phone Service')
#Multiple Lines
sns.countplot(x = df['MultipleLines'], ax=ax[0,1]).set_title('Multiple Lines')
ax[0,1].set_title('Multiple Lines')
#Internet Service
sns.countplot(x = df['InternetService'], ax=ax[0,2]).set_title('Internet Service')
ax[0,2].set_title('Internet Service')
#Online Security
sns.countplot(x = df['OnlineSecurity'], ax=ax[1,0]).set_title('Online Security')
ax[1,0].set_title('Online Security')
#Online Backup
sns.countplot(x = df['OnlineBackup'], ax=ax[1,1]).set_title('Online Backup')
ax[1,1].set_title('Online Backup')
#Device Protection
sns.countplot(x = df['DeviceProtection'], ax=ax[1,2]).set_title('Device Protection')
ax[1,2].set_title('Device Protection')
#Tech Support
sns.countplot(x = df['TechSupport'], ax=ax[2,0]).set_title('Tech Support')
ax[2,0].set_title('Tech Support')
#Streaming TV
sns.countplot(x = df['StreamingTV'], ax=ax[2,1]).set_title('Streaming TV')
ax[2,1].set_title('Streaming TV')
#Streaming Movies
sns.countplot(x = df['StreamingMovies'], ax=ax[2,2]).set_title('Streaming Movies')
ax[2,2].set_title('Streaming Movies')

# %% [markdown] (cell 27)
# The above graphs visualizes the services taken by the customers from the telecom company. Nearly 6000 customers have taken phone service. However, nealry half of the cutomers have taken multiplt lines from the company. Almost 5500, have taken internet services from the company, where 3000 customers opted fibre optcs and rest of them opted DSL  which could possible for business purposes. From these three major services related to telecom, the phone services and the internet services are the most popular services among the customers.
# 
# Coming to other services which includes- Online Security, Online Backup, Device Protection, Tech Support, and Streaming Services. The online backup and device protection service is opted by almost 2500 customers, which higlights the customers concern regarding their device safety and data protection. The online security and tech support is opted by almost 2000 customers which are least opted services among the customers. The streaming services are the most popular services, with more than 2500 customers opting for it.
# 
# From this,I conclude that part from the internet and phone services, the streaming services are most opted ones. Therefore, the company should focus on providing better streaming services to the customers.

# %% [markdown] (cell 28)
# ### Tenure and Contract

# %% (cell 29)
fig, ax = plt.subplots(1, 2, figsize=(15, 8))
sns.histplot(x = 'tenure', data = df, ax= ax[0]).set_title('Cutomer Tenure in Months')
sns.countplot(x = 'Contract', data = df, ax= ax[1]).set_title('Contract Type')

# %% [markdown] (cell 30)
# In the aboove graphs we can see the distribution of customer tenure with the comapny and the count of the type of contarct the company had with customer. Here, most if the customers had tenure less than a month, and most of the customers had a month-to-month contract with the company. Therfore, the customers with shorter tenure have month-to-month contract with the company. In addtion to that, a significant number of customers have tenure of nealry 70 months, which highlights the loyalty of the customers towards the company. Moreover, after month-to-month contract, the second most popular contract is two-year contract, which is opted by almost 1700 customers. Rest of the customers have tenure between 1-5 years.

# %% [markdown] (cell 31)
# ### Billing and Charges

# %% (cell 32)
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.tight_layout(pad=5.0)

#spacing between subplots
fig.subplots_adjust(hspace=0.9)


#papaerless billing
sns.countplot(x = df['PaperlessBilling'], ax=ax[0,0]).set_title('Paperless Billing')

#Payment Method
sns.countplot(x = df['PaymentMethod'], ax=ax[0,1]).set_title('Payment Method')
ax[0,1].xaxis.set_tick_params(rotation=90)

#Monthly Charges
sns.histplot(x = 'MonthlyCharges', data = df, ax = ax[1,0]).set_title('Monthly Charges')

#Total Charges
sns.histplot(x = 'TotalCharges', data = df, ax = ax[1,1]).set_title('Total Charges')

# %% [markdown] (cell 33)
# These graphs shows the method of billing and the bill amounts. Most of the customers, nearly 4000 prefer paperless billing, however, a little bit over half of them pays through electronic check. But still a significant number of customers prefer paper bills. Apart from electronic checks, the other modes of payment accepted by the company includes -  mailed checks, bank transfer and credit cards. Nearly 4500 customers altogether prefer these modes of payment.
# 
# Now, for the montly charges, huge number of customers pays near 20 dollars for the montly services and majority of the customer having total charges less than 200 dollars. However, there are considerable number of customers having monthly charges between 70 to 100 dollars and total charges between 200-800 dollars. Interestingly, If we look at the total charges graph, we can see that some customers have a total bill more than 4000 and even 8000 as well. This could be possible, if the customer has a long tenure or uses alot of services.
# 
# Now, I conclude that company mainly have customers with low charges, which means comany should focus on these customers by providing even more afforadable services.

# %% [markdown] (cell 34)
# ### Churn Count

# %% (cell 35)
plt.pie(x = df['Churn'].value_counts(), labels = df['Churn'].unique(), autopct = '%1.2f%%')
plt.title('Churn Count')

# %% [markdown] (cell 36)
# In the dataset, the number of churning customers is very less as compared to non churning. Only 26.49% churnned from the telecom company. This could be a potential proof, that company is quite good at retaning its customers.

# %% [markdown] (cell 37)
# ##### Till now, I have visualized the data and got a better understanding of the data. Now, I will look at the relationship between the independent variables and the target variable.

# %% [markdown] (cell 38)
# ### Customer Demogrpahics and Churn

# %% (cell 39)
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

#Gender Distribution
sns.countplot(x = 'gender', data = df, hue = 'Churn', ax=ax[0,0])
ax[0,0].set_title('Gender Distribution')

#Senior Citizen Distribution
sns.countplot(x = df['SeniorCitizen'], ax=ax[0,1], hue = df['Churn']).set_title('Senior Citizen and Churn')

#Partner Distribution
sns.countplot( x = df['Partner'], ax=ax[1,0], hue = df['Churn']).set_title('Partner and Churn')

#Dependents Distribution
sns.countplot(x = df['Dependents'], ax=ax[1,1], hue = df['Churn']).set_title('Dependents and Churn')

# %% [markdown] (cell 40)
# From these graphs, we can get know about the relation between customer demographics and customer churn. Both makes and females have equal number of churn count, so there is not relation between gender and customer churn. However, the senior citizens have a lesser churn count as compared to non senior citizens, which may be because their age and they don't want to hasle with the process of changing the telecom company. The customers with no partners have higher churn count as compared to customers with partners. Similarly, customers with no dependents have higher churn count as compared to customers with dependents.
# 
# From this I conclude that customers whom are single with no partner or have no dependents have higher churn count and senior citizens have lower churn count.

# %% [markdown] (cell 41)
# ### Services and Churn

# %% (cell 42)
fig, ax = plt.subplots(3, 3, figsize=(20, 20))
#phone service
sns.countplot(x = df['PhoneService'], ax=ax[0,0], hue = df['Churn']).set_title('Phone Service')
ax[0,0].set_title('Phone Service')
#Multiple Lines
sns.countplot(x = df['MultipleLines'], ax=ax[0,1], hue = df['Churn']).set_title('Multiple Lines')
ax[0,1].set_title('Multiple Lines')
#Internet Service
sns.countplot(x = df['InternetService'], ax=ax[0,2], hue = df['Churn']).set_title('Internet Service')
ax[0,2].set_title('Internet Service')
#Online Security
sns.countplot(x = df['OnlineSecurity'], ax=ax[1,0], hue = df['Churn']).set_title('Online Security')
ax[1,0].set_title('Online Security')
#Online Backup
sns.countplot(x = df['OnlineBackup'], ax=ax[1,1], hue = df['Churn']).set_title('Online Backup')
ax[1,1].set_title('Online Backup')
#Device Protection
sns.countplot(x = df['DeviceProtection'], ax=ax[1,2], hue = df['Churn']).set_title('Device Protection')
ax[1,2].set_title('Device Protection')
#Tech Support
sns.countplot(x = df['TechSupport'], ax=ax[2,0], hue = df['Churn']).set_title('Tech Support')
ax[2,0].set_title('Tech Support')
#Streaming TV
sns.countplot(x = df['StreamingTV'], ax=ax[2,1], hue = df['Churn']).set_title('Streaming TV')
ax[2,1].set_title('Streaming TV')
#Streaming Movies
sns.countplot(x = df['StreamingMovies'], ax=ax[2,2], hue = df['Churn']).set_title('Streaming Movies')
ax[2,2].set_title('Streaming Movies')

# %% [markdown] (cell 43)
# These graphs visualizes the relation between customer churn based on services opted by the customer. In the phone and internet service, there is no relation between churn and service opted, however the churn count is higher for the customers, who have taken multiple lines. Coming to other services, where customers who have not taken Online backup or Device Protection service has higher churn count, than those who have opted. Moreover, the customers with streaming services have lower churn count as compared to those who have not opted for it.
# 
# Therefore, certain services have relation with the customer churn, which are multiple lines, Online Backup, Device Protection, and Streaming Services.

# %% [markdown] (cell 44)
# ### Tenure/Contract and Churn

# %% (cell 45)
fig, ax = plt.subplots(1, 2, figsize=(15, 8))
sns.histplot(x = 'tenure', data = df, ax= ax[0], hue = 'Churn', multiple = 'stack').set_title('Cutomer Tenure (in Months) and Churn')
sns.countplot(x = 'Contract', data = df, ax= ax[1], hue = 'Churn').set_title('Contract Type and Churn')

# %% [markdown] (cell 46)
# Looks like the customer tenure and contract has a inverse relation. The customers with shorter tenure or tenure less than 5 months have higher churn count. The churn count decreases with increase in tenure. Moreover, the customers with month-to-month contract have higher churn count as compared to those with one or two year contract which also proves that customer who have longer contract with the company have lower churn count.

# %% [markdown] (cell 47)
# ### Billing/Charges and Churn

# %% (cell 48)
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.tight_layout(pad=5.0)

#spacing between subplots
fig.subplots_adjust(hspace=0.9)


#papaerless billing
sns.countplot(x = df['PaperlessBilling'], ax=ax[0,0], hue = df['Churn']).set_title('Paperless Billing')

#Payment Method
sns.countplot(x = df['PaymentMethod'], ax=ax[0,1], hue = df['Churn']).set_title('Payment Method')
ax[0,1].xaxis.set_tick_params(rotation=90)

#Monthly Charges
sns.histplot(x = 'MonthlyCharges', data = df, ax = ax[1,0], hue = 'Churn', multiple= 'stack').set_title('Monthly Charges')

#Total Charges
sns.histplot(x = 'TotalCharges', data = df, ax = ax[1,1], hue = 'Churn', multiple= 'stack').set_title('Total Charges')

# %% [markdown] (cell 49)
# The paperless billing and payment method have not significant relation with the customer churn. However, the montly and total charges do have a interesting relation with the customer churn. The customers with higher monthly charges have higher churn count, which is quite obvious. But, the customers with higher total charges have lower churn count, which is quite interesting. This could be possible, if the customer has a long tenure or uses alot of services. Therefore, the company should focus on lowering the monthly charges for the customers in order to reduce the churn count.

# %% [markdown] (cell 50)
# ## Data Preprocessing Part 2

# %% [markdown] (cell 51)
# ### Outlier Removal

# %% (cell 52)
#Columns for outlier removal
cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

#Using IQR method to remove outliers
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1

#Removing the outliers
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# %% [markdown] (cell 53)
# ### Label Encoding

# %% (cell 54)
from sklearn.preprocessing import LabelEncoder

#colums for label encoding
cols = df.columns[df.dtypes == 'object']

#Label encoder object
le = LabelEncoder()

#Label encoding the columns
for i in cols:
    le.fit(df[i])
    df[i] = le.transform(df[i])
    print(i, df[i].unique(), '\n')

# %% [markdown] (cell 55)
# ### Feature Scaling

# %% (cell 56)
from sklearn.preprocessing import StandardScaler

#Standardizing the data
sc = StandardScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = sc.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

# %% [markdown] (cell 57)
# ## Correlation Matrix Heatmap

# %% (cell 58)
plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True)

# %% [markdown] (cell 59)
# ## Train Test Split

# %% (cell 60)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='Churn'), df['Churn'], test_size=0.2, random_state=42)

# %% [markdown] (cell 61)
# ## Model Building

# %% [markdown] (cell 62)
# I will be using the following models to predict the customer churn:
# 1. Decision Tree Classifier
# 2. Random Forest Classifier
# 3. K Nearest Neighbors Classifier

# %% [markdown] (cell 63)
# ### Decision Tree Classifier

# %% (cell 64)
from sklearn.tree import DecisionTreeClassifier

#Decision Tree Classifier Object
dtree  = DecisionTreeClassifier()

# %% [markdown] (cell 65)
# #### Hyperparameter Tuning using GridSearchCV

# %% (cell 66)
from sklearn.model_selection import GridSearchCV

#parameter grid
param_grid = {
    'max_depth': [2,4,6,8,10],
    'min_samples_leaf': [2,4,6,8,10],
    'min_samples_split': [2,4,6,8,10],
    'criterion': ['gini', 'entropy'],
    'random_state': [0,42]
}

#Grid Search Object with Decision Tree Classifier
grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2, scoring='accuracy')

#Fitting the data
grid_search.fit(X_train, y_train)

#Best parameters
print(grid_search.best_params_)

# %% (cell 67)
#Decision Tree Classifier Object with best parameters
dtree  = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=2, min_samples_split=10, random_state=42)

#Fitting the data
dtree.fit(X_train, y_train)

#Training accuracy
print('Training Accuracy: ', dtree.score(X_train, y_train))

#Predicting the values
d_pred = dtree.predict(X_test)

# %% [markdown] (cell 68)
# ### Random Forest Classifier

# %% (cell 69)
from sklearn.ensemble import RandomForestClassifier

#Random Forest Classifier Object
rfc = RandomForestClassifier()

# %% [markdown] (cell 70)
# #### Hyperparameter Tuning using GridSearchCV

# %% (cell 71)
from sklearn.model_selection import GridSearchCV

#parameter grid
param_grid = {
    'max_depth': [2,4,6,8,10],
    'min_samples_leaf': [2,4,6,8,10],
    'min_samples_split': [2,4,6,8,10],
    'criterion': ['gini', 'entropy'],
    'random_state': [0,42]
}

#Grid Search Object with Random Forest Classifier
grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2, scoring='accuracy')

#Fitting the data
grid_search.fit(X_train, y_train)

#Best parameters
print(grid_search.best_params_)

# %% (cell 72)
#Random Forest Classifier Object with best parameters
rfc = RandomForestClassifier(criterion='entropy', max_depth=10, min_samples_leaf=10, min_samples_split=2, random_state=42)

#Fitting the data
rfc.fit(X_train, y_train)

#Training accuracy
print('Training Accuracy: ', rfc.score(X_train, y_train))

#Predicting the values
r_pred = rfc.predict(X_test)

# %% [markdown] (cell 73)
# ### K Nearest Neighbors Classifier

# %% (cell 74)
from sklearn.neighbors import KNeighborsClassifier

#KNN Classifier Object
knn = KNeighborsClassifier()

# %% [markdown] (cell 75)
# #### Hyperparameter Tuning using GridSearchCV

# %% (cell 76)
from sklearn.model_selection import GridSearchCV

#parameter grid
param_grid = {
    'n_neighbors': [2,4,6,8,10],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

#Grid Search Object with KNN Classifier
grid_search = GridSearchCV(estimator = knn, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2, scoring='accuracy')

#Fitting the data
grid_search.fit(X_train, y_train)

#Best parameters
print(grid_search.best_params_)

# %% (cell 77)
#KNN Classifier Object with best parameters
knn = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=6, weights='uniform')

#Fitting the data
knn.fit(X_train, y_train)

#Training accuracy
print('Training Accuracy: ', knn.score(X_train, y_train))

#Predicting the values
k_pred = knn.predict(X_test)

# %% [markdown] (cell 78)
# ## Model Evaluation

# %% [markdown] (cell 79)
# ### Confusion Matrix Heatmap

# %% (cell 80)
from sklearn.metrics import confusion_matrix

fig, ax = plt.subplots(1, 3, figsize=(20, 5))

#Decision Tree Confusion Matrix
sns.heatmap(confusion_matrix(y_test, d_pred), annot=True, ax=ax[0], cmap='coolwarm').set_title('Decision Tree Confusion Matrix')

#Random Forest Confusion Matrix
sns.heatmap(confusion_matrix(y_test, r_pred), annot=True, ax=ax[1], cmap='coolwarm').set_title('Random Forest Confusion Matrix')

#KNN Confusion Matrix
sns.heatmap(confusion_matrix(y_test, k_pred), annot=True, ax=ax[2], cmap='coolwarm').set_title('KNN Confusion Matrix')

# %% [markdown] (cell 81)
# The confusion matrix heatmaps visulaizes the true positive and true negative results from the machine learning model. Here, in the above confusion matrix, when see that the Random Forest Classifier has the highest true positive and true negative results, with considerbly less false positive and false negative results. Therefore, the Random Forest Classifier is the best model for predicting the customer churn.

# %% [markdown] (cell 82)
# ### Distribution Plot

# %% (cell 83)
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

#Decision Tree 
sns.distplot(y_test, hist=False, color="r", label="Actual Value", ax=ax[0]).set_title('Decision Tree')
sns.distplot(d_pred, hist=False, color="b", label="Fitted Values" , ax=ax[0])

#Random Forest
sns.distplot(y_test, hist=False, color="r", label="Actual Value", ax=ax[1]).set_title('Random Forest')
sns.distplot(r_pred, hist=False, color="b", label="Fitted Values" , ax=ax[1])

#KNN
sns.distplot(y_test, hist=False, color="r", label="Actual Value", ax=ax[2]).set_title('KNN')
sns.distplot(k_pred, hist=False, color="b", label="Fitted Values" , ax=ax[2])

# %% [markdown] (cell 84)
# ### Classification Report

# %% (cell 85)
from sklearn.metrics import classification_report

print('Decision Tree Classification Report: \n', classification_report(y_test, d_pred))

print('Random Forest Classification Report: \n', classification_report(y_test, r_pred))

print('KNN Classification Report: \n', classification_report(y_test, k_pred))

# %% [markdown] (cell 86)
# ### Model Metrics

# %% (cell 87)
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, f1_score

#Bar plots
fig, ax = plt.subplots(2,2, figsize=(20, 10))

#Accuracy Score
sns.barplot(x = ['Decision Tree', 'Random Forest', 'KNN'], y = [accuracy_score(y_test, d_pred), accuracy_score(y_test, r_pred), accuracy_score(y_test, k_pred)], ax=ax[0,0]).set_title('Accuracy Score')

#Mean Squared Error
sns.barplot(x = ['Decision Tree', 'Random Forest', 'KNN'], y = [mean_squared_error(y_test, d_pred), mean_squared_error(y_test, r_pred), mean_squared_error(y_test, k_pred)], ax=ax[0,1]).set_title('Mean Squared Error')

#Mean Absolute Error
sns.barplot(x = ['Decision Tree', 'Random Forest', 'KNN'], y = [mean_absolute_error(y_test, d_pred), mean_absolute_error(y_test, r_pred), mean_absolute_error(y_test, k_pred)], ax=ax[1,0]).set_title('Mean Absolute Error')

#F1 Score
sns.barplot(x = ['Decision Tree', 'Random Forest', 'KNN'], y = [f1_score(y_test, d_pred), f1_score(y_test, r_pred), f1_score(y_test, k_pred)], ax=ax[1,1]).set_title('F1 Score')

# %% [markdown] (cell 88)
# These graphs compares the models based on the metrics such as accuracy score, mean squared error, mean absolute error and f1 score. The Random Forest Classifier has the highest accuracy score and F1 Score, and lowest mean squared error, mean absolute error. Therefore, the Random Forest Classifier is a good fit for predicting the customer churn.

# %% [markdown] (cell 89)
# ## Feature Importance

# %% (cell 90)
fig, ax = plt.subplots(1, 2, figsize=(20, 8))
# Decision Tree Classifier Feature Importance
feature_df = pd.DataFrame({'Features': X_train.columns, 'Importance': dtree.feature_importances_})
feature_df.sort_values('Importance', ascending=False, inplace=True)
sns.barplot(x = 'Importance', y = 'Features', data = feature_df, ax=ax[0]).set_title('Decision Tree Classifier Feature Importance')

# Random Forest Classifier Feature Importance
feature_df = pd.DataFrame({'Features': X_train.columns, 'Importance': rfc.feature_importances_})
feature_df.sort_values('Importance', ascending=False, inplace=True)
sns.barplot(x = 'Importance', y = 'Features', data = feature_df, ax=ax[1]).set_title('Random Forest Classifier Feature Importance')

# %% [markdown] (cell 91)
# From both the models, it is clear that the tenure, monthly charges, and total charges are the most important features for predicting the customer churn. Therefore, the company should focus on these features to reduce the customer churn.

# %% [markdown] (cell 92)
# ## Conclusion

# %% [markdown] (cell 93)
# From the exploratory data analysis, I came to know that, the senior citizens have lower churn count whereas the customers who are single or don't have dependents ahve higher churn count. In addition to that, customers are more satified with the streaming services than other services such as Online backup and Device protection, which has resulted in lower churn count in customer with streaing services than the other services.
# 
# The tenure have an inverse relation with churn count, where customer with tenure shorter than 5 months have higher churn count. Moreover, the customers with month-to-month contract have higher churn count as compared to those with one or two year contract which also proves that customer who have longer contract with the company have lower churn count.
# 
# It has been observed that the customers with higher monthly charges and lower total charges have higher churn count. Therefore, the company should focus on lowering the monthly charges for the customers in order to reduce the churn count. From the feature importance, it is clear that the tenure, contract, monthly charges, and total charges are the most important features for predicting the customer churn. Therefore, the company should focus on these features to reduce the customer churn.
# 
# Coming to the machine learning models, I have used three models - Decision Tree Classifier, Random Forest Classifier, and K Nearest Neighbors Classifier. The Random Forest Classifier has the highest accuracy i.e. 82% and F1 Score, and lowest mean squared error, mean absolute error. Therefore, the Random Forest Classifier is a good fit for predicting the customer churn.
