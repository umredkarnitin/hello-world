#1. Install the required libraries
# %matplotlib notebook
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils as ut
     
from sklearn.cross_validation import train_test_split     
from bokeh.layouts import column
from mpmath import frac
from sklearn.model_selection import train_test_split


# 2. Download and import data into Python
#  Read data from web
import os
# from Trial import hbat2
os.getcwd()
os.chdir('F:\Data Scientist\Project')


# Importing HBAT SPSS Dataset
# from rpy2.robjects import pandas2ri, r
# filename = 'HBAT_200.sav'
# w = r('foreign::read.spss("%s", to.data.frame=TRUE)' % filename)
# hbat = pandas2ri.ri2py(w)
# print(hbat.head())
# print(hbat.columns)

# import pandas.rpy.common as com
# filename = "HBAT_200.sav"
# w = com.robj.r('foreign::read.spss("%s", to.data.frame=TRUE)' % filename)
# hbat = com.convert_robj(w)
# hbat.head()

# Importing HBAT csv. Dataset
hbat= pd.read_csv('MT_HBAT.csv')
print(hbat.head(5))

# Renaming the columns for better feature/Attributes visibility
hbat= hbat.rename(columns={'x1':'X1_Customer_type','x2':'X2_Industry_Type','x3':'X3_Firm_Size','x4':'X4_Region','x5':'X5_Distribution_System','x6':'X6_Product_Quality','x7':'X7_E_Commerce','x8':'X8_Technical_Support','x9':'X9_Complaint_Resolution','x10':'X10_Advertising','x11':'X11_Product_Line','x12':'X12_Salesforce_Image','x13':'X13_Competitive_Pricing','x14':'X14_Warranty_Claims','x15':'X15_New_Products','x16':'X16_Order_Billing','x17':'X17_Price_Flexibility','x18':'X18_Delivery_Speed','x19':'X19_Satisfaction','x20':'X20_Likely_to_Recommend','x21':'X21_Likely_to_Purchase','x22':'X22_Purchase_Level','x23':'X23_Consider_Strategic_Alliance'})
#Drop Variables
hbat2 = hbat.drop(['id','Unnamed: 0'], axis = 1)

#Get column names
print(hbat.columns)

#Structure of a DataFrame
print(hbat2.info())

#Select Variables
print(hbat[['X1_Customer_type', 'X2_Industry_Type']])

#Column Index starts from 0. Hence, 1 refers to second column.
print(hbat.iloc[: , 1])


#Structure of a new DataFrame
print(hbat2.info())

print(hbat2.columns)
print(hbat2.info())

#Get number of rows
print(hbat2.shape[0])

#Get number of columns
print(hbat2.shape[1])

#Get random 3 rows from dataframe
print(hbat2.sample(n=2))

#Get random 80% rows
print(hbat2.sample(frac=.8))

#Check Missing Values
hbat2.isnull()

# Number of Missing Values
# We can write a simple loop to figure out the number of blank values in 
# all variables in a dataset.
for i in list(hbat2.columns) :
    k = sum(pd.isnull(hbat2[i]))
    print(i, k)

#Summarize all numeric/continuous features from new dataframe defined in above statement
print(hbat2.describe())
print(hbat2.describe(include=['float64'])) 

#To summarise all the character variables, you can use the following script.
print(hbat2.describe(include=[object]))

#Group By : Summary by Grouping Variable
for i in list(hbat2.loc[:, hbat2.dtypes == object].columns.values) :
    for j in list(hbat2.loc[:, hbat2.dtypes == 'float64'].columns.values) :
#         print(hbat2.groupby(hbat2[i]).mean())
        print(hbat2[j].groupby(hbat2[i]).mean())
        
#Define Categorical Variable & transform them into Dummy Variable for Linear Modeling purpose
ut.myCategoryToDummyFunc(hbat2)


# You can also choose the plot kind by using the DataFrame.plot.kind methods instead of providing the kind keyword argument.
# kind :
# 'line' : line plot (default)
# 'bar' : vertical bar plot
# 'barh' : horizontal bar plot
# 'hist' : histogram
# 'box' : boxplot
# 'kde' : Kernel Density Estimation plot
# 'density' : same as 'kde'
# 'area' : area plot
# 'pie' : pie plot
# 'scatter' : scatter plot
# 'hexbin' : hexbin plot

# # create a scatter plot of columns 'A' and 'C', with changing color (c) and size (s) based on column 'B'



# Categorical variable Analysis
# It is important to check the frequency distribution of categorical variable. It helps to answer the question whether data is skewed.
# Frequency Distribution
# Summarize
ut.myFrequencyDistribution(hbat2)

#Generate Histogram
ut.myContinousPlot(hbat2)
ut.myContinousHistPlot(hbat2)

#BoxPlot
Dependent_Continous_Variable = ['X19_Satisfaction','X20_Likely_to_Recommend','X21_Likely_to_Purchase','X22_Purchase_Level']
ut.myContinousBoxPlot(hbat2, Dependent_Continous_Variable)

# Vio Plot
ut.myVioplotPlot(hbat2, Dependent_Continous_Variable)
 
 
# # Scatter Plot
# ut.myContinousJointPlot(hbat2)
# ut.myContinousScatterPlot(hbat2)

#Scatter Joint Plot
# create a scatter plot of columns 'A' and 'C', with changing color (c) and size (s) based on column 'B'
Study_Dependent_Variable= ['X19_Satisfaction']
ut.myPairWiseScatterColorPlot(hbat2, Dependent_Continous_Variable, Study_Dependent_Variable) 


#Scatter Joint Plot
# ut.myContinousJointContourPlot(hbat2)
ut.myPairWiseScatterPlot(hbat2)
ut.myScatterMatrixPlot(hbat2)

# Correlation Matrix

ut.myCorrelationPlot(hbat2, Dependent_Continous_Variable )



# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn import datasets
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.dummy import DummyRegressor
# from numpy import random, column_stack
# from statsmodels.api import add_constant, OLS
# 
# X = hbat2.iloc[:,[6,7,8,9,10,11,12,13,14,15,16,17,18]]
# y = hbat2.iloc[:,19]
# 
# print(X.head(5))
# print(y.head(5))
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# lm = LinearRegression().fit(X_train, y_train)
# lm_dummy_mean = DummyRegressor(strategy = 'mean').fit(X_train, y_train)
# 
# y_predict = lm.predict(X_test)
# y_predict_dummy_mean = lm_dummy_mean.predict(X_test)
# 
# print('Linear model, coefficients: ', lm.coef_)
# print("Mean squared error (dummy): {:.2f}".format(mean_squared_error(y_test, 
#                                                                      y_predict_dummy_mean)))
# print("Mean squared error (linear model): {:.2f}".format(mean_squared_error(y_test, y_predict)))
# print("r2_score (dummy): {:.2f}".format(r2_score(y_test, y_predict_dummy_mean)))
# print("r2_score (linear model): {:.2f}".format(r2_score(y_test, y_predict)))
# 
# print(lm.summary())
# # Plot outputs
# plt.scatter(X_test['X6_Product_Quality'], y_test,  color='black')
# plt.plot(X_test['X6_Product_Quality'], y_predict, color='green', linewidth=2)
# plt.plot(X_test['X6_Product_Quality'], y_predict_dummy_mean, color='red', linestyle = 'dashed', 
#          linewidth=2, label = 'dummy')
# 
# plt.show()

# # Outliers 
# Dependent_Continous_Variable = ['X19_Satisfaction','X20_Likely_to_Recommend','X21_Likely_to_Purchase','X22_Purchase_Level']
# ut.is_outlier(hbat2, Dependent_Continous_Variable)


Dependent_Variable = ['X19_Satisfaction','X20_Likely_to_Recommend','X21_Likely_to_Purchase','X22_Purchase_Level','X23_Consider_Strategic_Alliance']
Dependent_Continous_Variable = ['X19_Satisfaction','X20_Likely_to_Recommend','X21_Likely_to_Purchase','X22_Purchase_Level']
Dependent_Categorical_Variable = ['X23_Consider_Strategic_Alliance']

# https://alstatr.blogspot.in/2015/08/r-python-and-sas-getting-started-with.html
# Linear Regression Modeling

ut.myLinearRegressionFunction(hbat2,Dependent_Continous_Variable)

ut.myLinearRegressionWithCategoryFunction(hbat2, Dependent_Variable, Dependent_Continous_Variable)


# # Best Subsetting
# ut.myLinearRegressionBestSubsetFunction(hbat2, Dependent_Continous_Variable)


# #Ridge Regression
ut.myLinearRidgesWithCategoryFunction(hbat2, Dependent_Variable, Dependent_Continous_Variable)

## Logistic Regression
  
ut.myLogisticRegressionFunction(hbat2, Dependent_Variable, Dependent_Categorical_Variable)


#Best Subsetting for Logistic Regression
Dependent_Variable = ['X19_Satisfaction','X20_Likely_to_Recommend','X21_Likely_to_Purchase','X22_Purchase_Level','X23_Consider_Strategic_Alliance']
Dependent_Categorical_Variable = ['X23_Consider_Strategic_Alliance']
ut.myBestSubsetClassificationFunction(hbat2, Dependent_Variable, Dependent_Categorical_Variable)

# Naive Bayes Calssification Modelling
ut.myNaiveBayesClassificationFunction(hbat2, Dependent_Variable, Dependent_Categorical_Variable)

#Knn Classification Modeling
ut.myKnnClassificationFunction(hbat2, Dependent_Variable, Dependent_Categorical_Variable)


## Decision Tree for Classification 
ut.myDecisionTreeClassificationFunction(hbat2, Dependent_Variable, Dependent_Categorical_Variable)

#Random Forest 
ut.myRandomForestClassificationFunction(hbat2, Dependent_Variable, Dependent_Categorical_Variable)

# SVM Classification
ut.mySVMClassificationFunction(hbat2, Dependent_Variable, Dependent_Categorical_Variable)


# # Data Mining : PreProcessing Steps
# # 
# # 1.  The machine learning package sklearn requires all categorical variables in numeric form. Hence, we need to 
# # convert all character/categorical variables to be numeric. This can be accomplished using the following script. 
# # In sklearn,  there is already a function for this step.
# from sklearn.preprocessing import LabelEncoder
# def ConverttoNumeric(hbat2):
#     cols = list(hbat2.select_dtypes(include=['category','object']))
#     le = LabelEncoder()
#     for i in cols:
#         try:
#             hbat2[i] = le.fit_transform(hbat2[i])
#         except:
#             print('Error in Variable :'+i)
#     return hbat2
# 
# ConverttoNumeric(hbat2)

# # 2. Create Dummy Variables
# # 
# # Suppose you want to convert categorical variables into dummy variables. It is different to the previous example 
# # as it creates dummy variables instead of convert it in numeric form.
# 
# # Create k-1 Categories
# # 
# # To avoid multi-collinearity, you can set one of the category as reference category 
# # and leave it while creating dummy variables. In the script below, we are leaving first category.
# 
# 
# X1_Customer_type_dummy = pd.get_dummies(hbat2["X1_Customer_type"], drop_first=True)
# hbat2 = pd.concat([hbat2, X1_Customer_type_dummy], axis=1)
#  
# X2_Industry_Type_dummy = pd.get_dummies(hbat2["X2_Industry_Type"], drop_first=True)
# hbat2 = pd.concat([hbat2, X2_Industry_Type_dummy], axis=1)
#  
# X3_Firm_Size_dummy = pd.get_dummies(hbat2["X3_Firm_Size"], drop_first=True)
# hbat2 = pd.concat([hbat2, X3_Firm_Size_dummy], axis=1)
#  
# X4_Region_dummy = pd.get_dummies(hbat2["X4_Region"], drop_first=True)
# hbat2 = pd.concat([hbat2, X4_Region_dummy], axis=1)
#  
# X5_Distribution_System_dummy = pd.get_dummies(hbat2["X5_Distribution_System"], drop_first=True)
# hbat2 = pd.concat([hbat2, X5_Distribution_System_dummy], axis=1)
#  
# X23_Consider_Strategic_Alliance_dummy = pd.get_dummies(hbat2["X23_Consider_Strategic_Alliance"], drop_first=True)
# hbat2 = pd.concat([hbat2, X23_Consider_Strategic_Alliance_dummy], axis=1)


# # 3. Impute Missing Values
# # Fill missing values of a particular variable
# # fill missing values with 0
# df['var1'] = df['var1'].fillna(0)
# # fill missing values with mean
# df['var1'] = df['var1'].fillna(df['var1'].mean())


# # Apply imputation to the whole dataset
# from sklearn.preprocessing import Imputer 
# 
# # Set an imputer object
# mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# 
# # Train the imputor
# mean_imputer = mean_imputer.fit(hbat2)
# 
# # Apply imputation
# hbat2_new = mean_imputer.transform(hbat2.values)
# 
# # 4. Outlier Treatment
# # 
# # There are many ways to handle or treat outliers (or extreme values). Some of the methods are as follows -
# # Cap extreme values at 95th / 99th percentile depending on distribution
# # Apply log transformation of variables. See below the implementation of log transformation in Python.
# 
# import numpy as np
# hbat2['X10_Advertising'] = np.log(hbat2['X10_Advertising'])
# 
# 
# # 5. Standardization
# # 
# # In some algorithms, it is required to standardize variables before running the actual algorithm. 
# # Standardization refers to the process of making mean of variable zero and unit variance (standard deviation).
# 
# #load dataset
# dataset = load_boston()
# predictors = dataset.data
# target = dataset.target
# df = pd.DataFrame(predictors, columns = dataset.feature_names)
# 
# #Apply Standardization
# from sklearn.preprocessing import StandardScaler
# k = StandardScaler()
# df2 = k.fit_transform(df)
