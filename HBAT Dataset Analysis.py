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
ut.myContinousJointPlot(hbat2)
ut.myContinousScatterPlot(hbat2)

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



Dependent_Variable = ['X19_Satisfaction','X20_Likely_to_Recommend','X21_Likely_to_Purchase','X22_Purchase_Level','X23_Consider_Strategic_Alliance']
Dependent_Continous_Variable = ['X19_Satisfaction','X20_Likely_to_Recommend','X21_Likely_to_Purchase','X22_Purchase_Level']
Dependent_Categorical_Variable = ['X23_Consider_Strategic_Alliance']

# Linear Regression Modeling

ut.myLinearRegressionFunction(hbat2,Dependent_Continous_Variable)
ut.myLinearRegressionWithCategoryFunction(hbat2, Dependent_Variable, Dependent_Continous_Variable)


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
