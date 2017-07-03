import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from datashape.coretypes import float64
# from dask.tests.test_cache import flag
from bokeh.layouts import column
from docutils.utils.math.tex2unichar import space
from Cython.Compiler.Options import annotate
from blaze.expr.expressions import label
def myCategoryToDummyFunc(st):
    print(st.info())
    print(st.loc[:, st.dtypes == object].columns.values)
    categorical_column_name = list(st.loc[:, st.dtypes == object].columns.values)
    print(categorical_column_name)
    for i in list(st.loc[:, st.dtypes == object].columns.values) :
#         print(st[i])
        st[i]= st[i].astype('category')
        le = preprocessing.LabelEncoder()
        le.fit(st[i])
        list(le.classes_)
        st[i]=le.transform(st[i])

def myContinousPlot(st):
    print(st.loc[:, st.dtypes == 'float64'].columns.values)
    continous_column_name = list(st.loc[:, st.dtypes == 'float64'].columns.values)
    print(continous_column_name)
    # to change the space between the cells that hold the plots:
    # to create a grid comprised of varying cell sizes:
#     gs = gridspec.GridSpec(5, 4, width_ratios=[1, 2], height_ratios=[4, 1])
#     fig = plt.figure()
    for i in list(st.loc[:, st.dtypes == 'float64'].columns.values) :
        st[i].hist()
        plt.legend()
        plt.show()
def myContinousHistPlot(st):
    print(st.loc[:, st.dtypes == 'float64'].columns.values)
    continous_column_name = list(st.loc[:, st.dtypes == 'float64'].columns.values)
    print(continous_column_name)
    #Plot violin for all attributes in a 7x2 grid
    n_cols = 2
    n_rows = 7
    for i in range(n_rows):
        fg,ax = plt.subplots(nrows=1,ncols=n_cols,figsize=(12, 8))
        for i in list(st.loc[:, st.dtypes == 'float64'].columns.values) :
            plt.figure()
            plt.hist(st[i], alpha=0.7, bins=np.arange(0,10,0.2), label=i)
            plt.legend()
            plt.show()

def myContinousBoxPlot(st, dep):
    continous_column_name = list(st.loc[:, st.dtypes == 'float64'].columns.values)
    X= st[continous_column_name]
    X = X.drop(dep, axis=1)
    # Use cubehelix to get a custom sequential palette
    pal = sns.cubehelix_palette(len(continous_column_name)-len(dep), rot=-.5, dark=.3)
    plt.figure()
    sns.boxplot(data=X, palette=pal)
    # Add in points to show each observation
    plt.legend()
    plt.show()

def myCorrelationPlot(st, dep):
    continous_column_name = list(st.loc[:, st.dtypes == 'float64'].columns.values)
    X= st[continous_column_name]
    X = X.drop(dep, axis=1)
    plt.figure()
    sns.heatmap(data=X.corr(), annot=True)
    plt.show()


def myContinousScatterPlot(st):
     for i in list(st.loc[:, st.dtypes == 'float64'].columns.values) :
        for j in list(st.loc[:, st.dtypes == 'float64'].columns.values) :
#             fig, axs = plt.subplots(len(i), len(j), sharey=True)
            st.plot(kind='scatter', x=i, y=j)
            plt.legend()
            plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="white")
def myContinousJointContourPlot(st, dep):
    continous_column_name = list(st.loc[:, st.dtypes == 'float64'].columns.values)
    X= st[continous_column_name]
    X = X.drop(dep, axis=1)
    g = sns.PairGrid(X)
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=len(continous_column_name))
    
#      for i in list(st.loc[:, st.dtypes == 'float64'].columns.values) :
#         for j in list(st.loc[:, st.dtypes == 'float64'].columns.values) :  
#             grid=sns.jointplot(st[i], st[j], kind='kde', space=0)
#             grid.ax_joint.set_aspect('equal')
#                 plt.legend()
#             plt.show()



import seaborn as sns
def myContinousJointPlot(st):
     for i in list(st.loc[:, st.dtypes == 'float64'].columns.values) :
        for j in list(st.loc[:, st.dtypes == 'float64'].columns.values) :
#             fig, axs = plt.subplots(len(i), len(j), sharey=True)
            grid = sns.jointplot(st[i], st[j], alpha=0.4)
            grid.ax_joint.set_aspect('equal')
            plt.legend()
            plt.show()


sns.set(style="whitegrid");
import seaborn as sns
def myVioplotPlot(st, dep):
    continous_column_name = list(st.loc[:, st.dtypes == 'float64'].columns.values)
    X = st[continous_column_name]
    X = X.drop(dep, axis = 1)
    cols=X.columns 
#     # Use cubehelix to get a custom sequential palette
#     pal = sns.cubehelix_palette(len(continous_column_name), rot=-.5, dark=.3)
    #Plot violin for all attributes in a 7x2 grid
    n_cols = 2
    n_rows = 7
    
    for i in range(n_rows):
        fg,ax = plt.subplots(nrows=1,ncols=n_cols,figsize=(12, 8))
        for j in range(n_cols):
            sns.violinplot(y=cols[i*n_cols+j], data=X, ax=ax[j])

    # Show each distribution with both violins and points
    
#     plt.figure()
#     sns.violinplot(data=X, palette=pal, inner="points")
#     plt.legend()
    plt.show()
#      for i in list(st.loc[:, st.dtypes == 'float64'].columns.values) :
#             sns.violinplot(st[i])
#             plt.show()

def myPairWiseScatterColorPlot(st, dep, study_var):
    continous_column_name = list(st.loc[:, st.dtypes == 'float64'].columns.values)
    for i in list(st.loc[:, st.dtypes == 'float64'].columns.values) :
        for j in list(st.loc[:, st.dtypes == 'float64'].columns.values) :    
            ax=st.plot.scatter(i, j, c=st[study_var], s=st[study_var]**2,colormap='viridis')
            ax.set_aspect('equal')
            plt.legend()
            plt.show()



def myPairWiseScatterPlot(st):
    #Visualizing pairwise relationships in a dataset
    sns.pairplot(st)
    plt.legend()
    plt.show()

import seaborn as sns
def myPairWiseScatterKdePlot(st):
    g = sns.PairGrid(st)
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6)
    plt.legend()
    plt.show()

    
import pandas.tools.plotting 
def myScatterMatrixPlot(st):
    #Visualizing pairwise relationships in a dataset
    plt.figure()
    pandas.tools.plotting.scatter_matrix(st)
    plt.legend()
    plt.show()


import matplotlib.pyplot as plt
import pandas.tools.plotting 
def myParallelPlot(st):
    for i in list(st.loc[:, st.dtypes == 'Object'].columns.values) :
        #Visualizing pairwise relationships in a dataset
        plt.figure()
        pandas.tools.plotting.parallel_coordinates(st,i)
        plt.legend()
        plt.show()
                                                    



def myFrequencyDistribution(st):
           for i in list(st.loc[:, st.dtypes == object].columns.values) :
               print(st[i].value_counts(ascending=True))    



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from numpy import random, column_stack, rank
from statsmodels.api import add_constant, OLS
from statsmodels.stats import outliers_influence
# from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def myLinearRegressionFunction(st, dep):
    myCategoryToDummyFunc(st)
    continous_column_name = list(st.loc[:, st.dtypes == 'float64'].columns.values)
    X = st[continous_column_name]
    X = X.drop(dep, axis = 1)
    
    for i in dep :
        y = st[i]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        # Fit the model
        model = OLS(y_train, X_train)
        fit = model.fit()
        print(fit.summary())
        
#         # Break into left and right hand side; y and X
#         y_train, X_train = dmatrices(formula="X19_Satisfaction ~ X6_Product_Quality+ X7_E_Commerce  + X8_Technical_Support + X9_Complaint_Resolution + X10_Advertising+ X11_Product_Line + X12_Salesforce_Image + X13_Competitive_Pricing + X14_Warranty_Claims + X15_New_Products + X16_Order_Billing + X17_Price_Flexibility + X18_Delivery_Speed", data=st, return_type="dataframe")
#         
#         # For each Xi, calculate VIF
#         vif = [variance_inflation_factor(X_train.values, i) for i in range(X.shape[1])]
#         
#         # Fit X to y
#         result = sm.OLS(y, X).fit()
#         print(outliers_influence.variance_inflation_factor(X_train, 6))


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from numpy import random, column_stack
from statsmodels.api import add_constant, OLS
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

def myLinearRegressionBestSubsetFunction(st, dep):
    myCategoryToDummyFunc(st)
    continous_column_name = list(st.loc[:, st.dtypes == 'float64'].columns.values)
    X = st[continous_column_name]
    X = X.drop(dep, axis = 1)
    print(X.head())    
    for i in dep :
        y = st[i]
#         continous_column_name = list(st.loc[:, st.dtypes == 'float64'].columns.values)
        # This dataset is way too high-dimensional. Better do PCA:
        pca = PCA(n_components=2)
        
        # Maybe some original features where good, too?
        selection = SelectKBest(k=1)
        # Build estimator from PCA and Univariate selection:
        combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
        

        # Use combined features to transform dataset:
        X_features = combined_features.fit(X, y).transform(X)
        svm = SVC(kernel="linear")

        # Do grid search over k, n_components and C:
        
        pipeline = Pipeline([("features", combined_features), ("svm", svm)])

        param_grid = dict(features__pca__n_components=[1, 2, 3],
                          features__univ_select__k=[1, 2],
                          svm__C=[0.1, 1, 10])
        
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
        grid_search.fit(X_features, y)
        print(grid_search.best_estimator_)

        
#         X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state=0)
#         Fit the model
#         model = OLS(y_train, X_train)
#         fit = model.fit()
#         print(fit.summary())
#         # Run the model on X_test and show the first five results
#         print(list(fit.predict(X_test)[0:5]),y_test[0:5])
#         # Apply the model we created using the training data to the test data, and calculate the RSS.
#         print(((y_test - fit.predict(X_test)) **2).sum())
#         # Calculate the MSE
#         print(np.mean((fit.predict(X_test) - y_test) **2))



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from numpy import random, column_stack
from statsmodels.api import add_constant, OLS

def myLinearRegressionWithCategoryFunction(st, dep, dep_conti):
    myCategoryToDummyFunc(st)
#     continous_column_name = list(st.loc[:, st.dtypes == 'float64'].columns.values)
#     X = st[continous_column_name]
#     category_column_name = list(st.loc[:, st.dtypes == 'object'].columns.values)
#     X = st[category_column_name]
    X = st
    X = X.drop(dep, axis = 1)
    print(X.head())
    
    for i in dep_conti :
        y = st[i]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        # Fit the model
        model = OLS(y_train, X_train)
        fit = model.fit()
        print(fit.summary())
        # Run the model on X_test and show the first five results
        print(list(fit.predict(X_test)[0:5]),y_test[0:5])
        # Apply the model we created using the training data to the test data, and calculate the RSS.
        print(((y_test - fit.predict(X_test)) **2).sum())
        # Calculate the MSE
        print(np.mean((fit.predict(X_test) - y_test) **2))
        


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from numpy import random, column_stack
from statsmodels.api import add_constant, OLS
from sklearn.linear_model import Ridge
def myLinearRidgesWithCategoryFunction(st, dep, dep_conti):
    myCategoryToDummyFunc(st)
#     continous_column_name = list(st.loc[:, st.dtypes == 'float64'].columns.values)
#     X = st[continous_column_name]
#     category_column_name = list(st.loc[:, st.dtypes == 'object'].columns.values)
#     X = st[category_column_name]
    X = st
    X = X.drop(dep, axis = 1)
    print(X.head())
    
    for i in dep_conti :
        y = st[i]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        # Fit the model
        Ridge_model = Ridge(alpha=20.0).fit(X_train, y_train)
        print(y.head())
        print('ridge regression linear model intercept: {}'
             .format(Ridge_model.intercept_))
        print('ridge regression linear model coeff:\n{}'
             .format(Ridge_model.coef_))
        print('R-squared score (training): {:.3f}'
             .format(Ridge_model.score(X_train, y_train)))
        print('R-squared score (test): {:.3f}'
             .format(Ridge_model.score(X_test, y_test)))
        print('Number of non-zero features: {}'
             .format(np.sum(Ridge_model.coef_ != 0)))




import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
        
def myBestSubsetClassificationFunction(st, dep, dep_category):
    myCategoryToDummyFunc(st)
    X = st
    X = X.drop(dep, axis = 1)
    print(X.head())
    for i in dep_category :
        y = st[i]
        # Create the RFE object and compute a cross-validated score.
        svc = SVC(kernel="linear")
        # The "accuracy" scoring is proportional to the number of correct
        # classifications
        rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),scoring='accuracy')
        print(rfecv.fit(X, y))    
        print("Optimal number of features : %d" % rfecv.n_features_)
    
    
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.legend()
        plt.show()

from sklearn.cross_validation import train_test_split                 
import pandas as pd
import statsmodels.api as sm
import numpy as np
from nltk import ConfusionMatrix 
import numpy as np
from sklearn import metrics
def myLogisticRegressionFunction(st, dep, dep_category):
    myCategoryToDummyFunc(st)
    X = st
    X = X.drop(dep, axis = 1)
    print(X.head())
    for i in dep_category :
        y = st[i]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        print(X_train.head())
        print(y_train.head())
        #Build Logistic Regression Model
        #Fit Logit model
        logit = sm.Logit(y_train, X_train)
        result = logit.fit()
        #Summary of Logistic regression model
        print(result.summary())
        # use 0.5 cutoff for predicting 'default' 
        expected =y_test
        probs= result.predict(X_test)
        predicted = np.where(probs > 0.5, 1, 0) 
        print(ConfusionMatrix(list(expected), list(predicted)))
        # check accuracy, sensitivity, specificity 
        acc = metrics.accuracy_score(expected, predicted) 
        print("Accuracy for above logistic Model is :: %0.2f" %acc)
        # sensitivity:
        sens = metrics.recall_score(expected, predicted)
        print("Sensitivity for above logistic Model is :: %0.2f" %sens)
        #Specificity
        spec = metrics.precision_score(expected, predicted)
        print("Specificity for above logistic Model is :: %0.2f" %spec)
        # calculate AUC 
        auc = metrics.roc_auc_score(expected, probs)
        print("Area under curve for above logistic Model is :: %0.2f" %auc)
        
        fpr, tpr, thresholds = metrics.roc_curve(expected, probs) 
        plt.plot(fpr, tpr,color='red', label='ROC curve (area = %0.2f)'% auc) 
        
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate)') 
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.legend()
        plt.show()
        
        
        
from sklearn.cross_validation import train_test_split                 
import pandas as pd
import statsmodels.api as sm
import numpy as np
from nltk import ConfusionMatrix 
import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

def myNaiveBayesClassificationFunction(st, dep, dep_category):
    myCategoryToDummyFunc(st)
    X = st
    X = X.drop(dep, axis = 1)
    print(X.head())
    for i in dep_category :
        y = st[i]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        nbclf = GaussianNB().fit(X_train, y_train)
        print(y.head())
        print('Accuracy of GaussianNB classifier on training set: {:.2f}'
             .format(nbclf.score(X_train, y_train)))
        print('Accuracy of GaussianNB classifier on test set: {:.2f}'
             .format(nbclf.score(X_test, y_test)))
                # Plotting Histogram
        probs = nbclf.predict_proba(X_test)[:, 1] 
        plt.hist(probs) 
        plt.show()

        expected =y_test
        predicted = nbclf.predict(X_test)
        print (ConfusionMatrix(list(expected), list(predicted)))



        print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(metrics.accuracy_score(expected, predicted)))
        print('Sensitivity of GaussianNB classifier on training set: {:.2f}'.format(metrics.recall_score(expected, predicted)))
        print('Specificity of GaussianNB classifier on training set: {:.2f}'.format(metrics.precision_score(expected, predicted)))
        print('AUC of GaussianNB classifier on training set: {:.2f}'.format(metrics.roc_auc_score(expected, predicted)))
        
        fpr, tpr, thresholds = metrics.roc_curve(expected, probs) 
        plt.plot(fpr, tpr,color='red', label='ROC curve (area = %0.2f)'.format(metrics.roc_auc_score(expected, predicted))) 
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate)') 
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

from sklearn.cross_validation import train_test_split                 
import pandas as pd
import statsmodels.api as sm
import numpy as np
from nltk import ConfusionMatrix 
import numpy as np
from sklearn import metrics
from sklearn import neighbors
import matplotlib.pyplot as plt 

def myKnnClassificationFunction(st, dep, dep_category):
    myCategoryToDummyFunc(st)
    X = st
    X = X.drop(dep, axis = 1)
    print(X.head())
    for i in dep_category :
        y = st[i]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        clf = neighbors.KNeighborsClassifier(3, weights = 'uniform')
        trained_model = clf.fit(X_train, y_train)
        # Plotting Histogram
        probs = trained_model.predict_proba(X_test)[:, 1] 
        plt.hist(probs) 
        plt.show()

        expected =y_test
        predicted = trained_model.predict(X_test)
        print (ConfusionMatrix(list(expected), list(predicted)))



        print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(metrics.accuracy_score(expected, predicted)))
        print('Sensitivity of GaussianNB classifier on training set: {:.2f}'.format(metrics.recall_score(expected, predicted)))
        print('Specificity of GaussianNB classifier on training set: {:.2f}'.format(metrics.precision_score(expected, predicted)))
        print('AUC of GaussianNB classifier on training set: {:.2f}'.format(metrics.roc_auc_score(expected, predicted)))
        
        fpr, tpr, thresholds = metrics.roc_curve(expected, probs) 
        plt.plot(fpr, tpr,color='red', label='ROC curve (area = %0.2f)'.format(metrics.roc_auc_score(expected, predicted))) 
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate)') 
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

#         # Apply the learner to the new, unclassified observation.
#         print(trained_model.predict(X_test))
#         print(trained_model.predict_proba(X_test))
        
        
from sklearn.cross_validation import train_test_split                 
import pandas as pd
import statsmodels.api as sm
import numpy as np
from nltk import ConfusionMatrix 
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus 
from IPython.display import Image  

def myDecisionTreeClassificationFunction(st, dep, dep_category):
    myCategoryToDummyFunc(st)
    X = st
    X = X.drop(dep, axis = 1)
    print(X.head())
    for i in dep_category :
        y = st[i]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        #Decision Tree
        model_tree = DecisionTreeClassifier(max_depth=7)
         
        #Fit the model:
        model_tree.fit(X_train,y_train)
        print(model_tree) 

        # Plotting Histogram
        probs = model_tree.predict_proba(X_test)[:, 1] 
        plt.hist(probs) 
        plt.show()

        expected =y_test
        predicted = model_tree.predict(X_test)
        print (ConfusionMatrix(list(expected), list(predicted)))



        print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(metrics.accuracy_score(expected, predicted)))
        print('Sensitivity of GaussianNB classifier on training set: {:.2f}'.format(metrics.recall_score(expected, predicted)))
        print('Specificity of GaussianNB classifier on training set: {:.2f}'.format(metrics.precision_score(expected, predicted)))
        print('AUC of GaussianNB classifier on training set: {:.2f}'.format(metrics.roc_auc_score(expected, predicted)))
        
        fpr, tpr, thresholds = metrics.roc_curve(expected, probs) 
        plt.plot(fpr, tpr,color='red', label='ROC curve (area = %0.2f)'.format(metrics.roc_auc_score(expected, predicted))) 
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate)') 
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
 
#         #Make predictions on test set
#         predictions_tree = model_tree.predict_proba(X_test)
#         print(predictions_tree)   
#         dot_data = tree.export_graphviz(model_tree, out_file=None) 
#         graph = pydotplus.graph_from_dot_data(dot_data) 
#         graph.write_pdf("Decision_Tree_Map.pdf") 
#         #AUC
#         false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, predictions_tree[:,1])
#         metrics.roc_auc_score(false_positive_rate, true_positive_rate)
#         dot_data = tree.export_graphviz(model_tree, out_file=None, 
#                                  feature_names=X_train,  
#                                  class_names=y_train,  
#                                  filled=True, rounded=True,  
#                                  special_characters=True)  
#         graph = pydotplus.graph_from_dot_data(dot_data)  
#         Image(graph.create_png())



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
def myRandomForestClassificationFunction(st, dep, dep_category):
    myCategoryToDummyFunc(st)
    X = st
    X = X.drop(dep, axis = 1)
    print(X.head())
    for i in dep_category :
        y = st[i]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        #Random Forest
        model_rf = RandomForestClassifier(n_estimators=100, max_depth=7)

        #Fit the model:
        target = y_train
        model_rf.fit(X_train,y_train)
        
        # Plotting Histogram
        probs = model_rf.predict_proba(X_test)[:, 1] 
        plt.hist(probs) 
        plt.show()

        expected =y_test
        predicted = model_rf.predict(X_test)
        print (ConfusionMatrix(list(expected), list(predicted)))



        print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(metrics.accuracy_score(expected, predicted)))
        print('Sensitivity of GaussianNB classifier on training set: {:.2f}'.format(metrics.recall_score(expected, predicted)))
        print('Specificity of GaussianNB classifier on training set: {:.2f}'.format(metrics.precision_score(expected, predicted)))
        print('AUC of GaussianNB classifier on training set: {:.2f}'.format(metrics.roc_auc_score(expected, predicted)))
        
        fpr, tpr, thresholds = metrics.roc_curve(expected, probs) 
        plt.plot(fpr, tpr,color='red', label='ROC curve (area = %0.2f)'.format(metrics.roc_auc_score(expected, predicted))) 
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate)') 
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        #Make predictions on test set
        predictions_rf = model_rf.predict_proba(X_test)
        
        #AUC
        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, predictions_rf[:,1])
        print(metrics.auc(false_positive_rate, true_positive_rate))
        
        
        
        #Variable Importance
        importances = pd.Series(model_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        print(importances)
        importances.plot.bar()
        
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split                 
import pandas as pd
import statsmodels.api as sm
import numpy as np
from nltk import ConfusionMatrix 
import numpy as np
from sklearn import metrics
        
        
def mySVMClassificationFunction(st, dep, dep_category):
    myCategoryToDummyFunc(st)
    X = st
    X = X.drop(dep, axis = 1)
    print(X.head())
    for i in dep_category :
        y = st[i]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        #SVM Classification
        model = SVC(C=1.0, gamma='auto', kernel='linear', probability=True)
        model.fit(X_train, y_train)
        print(model)
        # Plotting Histogram
        probs = model.predict_proba(X_test)[:, 1] 
        plt.hist(probs) 
        plt.legend()
        plt.show()
        
        expected =y_test
        predicted = model.predict(X_test)
        print (ConfusionMatrix(list(expected), list(predicted)))

        # check accuracy, sensitivity, specificity 
        acc = metrics.accuracy_score(expected, predicted) 
        print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(acc))

        # sensitivity:
        sens = metrics.recall_score(expected, predicted)
        print('Sensitivity of GaussianNB classifier on training set: {:.2f}'.format(sens))

        #Specificity
        spec = metrics.precision_score(expected, predicted)
        print('Specificity of GaussianNB classifier on training set: {:.2f}'.format(spec))

        # calculate AUC 
        auc = metrics.roc_auc_score(expected, probs)
        print('AUC of GaussianNB classifier on training set: {:.2f}'.format(auc))



        #ROC CURVES and AUC 
        # plot ROC curve 
        fpr, tpr, thresholds = metrics.roc_curve(expected, probs) 
        plt.plot(fpr, tpr,color='red', label='ROC curve (area = %0.2f)'% auc) 
        #plt.xlim([0.0, 1.0]) 
        #plt.ylim([0.0, 1.0]) 
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate)') 
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.legend()
        plt.show()


from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans

def myKmeanClusteringFunction(st, dep):
    # subset clustering variables
    continous_column_name = list(st.loc[:, st.dtypes == 'float64'].columns.values)
    cluster= st[continous_column_name]
    cluster = cluster.drop(dep, axis=1)
    dep = cluster.columns
    print(cluster.describe())
    print(dep)
    clustervar=cluster.copy()

#     # standardize clustering variables to have mean=0 and sd=1
#     for i in dep:
#         clustervar=cluster.copy()
#         clustervar[dep]=preprocessing.scale(clustervar[dep].astype('float64'))
    # split data into train and test sets
    np.random.seed(1234)
    clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)
    print(clus_train.head())
    # k-means cluster analysis for 1-9 clusters                                                           
    from scipy.spatial.distance import cdist
    clusters=range(1,10)
    meandist=[]
      
    for k in clusters:
        model=KMeans(n_clusters=k)
        model.fit(clus_train)
        clusassign=model.predict(clus_train)
        meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
        / clus_train.shape[0])
    """
    Plot average distance from observations from the cluster centroid
    to use the Elbow Method to identify number of clusters to choose
    """
    print(meandist)  
    plt.plot(clusters, meandist)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average distance')
    plt.title('Selecting k with the Elbow Method')
    plt.show()

    # Interpret 3 cluster solution
    model3=KMeans(n_clusters=5)
    model3.fit(clus_train)
    clusassign=model3.predict(clus_train)
    # plot clusters
      
    from sklearn.decomposition import PCA
    pca_2 = PCA(2)
    plot_columns = pca_2.fit_transform(clus_train)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
    plt.xlabel('Canonical variable 1')
    plt.ylabel('Canonical variable 2')
    plt.title('Scatterplot of Canonical Variables for 3 Clusters')
    plt.show()

    """
    BEGIN multiple steps to merge cluster assignment with clustering variables to examine
    cluster variable means by cluster
    """
    # create a unique identifier variable from the index for the 
    # cluster training data to merge with the cluster assignment variable
    clus_train.reset_index(level=0, inplace=True)
    # create a list that has the new index variable
    cluslist=list(clus_train['index'])
    # create a list of cluster assignments
    labels=list(model3.labels_)
    # combine index variable list with cluster assignment list into a dictionary
    newlist=dict(zip(cluslist, labels))
    newlist
    # convert newlist dictionary to a dataframe
    newclus=DataFrame.from_dict(newlist, orient='index')
    newclus
    # rename the cluster assignment column
    newclus.columns = ['cluster']
      
    # now do the same for the cluster assignment variable
    # create a unique identifier variable from the index for the 
    # cluster assignment dataframe 
    # to merge with cluster training data
    newclus.reset_index(level=0, inplace=True)
    # merge the cluster assignment dataframe with the cluster training variable dataframe
    # by the index variable
    merged_train=pd.merge(clus_train, newclus, on='index')
    print(merged_train.head(n=100))
    # cluster frequencies
    print(merged_train.cluster.value_counts())
      
    """
    END multiple steps to merge cluster assignment with clustering variables to examine
    cluster variable means by cluster
    """
      
    # FINALLY calculate clustering variable means by cluster
    clustergrp = merged_train.groupby('cluster').mean()
    print ("Clustering variable means by cluster")
    print(clustergrp)
      
      
    # validate clusters in training data by examining cluster differences in GPA using ANOVA
    # first have to merge GPA with clustering variables and cluster assignment data 
    print(cluster['X7_Product_Quality'].head())

    test_data=cluster['X7_Product_Quality']
    # split GPA data into train and test sets
    test_train, test_test = train_test_split(test_data, test_size=.3, random_state=123)
    test_train1=pd.DataFrame(test_train)
    test_train1.reset_index(level=0, inplace=True)
    merged_train_all=pd.merge(test_train1, merged_train, on='index')
#     sub1 = merged_train_all[['X6_Product_Quality', 'cluster']].dropna()
    print(merged_train_all.head())
    sub1 = merged_train_all[['X7_Product_Quality_x', 'cluster']].dropna()

      
    import statsmodels.formula.api as smf
    import statsmodels.stats.multicomp as multi 
      
    gpamod = smf.ols(formula='X7_Product_Quality_x ~ C(cluster)', data=sub1).fit()
    print (gpamod.summary())
      
    print ('means for GPA by cluster')
    m1= sub1.groupby('cluster').mean()
    print (m1)
      
    print ('standard deviations for GPA by cluster')
    m2= sub1.groupby('cluster').std()
    print (m2)
      
    mc1 = multi.MultiComparison(sub1['X7_Product_Quality_x'], sub1['cluster'])
    res1 = mc1.tukeyhsd()
    print(res1.summary())
    
        

import numpy as np
import pandas as pd        
def is_outlier(st, dep, thresh=3.5):
    continous_column_name = list(st.loc[:, st.dtypes == 'float64'].columns.values)
    st = st[continous_column_name]
    st = st.drop(dep, axis = 1)
    if len(st.shape) == 1:
        st = st[:,None]
    print(st.head())
    col=st.columns
    for i in col:
        median = np.median(st[i], axis=0)
        diff = np.sum((st[i] - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        modified_z_score = 0.6745 * diff / med_abs_deviation
        return modified_z_score > thresh