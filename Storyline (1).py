#!/usr/bin/env python
# coding: utf-8

# In[293]:


#Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# create test set
from sklearn import model_selection, metrics

# get mutual information for classification data
from sklearn.feature_selection import mutual_info_classif

# get mutual information for regression data
from sklearn.feature_selection import mutual_info_regression

#select K best features
from sklearn.feature_selection import SelectKBest

#To select best x percentile features
from sklearn.feature_selection import SelectPercentile


# Standardization of data
from sklearn.preprocessing import StandardScaler

#For variance threshold calcuation
from sklearn.feature_selection import VarianceThreshold

#Calc p-value of the features
from sklearn.feature_selection import chi2

# to keep track of training time
import datetime

# standard scaler
from sklearn.preprocessing import StandardScaler

# normalizer
from sklearn.preprocessing import Normalizer

# logistic regression model
from sklearn.linear_model import LogisticRegression

# metrics used for evaluation
from sklearn.metrics import f1_score, matthews_corrcoef

# XG boost Tree
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report
import re



from sklearn.metrics import plot_roc_curve

# KNN imputation
from sklearn.impute import KNNImputer

# normalizer
from sklearn.preprocessing import Normalizer

# variance threshold
from sklearn.feature_selection import VarianceThreshold

# RFECV
from sklearn.model_selection import StratifiedKFold

# random forest
from sklearn.ensemble import RandomForestClassifier

#for classifications
from sklearn.datasets import make_classification

# evaluation metric
from sklearn.metrics import matthews_corrcoef, make_scorer


# In[220]:


# Reading the data
data = pd.read_csv("D:\\Master's\\Intern\\DataFiles\\data.csv")
data.head() # overview of data


# # Handling Missing Values

# Before Handling missing values we need to check for infinite values and replace it with NAN and then we can handle NAN values

# In[221]:


data.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[222]:


data.isnull().sum() # getting sum of missing values in the data set


# In[223]:


data.shape # number of rows and columns in data set


# As there are only three rows with missing values we aren't dropping the rows instead using mean, median or mode to fill

# In[224]:


# to find which technique is best using box-plot
fig, ab = plt.subplots(figsize=(10,6))
sns.boxplot(data.agreeableness_percentile_q1)
# to find skewness using distribution plot(if the data is skewed then mean is not used)
fig, ab = plt.subplots(figsize=(10,8))
sns.distplot(data.agreeableness_percentile_q1)


# From the above figures we can clearly skewness in the data so median or mode can be used. Let us take another feature and check if we can apply this on whole data

# In[225]:


data[data['sympathy_percentile_q1'].isnull()] # Choosen one of the column which isn't in the above missing values output


# In[356]:


# to find which technique is best using box-plot
fig, ab = plt.subplots(figsize=(10,6))
sns.boxplot(data.sympathy_percentile_q1)
# to find skewness using distribution plot(if the data is skewed then mean is not used)
fig, ab = plt.subplots(figsize=(10,8))
sns.distplot(data.sympathy_percentile_q1)


# Hence we can use median or mode to fill the missing values as the graph is positively skewed

# In[226]:


#Using mode to fill in the values as there are only 3 rows missing
data = data.fillna(data.mode())


# In[227]:


data.isnull().sum() # getting sum of missing values after filling out missing values in the data set


# There are no more missing data in the dataset

# In[228]:


#Let us import the target variable and check for missing values
y=pd.read_csv("D:\\Master's\\Intern\\DataFiles\\patient_diagnoses.csv")
y=y[['anxiety']]
y.head()


# First we handle the inf values then missing

# In[229]:


y.replace([np.inf, -np.inf], np.nan, inplace=True) # replacing infinite value with nan


# In[230]:


y.isnull().sum() # to get sum of missing values


# In[231]:


y[y['anxiety'].isnull()] #Rows that has missing values in it


# In[232]:


y.replace({False: 0, True: 1}, inplace=True) # Replacing True and False with 1 and 0
y.head()


# In[233]:


# to find skewness using distribution plot(if the data is skewed, mean is not used)
fig, ab = plt.subplots(figsize=(10,8))
sns.distplot(y.anxiety)


# As the data is skewed we are using mode imputation to fill in the missing values

# In[234]:


# Using mode to fill missing values
y = y.fillna(y.mode().iloc[0])
y.head()


# In[235]:


# Re check for the missing values
y.isnull().sum()


# Now the data doesn't have any missing values, let us go on to the next 
# step is **Data Visualization**

# # **Data** **Visualization**

# In[236]:


#Counting number of people with anexiety
anxiety_cnt=0
non_anexiety_cnt=0
for i in y.anxiety:
  if i==0:
    anxiety_cnt+=1
  else:
      non_anexiety_cnt+=1
print('Anxiety:',anxiety_cnt,'Non-Anxiety:',non_anexiety_cnt)

y['anxiety'].value_counts().plot(kind='bar')


# In[237]:


data.describe() #Breif description of the data


# Performing normalization or standirdization as differences between values of features are very high to observe on plot. 

# In[238]:


data.drop(columns=data.columns[0], axis=1, inplace=True) # dropping first column which has no label to it and it is just like ID whihc is not used for prediction
data.shape
data.head()


# In[239]:


#Standardization of the data
scaler=StandardScaler()
data_scaled=scaler.fit_transform(data)
data_scaled


# In[240]:


plt.hist(data_scaled[:,1],bins=20) # plotting the first feature after standardization


# In[241]:


plt.hist(data_scaled[:,100],bins=20) #similarly plotting 100th column


# violin plot is used to know how features are related with anxiety. Plotting first 10 features with respect to anxiety

# In[242]:


con_data=data
data_dia = y
data_n_2 = (data - data.mean()) / (data.std()) 
mix_data=pd.concat([y,data_n_2.iloc[:,0:10]],axis=1) #plotting 1-10 features
mix_data=pd.melt(mix_data,id_vars='anxiety',var_name='features',value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="anxiety", data=mix_data,split=True, inner="quart")
plt.xticks(rotation=90)


# In this first 10 features we can see that none of the features has difference in median which is not good for classification, however altruism_rawscore_q1 has minimum difference in median which is still not enough for classification let us observe other 10 features

# In[243]:


con_data=data
data_dia = y
data_n_2 = (data - data.mean()) / (data.std()) 
mix_data=pd.concat([y,data_n_2.iloc[:,10:20]],axis=1) #plotting 10-20 features
mix_data=pd.melt(mix_data,id_vars='anxiety',var_name='features',value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="anxiety", data=mix_data,split=True, inner="quart")
plt.xticks(rotation=90)


# Similarly in this plot with features from 10-20 none of the features show differences

# In[244]:


con_data=data
data_dia = y
data_n_2 = (data - data.mean()) / (data.std()) 
mix_data=pd.concat([y,data_n_2.iloc[:,20:30]],axis=1) #plotting 20-30 features
mix_data=pd.melt(mix_data,id_vars='anxiety',var_name='features',value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="anxiety", data=mix_data,split=True, inner="quart")
plt.xticks(rotation=90)


# From above plots which contains information about features from 1-30 none of them shows any difference in the median and it is difficult to plot all(~80k) the features and observe the median differences with respect to Anxiety
# 
# This median difference can be also be observed using boxplot
# 
# 

# In[245]:


plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue="anxiety", data=mix_data) # box-plot for the features 20:30
plt.xticks(rotation=90)


# With box-plot we can see that activity_level_percentile_q1 and assertiveness_percentile_q1 are related with each other. To know correlation between this features we can clearly see through joint plot which is below:

# In[246]:


sns.jointplot(data.loc[:,'activity_level_percentile_q1'], data.loc[:,'assertiveness_percentile_q1'], kind="reg", color="#ce1414")


# From this plot we can clearly how this two are related with each other

# Now let us compare three features together and try to know corelation between them

# In[247]:


sns.set(style="white")
df = data.loc[:,['activity_level_percentile_q1','assertiveness_percentile_q1','cheerfulness_percentile_q1']]
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)


# Using pair grid plot we can clearly see how this three features are related with each other.
# 

# Downcasts data-types of the features to reduce memory usages and it help us to deal with all the features which is done below:

# In[248]:


def downcast_dtypes(df):
    _start = df.memory_usage(deep=True).sum() / 1024 ** 2
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64","int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    _end = df.memory_usage(deep=True).sum() / 1024 ** 2
    saved = (_start - _end) / _start * 100
    return df


# In[249]:


data= downcast_dtypes(data)#Downcasting the data


# In[258]:


np.where(data.values >= np.finfo(np.float64).max)# identifies the infinite values in the datset


# #  **Feature Selection**

# Using Random Forest feature importance to get appropriate features related to target variable

# In[285]:


# define the model
model = RandomForestClassifier()

# fit the model
model.fit(data.fillna(0), y)

# get importance
importance = model.feature_importances_


# In[263]:


avg_score=sum(importance)/len(importance)
print('%.5f'% avg_score)#calc average score and then applying the condition in importance to filter out the columns


# Average score of the features is very low and it is not good to include features with 0.0001 importance, hence we need to drop such features.

# In[264]:


# create an Empty list to store features which gives score greater than average score
list=[]
# summarize feature importance
for i,v in enumerate(importance):
    if(v>avg_score):
        print('Feature: %0d, Score: %.5f' % (i,v))
        list.append(i)
        
        
 


# Here are the features which are related with anxiety other than need to be droped

# In[265]:


data_1=data.iloc[:,list] #data with required feature score


# In[266]:


data_1.shape


# There are only 634 features which are important for the anxiety

# In[267]:


data_1.head()


# Droping the columns which are co related, making sure that one of the column remains

# In[268]:


# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[269]:


#Using Pearson Correlation
corrmat = data_1.corr()
fig, ax = plt.subplots()
fig.set_size_inches(11,11)
sns.heatmap(corrmat)


# Using the heat map we can cleary see that there are some features which are correlated with each other

# In[270]:


corr_features=correlation(data_1,0.9)
len(corr_features)


# So there are 117 features which are correlated and need to be droped(Makin sure that one of the column remains)

# In[277]:


data_1=data_1.drop(corr_features,axis=1)


# In[286]:


data_1=data_1.fillna(0)


# In[287]:


data_1.head()


# Now the data has been dropped to 517 columns, lets try more feature selection using best k feature

# In[288]:


mutual_info = mutual_info_classif(data_1, y) #to get mutual information between features and target
mutual_info


# In[289]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = data_1.columns
mutual_info.sort_values(ascending=False) #feature with high mutual info are listed in desc order


# In[282]:


#let's plot the ordered mutual_info values per feature
mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))


# From the figure we can clearly see that there are only 21 features which are important

# In[291]:


# select the  top 10 important features from 21 as the other features from 21 doesn't meet the minimum criteria(0.05)
sel_ten_cols = SelectKBest(mutual_info_classif, k=10)
sel_ten_cols.fit(data_1, y)
data_1.columns[sel_ten_cols.get_support()]


# This are the top 10 features which are provinding us with the information with respect to anxiety

# In[309]:


data_top=data_1[["pcm_fftMag_spectralRollOff90.0_sma_linregc2_q1",
       'pcm_fftMag_spectralRollOff50.0_sma_de_de_iqr1-3_q1',
       'pcm_fftMag_spectralMinPos_sma_de_de_quartile2_q1',
       'pcm_fftMag_spectralCentroid_sma_de_iqr2-3_q2',
       'mfcc_sma_de[6]_percentile95.0_q6',
       'pcm_fftMag_spectralRollOff50.0_sma_de_quartile3_q9',
       'pcm_fftMag_spectralCentroid_sma_de_iqr2-3_q9',
       'pcm_fftMag_spectralCentroid_sma_de_iqr1-3_q10',
       'FaceEmbeddingDim66.min.q1', 'FaceEmbeddingDim101.kurtosis.q7']]#created new data frame for top 10 features


# In[310]:


data_top.head()


# Let us find out are this features significant and are really provinding us with useful information using Fisher’s Exact test p-value (chi-square test)

# In[320]:


f_p_values=chi2(data_top,y) # chi-square test on top 10 features
p_values=pd.Series(f_p_values[1])
p_values.index=data_top.columns
p_values.sort_index(ascending=False)


# So the most important feature are "pcm_fftMag_spectralRollOff90.0_sma_linregc2_q1" then "pcm_fftMag_spectralRollOff50.0_sma_de_quartile3_q9", followed by other featuers

# We are done with feature selection, now its time for us to build the model with this top 10 features

# # Machine Learning Model for Anxiety

# Multinomial Naive Bayes algorithm 

# In[322]:


[train_in, test_in, train_out, test_out] = model_selection.train_test_split(data_top, y, test_size=.25) # splitting the data
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB().fit(train_in, train_out)
print("score on test: " + str(mnb.score(test_in, test_out)))
print("score on train: "+ str(mnb.score(train_in, train_out)))


# Multinomial Naive Bayes model is an underfitting model

# Logistic Regresstion

# In[323]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=1000)
lr.fit(train_in, train_out)
print("score on test: " + str(lr.score(test_in, test_out)))
print("score on train: "+ str(lr.score(train_in, train_out)))


# Logistic Regression is an overfitting model

# K-Nearest-Neighbor(KNN)

# In[324]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm = 'brute', n_jobs=-1)
knn.fit(train_in, train_out)
print("score on test: " + str(knn.score(test_in, test_out)))
print("score on train: "+ str(knn.score(train_in, train_out)))


# Super Vector Machine

# In[325]:


from sklearn.svm import LinearSVC
svm=LinearSVC(C=0.0001)
svm.fit(train_in, train_out)
print("score on test: " + str(svm.score(test_in, test_out)))
print("score on train: "+ str(svm.score(train_in, train_out)))


# Support Vector Machine model is an optimal model

# Decision Tree

# In[326]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(train_in, train_out)
print("score on test: "  + str(clf.score(test_in, test_out)))
print("score on train: " + str(clf.score(train_in, train_out)))


# Decision Tree is an overfitting model

# Bagging Decision Tree

# In[327]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
# max_samples: maximum size 0.5=50% of each sample taken from the full dataset
# max_features: maximum of features 1=100% taken here all 10K 
# n_estimators: number of decision trees 
bg=BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=10)
bg.fit(train_in, train_out)
print("score on test: " + str(bg.score(test_in, test_out)))
print("score on train: "+ str(bg.score(train_in, train_out)))


# Bagging Decision Tree is an optimal model

# Boosting Decision Tree

# In[328]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
adb = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=10,max_depth=4),n_estimators=10,learning_rate=0.6)
adb.fit(train_in, train_out)
print("score on test: " + str(adb.score(test_in, test_out)))
print("score on train: "+ str(adb.score(train_in, train_out)))


# Boosting Decision Tree is an overfitting model

# Random Forest

# In[329]:


from sklearn.ensemble import RandomForestClassifier
# n_estimators = number of decision trees
rf = RandomForestClassifier(n_estimators=30, max_depth=9)
rf.fit(train_in, train_out)
print("score on test: " + str(rf.score(test_in, test_out)))
print("score on train: "+ str(rf.score(train_in, train_out)))


# Random Forest is an overfitting model

# Voting Classifier

# In[330]:


from sklearn.ensemble import VotingClassifier
# Let us take top 4 algorithms result to get best out of it
# 1) K-nearest-neighbor = knn
# 2) Bagging Decision Tree = bg
# 3) random forest =rf
# 4) Boosting Decision Tree = adb
evc=VotingClassifier(estimators=[('knn',knn),('bg',bg),('rf',rf),('adb',adb)],voting='hard')
evc.fit(train_in, train_out)
print("score on test: " + str(evc.score(test_in, test_out)))
print("score on train: "+ str(evc.score(train_in, train_out)))


# Voting Classifier is an overfitting model

# Deep Neural Network

# In[341]:


from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import regularizers
from keras import metrics
# add validation dataset
validation_split=10
x_validation=train_in[:validation_split]
x_partial_train=train_in[validation_split:]
y_validation=train_out[:validation_split]
y_partial_train=train_out[validation_split:]
model=models.Sequential()
model.add(layers.Dense(8,kernel_regularizer=regularizers.l2(0.003),activation='relu',input_shape=(10,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8,kernel_regularizer=regularizers.l2(0.003),activation='relu'))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(8,kernel_regularizer=regularizers.l2(0.003),activation='tanh',input_shape=(10,)))
model.add(layers.Dropout(0.7))
model.add(layers.Dense(8,kernel_regularizer=regularizers.l2(0.003),activation='sigmoid',input_shape=(10,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8,kernel_regularizer=regularizers.l2(0.003),activation='relu',input_shape=(10,)))
model.add(layers.Dropout(0.8))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_partial_train,y_partial_train,epochs=1000,batch_size=512,validation_data=(x_validation,y_validation))
print("score on test: " + str(model.evaluate(test_in,test_out)[1]))
print("score on train: "+ str(model.evaluate(train_in,train_out)[1]))


# Neural network is an optimal model

# Neural Network with hyper parameter tuning

# In[ ]:


import tensorflow as tf    
from tensorflow import keras 
from tensorflow.python.keras.metrics import Metric
is_class = True # use True if using classification and False if using regression

if is_class: # if classification
    #List Hyperparameters for MLPClassifier that we want to tune.
    hidden_layer_sizes = [(50,50,50), (50,100,50), (100), (10,20)]
    solver = ['sgd', 'lbfgs', 'adam']
    alpha = [0.0001, 0.05, 0.01]
    activation = ['logistic', 'tanh', 'relu']
    
    #Convert to dictionary
    hyper_mlpclass= dict(hidden_layer_sizes=hidden_layer_sizes, solver=solver, alpha=alpha, activation=activation)
    #Create new MLPCLassifier object
    annc = neural_network.MLPClassifier()
    #Use GridSearch
    clf = GridSearchCV(annc, hyper_mlpclass, n_jobs=-1, cv=10)
    
else: # if regression
    for x in range(1000):
        model = neural_network.MLPRegressor(solver="lbfgs", alpha=.0001, hidden_layer_sizes=1000, activation="tanh")# solver, hidden layer size and activation was modified to get minimum error



model = clf.fit(train_in, train_out) #gridsearchCV classifier implementation

pred_test_out = clf.predict(test_in) #gridsearchCV prediction classifier implementation

pred_train_out = clf.predict(train_in) #gridsearchCV prediction classifier implementation


# In[352]:


print("Testing accuracy:",accuracy_score(test_out, pred_test_out))
print("Training accuracy:",accuracy_score(train_out, pred_train_out))


# Neural network is an optimal model

# XG boost Tree

# In[354]:


regex = re.compile(r"\[|\]|<", re.IGNORECASE)

train_in.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in train_in.columns.values]
xgbcl = XGBClassifier()
xgbcl.fit(train_in,train_out)
y_xgbcl = xgbcl.predict(test_in)
print("XGBoost Accuracy: ", accuracy_score(y_xgbcl,test_out))


# In[365]:


print("score on test: " + str(xgbcl.score(test_in, test_out)))
print("score on train: "+ str(xgbcl.score(train_in, train_out)))


# XG boost Tree is an overfitting model

# Bagging desicion tree gave us the best accuracy and we are using it for plotting confusion matrix

# # Confusion Matrix

# In[364]:


from sklearn.metrics import confusion_matrix # to plot confusion matrix
#Using bagging decision tree model for plotting confusion matrix
y_pred=bg.predict(test_in)
conf_mat = confusion_matrix(test_out, y_pred)
print(conf_mat)
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['Zero', 'One']
make_confusion_matrix(conf_mat, 
                      group_names=labels,
                      categories=categories, 
                      cmap='binary')


# Confusion matrix with 92% accuracy for Bagging Decision Tree

# # Resuslts
Best Features with p-value that can detect clinical anxiety symptoms:



pcm_fftMag_spectralRollOff90.0_sma_linregc2_q1         0.000000e+00

pcm_fftMag_spectralRollOff50.0_sma_de_quartile3_q9     2.595696e-70

pcm_fftMag_spectralRollOff50.0_sma_de_de_iqr1-3_q1     9.568180e-118

pcm_fftMag_spectralMinPos_sma_de_de_quartile2_q1       2.081303e-07

pcm_fftMag_spectralCentroid_sma_de_iqr2-3_q9           1.489952e-121

pcm_fftMag_spectralCentroid_sma_de_iqr2-3_q2           9.438956e-59

pcm_fftMag_spectralCentroid_sma_de_iqr1-3_q10          1.077203e-133

mfcc_sma_de[6]_percentile95.0_q6                       9.896895e-01

FaceEmbeddingDim66.min.q1                              7.182895e-01

FaceEmbeddingDim101.kurtosis.q7                        3.118199e-28
# Ranking of models:
# 
# 
# 1)Bagging Decision Tree:       Accuracy = 92%
# 
# 
# 
# 
# 2.a)K-Nearest-Neighbors:       Accuracy = 84%
# 
# 
# 2.b)Random Forest:             Accuracy = 84%
# 
# 
# 2.c)Votting Classifier:        Accuracy = 84%
# 
# 
# 
# 
# 3.a)Boosting Decision Tree:    Accuracy = 76%
# 
# 
# 3.b)Neural Network(Tuning):    Accuracy = 76%
# 
# 
# 3.c)XG Boost Tree:             Accuracy = 76%
# 
# 
# 
# 
# 4.a)Logistic Regression:       Accuracy = 72%
# 
# 
# 4.b)Support Vector Machine:    Accuracy = 72%
# 
# 
# 4.c)Decision Tree:             Accuracy = 72%
# 

# In[ ]:




