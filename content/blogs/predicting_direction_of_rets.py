#!/usr/bin/env python
# coding: utf-8

# 
# 

# <h1> Import Libraries </h1>

# In[2]:


#Data manipulation
import pandas as pd
import numpy as np

#Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

#Data Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler, Normalizer, normalize
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score

from xgboost import XGBClassifier, plot_importance, Booster, plot_tree, to_graphviz

#Classifier
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
import yfinance as yf

#Feature selection
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Evaluation Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix, auc, roc_curve, plot_roc_curve, f1_score
from sklearn.metrics import mean_squared_error
import scikitplot as skplt

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#Load locally stored data for Johnson & Johnson
df=pd.read_csv('JNJ.csv', index_col=0, parse_dates=True)


# In[4]:


#Check whether there are any missing values
pd.DataFrame(df).isnull().sum()


# In[5]:


#Calculate returns

df['return']=np.log(df['Adj Close']).diff()

df.head()


# <h1> Feature Specification </h1>

# In[6]:


#Specifying features

#Set a counter for the for loop
cnts=5

#1-5 days lagged returns
for cnt in range (1,cnts+1):
    df['ret'+str(cnt)]=df['return'].shift(cnt)
    
#O-C & H-L of t-1
df['O-C']=(df['Open']-df['Close']).shift(1)
df['H-L']=(df['High']-df['Low']).shift(1)

#Momentum 1-5 days
for cnt in range (1,cnts+1):
    #Shift 1 day to avoid data leakge
    df['momentum'+str(cnt)]=(df['Adj Close']-df['Adj Close'].shift(cnt)).shift(1)

#Moving average 5-10-15 days
for i in range (5,16,5):
    #Shift 1 day to avoid data leakage
    df['MA'+str(i)]=(df['Adj Close'].rolling(i).mean()).shift(1)
    
#Exponential moving average 5-10-15 days
for j in range(5,16,5):
    #Shift 1 day to avoid data leakage
    df['EMA'+str(j)]=(df['Adj Close'].ewm(j,adjust=False).mean()).shift(1)


# <h1> Define Label </h1>

# In[7]:


#Define target, +1 if the price has increased (positive return), -1 if it has decreased (negative return)
df['Target']=np.sign(df['return'].values)
#Treat 0 return signs as increase
df['Target']=df['Target'].replace(0,1)
#Drop the columns except features and target
df=df.drop(columns=['Open','High','Low','Close','Volume'],axis=1)


# In[8]:


#Drop Nan values
df.dropna(inplace=True)
#Check output
df.head()


# In[9]:


#Define the features matrix X

X=df.drop(columns=['Adj Close', 'return', 'Target'], axis=1)

X.head()


# In[10]:


#Shape of X
X.shape


# In[11]:


#Define dependent variable
y=df['Target']
occurrences=np.count_nonzero(y==0)
occurrences


# In[12]:


#Shape of y
y.shape


# <p>L1 and L2 penalty and The Effects on Coefficients</p>

# In[13]:


#Demonstrate effect of regularization term on coefficients
l2_range=10**np.linspace(6,-2,100)*0.57
#l1_range = np.linspace(0.01,1,50)
l1_range=10**np.linspace(6,-2,100)*0.57

l2regressioncoef=[]
l1regressioncoef=[]
for i in l2_range:
    l2regg=Pipeline([('scaler', StandardScaler()), ('regressor', Ridge(alpha=i))])
    l2regg.fit(X, y)
    l2regressioncoef.append(l2regg['regressor'].coef_)
for i in l1_range:
    l1regg=Pipeline([('scaler', StandardScaler()), ('regressor', Lasso(alpha=i))])
    l1regg.fit(X,y)
    l1regressioncoef.append(l1regg['regressor'].coef_)
    
fig = plt.figure(figsize=(20,8))
ax = plt.axes()
ax.plot(l1_range, l1regressioncoef)
ax.set_xscale('log')
ax.set_title('Lasso coefficients as a function of the regularization')
ax.set_xlim(ax.get_xlim()[::-1]) 
ax.set_xlabel('$\lambda$')
ax.set_xlim(left=100)
ax.set_ylabel('$\mathbf{w}$');

fig = plt.figure(figsize=(20,8))
ax = plt.axes()
ax.plot(l2_range, l2regressioncoef)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1]) 
ax.set_title('Ridge coefficients as a function of the regularization')
ax.set_xlim(left=100)
ax.set_xlabel('$\lambda$')
ax.set_ylabel('$\mathbf{w}$');


# <p> The Case of Lambda = 0</p>

# In[14]:


#Testing for equivalance of coefficients when lambda=0
testa=Pipeline([('scaler', StandardScaler()), ('regressor', Lasso(alpha=0))])
testa.fit(X,y)
testa['regressor'].coef_


# In[15]:


#Testing for equivalance of coefficients when lambda=0
testb=Pipeline([('scaler', StandardScaler()), ('regressor', Ridge(alpha=0))])
testb.fit(X,y)
testb['regressor'].coef_


# <p> L1 and L2 penalized logistic regressions of JNJ </p>

# In[16]:


#Split Data
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42,shuffle=False)
print(f"Train and test size, respectively: {len(y_train)}, {len(y_test)}")

#Create a function that scales and fits the model
def scale_fit(X_train, y_train,scaler,C,pen,solv):
    #Take input parameters
    scaler=scaler
    X_train=X_train
    y_train=y_train
    C=C
    pen=pen
    solv=solv
    #Fit and train
    logreg1=Pipeline([("scaler",scaler()),("regressor", LogisticRegression(C=C,penalty=pen,solver=solv))])
    logreg=logreg1.fit(X_train, y_train)
    scaledX=scaler().fit_transform(X_train)
    #Create a dictionary for output
    reg_dict={"reg":logreg,"Xs":scaledX}
    return reg_dict


# <u> L2 penalized logistic regression </u>

# In[17]:


#Perform L2 penalization; print out coefficients and R^2
l2reg = scale_fit(X_train,y_train,StandardScaler,1,'l2','liblinear')
L2=l2reg["reg"]
y_pred0=L2.predict(X_test)
print(f"L2 regression train R^2: {L2.score(X_train,y_train):0.4}\n")
print(f"L2 regression test R^2: {L2.score(X_test,y_test):0.4}\n")
print(f"L2 regression coefficients: {L2['regressor'].coef_}\n")
print(f"L2 regression MSE: {mean_squared_error(y_test,y_pred0)}")


# <u> L1 penalized logistic regression </u>

# In[18]:


#Perform L1 penalization; print out coefficients and R^2
l1reg = scale_fit(X_train,y_train,StandardScaler,1,'l1','liblinear')
L1=l1reg["reg"]
y_pred1=L1.predict(X_test)
#Output accuracy
print(f"L1 regression train R^2: {L1.score(X_train,y_train):0.4}\n")
print(f"L1 regression test R^2: {L1.score(X_test,y_test):0.4}\n")
print(f"L1 regression coefficients: {L1['regressor'].coef_}\n")
print(f"L1 regression MSE: {mean_squared_error(y_test,y_pred1)}")

#Plot the differences in coefficients
dfgr=pd.DataFrame(X.columns)
dfgr['L2 regression coefficients']=L2['regressor'].coef_[0]
dfgr['L1 regression coefficients']=L1['regressor'].coef_[0]
dfgr['Coefficients']=pd.DataFrame(X.columns)
dfgr.plot(x="Coefficients",y=['L2 regression coefficients','L1 regression coefficients'],kind='bar',figsize=(13,5), title='Penalty Type and Regression Coefficients',fontsize=10,ylabel='Value')


# <p> L1 penalized regression with MinMaxScale</p>

# In[19]:


#Perform L1 penalization; use minmaxscaling, and print out coefficients and R^2
l1regminmax = scale_fit(X_train,y_train,MinMaxScaler,1,'l1','liblinear')
L1m_m=l1regminmax["reg"]
y_pred2=L1m_m.predict(X_test)
#Output accuracy
print(f"L1 regression MinMaxScaled train R^2: {L1m_m.score(X_train,y_train):0.4}\n")
print(f"L1 regression MinMaxScaled test R^2: {L1m_m.score(X_test,y_test):0.4}\n")
print(f"L1 regression MinMaxScaled coefficients: {L1m_m['regressor'].coef_}\n")
print(f"L1 regression MinMaxScaled MSE: {mean_squared_error(y_test,y_pred2)}")

#Plot for selected features
Xscaled=l1regminmax["Xs"]
a_s=[0,5,6,7,12,15]
head=list(X)
colors=['b','g','r','c','m','y']
for a in range (0,6):
    Xs=Xscaled[:,a_s[a]]
    headings=head[a_s[a]]
    plt.figure(figsize=(5,3))
    plt.hist(Xs, bins='auto', alpha=0.5, color=colors[a])
    plt.xlabel('Value', fontsize=10)
    plt.ylabel('Frequency',fontsize=10)
    plt.title('Scaled_'+ str(headings),fontsize=10)
    plt.xlim([-1,1.5])
    plt.show()


# 
# <p> L1 penalized regression with uniform [0,1] range </p>

# In[20]:


#Perform L1 penalization; scale to [0,1] range
l1normalizer = scale_fit(X_train,y_train,QuantileTransformer,1,'l1','liblinear')
L1n=l1normalizer["reg"]
y_pred3=L1n.predict(X_test)
#Output accuracy, MSE, and coefficients
print(f"L1 [0,1] scaled  regression train R^2: {L1n.score(X_train,y_train):0.4}\n")
print(f"L1 [0,1] scaled regression test R^2: {L1n.score(X_test,y_test):0.4}\n")
print(f"L1 [0,1] scaled regression coefficients: {L1n['regressor'].coef_}\n")
print(f"L1 [0,1] scaled regression MSE: {mean_squared_error(y_test,y_pred3)}")

#Plot for selected features
Xscaled2=l1normalizer["Xs"]
for a in range (0,6):
    Xs=Xscaled2[:,a_s[a]]
    headings=head[a_s[a]]
    plt.figure(figsize=(5,3))
    plt.hist(Xs, bins='auto', alpha=0.5, color=colors[a])
    plt.xlabel('Value', fontsize=10)
    plt.ylabel('Frequency',fontsize=10)
    plt.title('Scaled_'+ str(headings),fontsize=10)
    plt.xlim([-1,1.5])
    plt.show()


# <p>Reporting Changes on Coefficients based on Scaling Type <p>

# In[21]:


#Get the X axis headings
dfgr1=pd.DataFrame(X.columns)
dfgr1['L1 regression StandardScaled coefficients']=L1['regressor'].coef_[0]
dfgr1['L1 regression MinMaxScaled coefficients'] =L1m_m['regressor'].coef_[0]
dfgr1['L1 regression uniform [0,1] scaled coefficients'] =L1n['regressor'].coef_[0]
dfgr1['Coefficients']=pd.DataFrame(X.columns)
dfgr1.plot(x="Coefficients",y=['L1 regression StandardScaled coefficients','L1 regression MinMaxScaled coefficients','L1 regression uniform [0,1] scaled coefficients'],kind='bar',figsize=(13,5), title='Scaling Type & Coefficients',fontsize=10,ylabel='Value')
plt.legend(loc='upper right',fontsize=8)
plt.show()


# <p>Feature Selection </p>

# In[22]:


#Feature Selection with f-test
from sklearn.feature_selection import f_regression, SelectKBest, SelectPercentile, f_classif

#Show f-scores of each feature, take k=5 for an example
method=SelectKBest(f_classif, k=5)
method.fit(X,y)
method.get_support(indices=True)
for f,s in zip(X.columns,method.scores_):
    print(f'F-score: {s:0.4} for feature {f}')


# In[23]:


#Create a dictionary to calibrate the number of features based on f-score
kselectordict={0:"ret1",1:"ret2",2:"ret3",3:"ret4",4:"ret5",5:"O-C",6:"H-L",7:"momentum1",8:"momentum2",9:"momentum3",10:"momentum4",11:"momentum5",12:"MA5",13:"MA10",14:"MA15",15:"EMA5",16:"EMA10",17:"EMA15"}
#Conduct selection based on f-statistic and return the feature matrix
def k_selector(X,y,kselectordict,n):
    X_method2=pd.DataFrame()
    X=X
    y=y
    method2=SelectKBest(f_regression,k=n)
    method2.fit(X,y)
    keys=method2.get_support(indices=True)
    #For the chosen features, create a new feature matrix and assign the correct values
    for key in keys:
        X_method2[kselectordict[key]]=X[kselectordict[key]]
    #Return the new feature matrix 
    return X_method2


# In[24]:


#Optimizing
#Create an array to calibrate the hyperparameters C and k
storage=np.array([[0,0,0]])
#Create a counter for iteration to find the optimal C and k that minimise MSE
#For each k
for n in range(3,15,1):
    X_method2=k_selector(X,y,kselectordict,n)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_method2,y,test_size=0.2,random_state=42,shuffle=False)
    #For each C between 0.01 and 1 C
    for i in range(1,101,1):
        reducedvar=scale_fit(X_train2,y_train2,MinMaxScaler,i/100,'l2','liblinear')
        redvar=reducedvar["reg"]
        y_pred5=redvar.predict(X_test2)
        mse=mean_squared_error(y_test2,y_pred5)
        mse_i=np.vstack([[mse,i/100,n]])
        #Stack mse, C, and k highest f scored features
        storage=np.vstack((storage,mse_i))

#Delete the inital 0 row
storage=np.delete(storage,(0),axis=0)
#Rounding
storage=np.round(storage,4)
#Find the minimum MSE in the iterations with different C and k
min_mse=np.amin(storage,axis=0)
min_mse


# In[25]:


#Create a counter to find the index of the lowest MSE model 
i=0
j=0
#Iterate the 1200 rows of storage array
while i<1200:
    a=storage[:,0][i]
    if a==1.8048:
        j=i
        i=i+1
    else:
        i=i+1
#The MSE, C, and k for the lowest MSE model
m=storage[j]
m


# In[26]:


#Rerun the optimal model to print out coefficients, MSE, and R^2
X_reducedvar=k_selector(X,y,kselectordict,int(m[2]))
X_trainrv, X_testrv, y_trainrv, y_testrv = train_test_split(X_reducedvar,y,test_size=0.2,random_state=42,shuffle=False)
#Fit and train using local scale_fit function
reducedvar=scale_fit(X_trainrv,y_trainrv,MinMaxScaler,m[1],'l2','liblinear')
redvar=reducedvar["reg"]
#Predict
y_predrv=redvar.predict(X_testrv)
print(f"L2 regression lower variance test R^2: {redvar.score(X_testrv,y_testrv):0.4}\n")
print(f"L2 regression lower variance MSE: {mean_squared_error(y_testrv,y_predrv)}\n")
print(f"These are all of the coefficients of the lowest MSE model:\n")
#For each feature, print out the coefficient
for i in range(0,int(m[2]),1):
    print(f"L2 regression lower variance coefficient of {list(X_reducedvar.columns)[i]} is: {redvar['regressor'].coef_[0][i]}\n")


# <p>Plotting the Logistic Sigmoids</p>

# In[27]:


#Define sigmoid function
def logistic_sigmoid(xb):
    return (1/(1+np.exp(-xb)))
#Merge prediction and features of the test data points
X_testreducedvar=X_testrv
X_testreducedvar["logit_pred"]=y_predrv

#Plot sigmoid with respect to the chosen feature
def logistic_plotter(XMin,XMax,featurename,featurecoefindex):
    plt.clf()
    fig,ax=plt.subplots(figsize=(10,5))
    #Plot clusters of observations
    ax.scatter(X_testreducedvar[featurename],X_testreducedvar["logit_pred"], c=X_testreducedvar["logit_pred"],zorder=20)
    #Plot the sigmoid 
    X_Sim=np.linspace(-2.5,2.5,3000)
    Y_Loss=logistic_sigmoid(X_Sim*redvar['regressor'].coef_[0][featurecoefindex]+redvar['regressor'].intercept_)
    #Add coloring, titles, and ranges
    ax.plot(X_Sim,Y_Loss, color='orange',linewidth=3)
    plt.ylabel('Class Indicator 1 or -1',fontsize=10)
    plt.xlabel('Wrt to feature_'+str(featurename),fontsize=10)
    plt.yticks(np.arange(-1,1.1,0.25),fontsize=8)
    plt.legend(('Logistic Regression',),loc="upper right",fontsize=7)
    ax.set_title('Logistic Sigmoid of_'+str(featurename),fontsize=10)
    plt.ylim(-1.25,1.25)
    plt.xlim(XMin,XMax)
    #Return the plot
    return ax


# In[28]:


logistic_plotter(0.75,-0.75,'ret1',0)


# In[29]:


logistic_plotter(-3.5,3.5,'momentum1',1)


# In[30]:


logistic_plotter(-3.5,3.5,'momentum2',1)


# <p> Evaluating the Low-Variance Model </p>

# In[31]:


#Evaluation metrics for the restricted model redvar
#prediction=ypredrv, Xtest=X_testrv
#Plot the confusion matrix
#Sigmoid plotting has ended, drop the logit pred column from X
X_testrv=X_testrv.drop(columns=['logit_pred'],axis=1)


# In[32]:


fig,ax=plt.subplots(figsize=(10,8))
plot_confusion_matrix(redvar,X_testrv,y_testrv, ax=ax,cmap='Blues')
plt.title('Confusion Matrix',fontsize=15)
plt.grid(False)


# In[33]:


#Classification report
print(classification_report(y_testrv,y_predrv))
f_measure=f1_score(y_testrv,y_predrv,'binary')
f_measure


# In[34]:


probs=redvar.predict_proba(X_testrv)

preds1 = probs[:, 0]
preds2 = probs[:, 1]

fpr1, tpr1, threshold1 = roc_curve(y_testrv, preds1, pos_label=-1)
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, threshold2 = roc_curve(y_testrv, preds2, pos_label=1)
roc_auc2 = auc(fpr2, tpr2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

ax[0].plot([0, 1], [0, 1], 'r--')
ax[0].plot(fpr1, tpr1, 'cornflowerblue', label=f'ROC Curve of Class -1 Area = {roc_auc1:0.3}')
ax[0].set_title("Receiver Operating Characteristic for Down Moves")
ax[0].set_xlabel('False Positive Rate',fontsize=13)
ax[0].set_ylabel('True Positive Rate',fontsize=13)

ax[1].plot([0, 1], [0, 1], 'r--')
ax[1].plot(fpr2, tpr2, 'g', label=f'ROC Curve of Class 1 Area = {roc_auc2:0.3}')
ax[1].set_title("Receiver Operating Characteristic for Up Moves")
ax[1].set_xlabel('False Positive Rate',fontsize=13)
ax[1].set_ylabel('True Positive Rate',fontsize=13)

# Define legend
ax[0].legend(), ax[1].legend();

