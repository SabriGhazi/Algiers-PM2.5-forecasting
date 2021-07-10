import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math as math
from sklearn import svm
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from lightgbm import LGBMRegressor
import catboost as cb
from missingpy import KNNImputer
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from sklearn.preprocessing import MinMaxScaler


def loadData():
    idx=pd.date_range(start="2019/4/25", end="2021/6/30",freq="1D")
    df=pd.read_csv("algiers-us embassy-air-quality.csv",
                   sep=',',
                   parse_dates=["date"],
                   header=0,
                   index_col="date")
    dfClimatic=pd.read_csv("climatic-algiers.csv",
                   sep=',',
                   parse_dates=["DATE"],
                   header=0,
                   index_col="DATE"
                   )
    df.sort_values(by='date',inplace=True, ascending=True)
    dfClimatic.sort_values(by='DATE', inplace=True,ascending=True)
    df=df.reindex(idx,fill_value=np.nan)
    dfClimatic=dfClimatic.reindex(idx,fill_value=np.nan)
    imputer = KNNImputer(n_neighbors=8, weights="uniform")
    imputer.fit(df)
    cleanedX=imputer.transform(df)
    df[' pm25']=cleanedX
    df = pd.concat([df,dfClimatic], axis=1)
    return df

def displayCorrelation(df):
    corr = df.corr()
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)

def cleanData(df):
    df.drop(columns=["OPINION",
                             "MIN_TEMPERATURE_C",
                             'WEATHER_CODE_NOON', 
                             'WEATHER_CODE_EVENING', 
                             'TOTAL_SNOW_MM',
                             'HEATINDEX_MAX_C',
                             'DEWPOINT_MAX_C',
                             'PRESSURE_MAX_MB',
                             'TEMPERATURE_MORNING_C', 
                             'TEMPERATURE_NOON_C', 
                             'TEMPERATURE_EVENING_C',
                             'WEATHER_CODE_MORNING',
                             'CLOUDCOVER_AVG_PERCENT',
                             'UV_INDEX',
                             'VISIBILITY_AVG_KM',
                             'WINDTEMP_MAX_C',
                             'SUNHOUR'],inplace=True)
    df.dropna(inplace=True)
    df.rename(columns = {' pm25':'y'}, inplace=True)
    return df

def deleteCorrelatedColumns(df,minCorrelationToKeep):
    """Drop highly correlated features"""
    corrMartrix = df.corr().abs()
    upper_tri = corrMartrix.where(
        np.triu(np.ones(corrMartrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns 
               if any(upper_tri[column] > minCorrelationToKeep)]
    print('Columns to drop ', len(to_drop))
    df.drop(to_drop, inplace=True, axis='columns')
    print("new shape of the dataset:", df.shape)
    return df

def DropFeaturesUncorrelatedWithTarget(df,minUnCorrelation=0.50):
    """Drop Features which are not correlated with the target"""
    dfCor = pd.DataFrame(df.drop("y", axis=1).apply(lambda x: x.corr(df.y)))
    dfCor.columns = ['val']
    dfCor.val = dfCor.abs()
    dfCor.sort_values(axis=0, inplace=True, by='val', ascending=False)
    columnsToKeep = dfCor[dfCor.val > minUnCorrelation].index
    df = df[columnsToKeep]
    print("After Droping uncorrelated features the new shape is",df.shape)
    return df

def rmse(y_val,y_pred):
    return math.sqrt(mean_squared_error(y_val,y_pred))

def shiftWeekllyMax(y,window=5):
    ret=[]
    for i in range(y.shape[0]):
        if(i+window<y.shape[0]):
            temp=y[i:i+window]
            ret.append(max(temp))
    return ret

def shifData(df,STEPS=5):
    for i in range(1, STEPS):
        col_name = '_T_{}'.format(i)
        df['MAX_TEMPERATURE_C'+col_name] = df['MAX_TEMPERATURE_C'].shift(periods=-1 * i)
        df['WINDSPEED_MAX_KMH'+col_name] = df['WINDSPEED_MAX_KMH'].shift(periods=-1 * i)
        df['PRECIP_TOTAL_DAY_MM'+col_name] = df['PRECIP_TOTAL_DAY_MM'].shift(periods=-1 * i)
        df['HUMIDITY_MAX_PERCENT'+col_name] = df['HUMIDITY_MAX_PERCENT'].shift(periods=-1 * i)
        df['PM25_t'+col_name] = df['PM25_t'].shift(periods=-1 * i)
    df = df.dropna()
    return df

def trainModels(df):
    """train model and show thier accuracy."""
    #Train a Decision Tree
    rng = np.random.RandomState(1)
    DTC = DecisionTreeRegressor(max_depth=100)
    DTC.fit(X_train, y_train)
    
    #Train an AdaBoost with Decision Tree
    rng = np.random.RandomState(1)
    AdaBoostDT = AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=100), n_estimators=100, random_state=rng)
    AdaBoostDT.fit(X_train, y_train)

    #Train a MLP model
    mlp = MLPRegressor(alpha=1e-5,
                        hidden_layer_sizes=(300, 100),
                        random_state=1,
                        activation="relu",
                        max_iter=450)
    mlp.fit(X_train, y_train)
    
    # in case where xgboost isn't installed.
    #!pip install xgboost   
    #Train a XGBoost Classifier
    xgb = XGBRegressor(n_estimators = 100)
    xgb.fit(X_train, y_train)
    #Train a Random Forest
    rndmForest = RandomForestRegressor()
    rndmForest.fit(X_train, y_train)
    
    #Train an SVM model
    svmModel = svm.SVR()
    svmModel.fit(X_train, y_train)
    # this dummy model is just used for purpose of comparison with other classifier.
    dummy_clf = DummyRegressor(strategy="mean")
    dummy_clf.fit(X_train, y_train)
    
    #clfLR = LogisticRegression().fit(X_train, y_train)    
    
    LightGBM  = LGBMRegressor()
    LightGBM.fit(X_train, y_train)
    
    
    ctmodel = cb.CatBoostRegressor( eval_metric='RMSE',verbose=0)
    train_dataset = cb.Pool(X_train,y_train)
    test_dataset = cb.Pool(X_val,y_val)
    grid = {'learning_rate': [0.03, 0.1],
    'depth': [4, 6, 10,20, 30],
    'l2_leaf_reg': [1, 3, 5, 8, 10],
    'iterations': [50, 100, 150,200],}
    ctmodel.grid_search(grid,X_train,y_train)
    
    modelsrmse=np.array([
    rmse(y_val,DTC.predict(X_val)),
    rmse(y_val,AdaBoostDT.predict(X_val)),
    rmse(y_val,mlp.predict(X_val)),
    rmse(y_val,rndmForest.predict(X_val)),
    rmse(y_val,xgb.predict(X_val)),
    rmse(y_val,svmModel.predict(X_val)),
    rmse(y_val,LightGBM.predict(X_val)),
    rmse(y_val,ctmodel.predict(X_val)),
    rmse(y_val,dummy_clf.predict(X_val))])
    
    models_r2=np.array([
    r2_score(y_val,DTC.predict(X_val)),
    r2_score(y_val,AdaBoostDT.predict(X_val)),
    r2_score(y_val,mlp.predict(X_val)),
    r2_score(y_val,rndmForest.predict(X_val)),
    r2_score(y_val,xgb.predict(X_val)),
    r2_score(y_val,svmModel.predict(X_val)),
    r2_score(y_val,LightGBM.predict(X_val)),
    r2_score(y_val,ctmodel.predict(X_val)),
    r2_score(y_val,dummy_clf.predict(X_val))])
    
    modelNames=['DecisionTree',
                'AdaBoost',
                'MLP',
                'RandomForest',
                'XGB',
                'SVM',
                'LightGBM',
                'catBoost',
                'Dummy Classifier']
    
    plt.figure(figsize=(10, 4))
    plt.bar(modelNames, modelsrmse, align='center') # A bar chart
    plt.title('Performances of Model to predict PM_2.5 cocnentration prediction')
    plt.xlabel('Model')
    plt.grid(True)
    plt.ylabel('RMSE')
    plt.show()
#
    print('------------------------------------------------------------------')
    print('Models Accuracy   ')
    print("DecisionTree", "{:.2f}".format(rmse(y_val,DTC.predict(X_val))))
    print("AdaBoost Decition Tree", "{:.2f}".format(rmse(y_val,AdaBoostDT.predict(X_val))))
    print("MLP", "{:.2f}".format(rmse(y_val,mlp.predict(X_val))))
    print("RandomForestClassifier ",  "{:.2f}".format(rmse(y_val,rndmForest.predict(X_val))))
    print("XGB ",  "{:.2f}".format(rmse(y_val,xgb.predict(X_val))))
    print("SVM: ",  "{:.2f}".format(rmse(y_val,svmModel.predict(X_val))))
    print("LightGBM: ",  "{:.2f}".format(rmse(y_val,LightGBM.predict(X_val))))
    print("catBoost: ",  "{:.2f}".format(rmse(y_val,ctmodel.predict(X_val))))
    print("Dummy Classifier :",  "{:.2f}".format(rmse(y_val,dummy_clf.predict(X_val))))
    print("------------------------------------------------------------------")
    return pd.DataFrame([modelNames,modelsrmse,models_r2])
#for lag in range(9,21):
lag=1
print("lags value is "+str(lag))
print('Loading data')
df=loadData()
print('Cleaning Data removing unused features')
df=cleanData(df=df)
y=df.y
df['PM25_t']=y
#form a vector with only PM25 peaks of the "window" size days
y=shiftWeekllyMax(y,window=7)
#Wat lagged value should be introduced as inputs.
#df=shifData(df,STEPS=lag)
y=np.array(y)
df.drop(columns=['y'],inplace=True)
X=df[0:y.shape[0]]
y_val=y[680:X.shape[0]]
X_val=X[680:X.shape[0]]
y_train=y[0:680]
X_train=X[0:680]
performance=trainModels(df)
performance.to_csv(str(lag)+"RES.csv",sep=';')
# model = RandomForestRegressor()
# param_search = {'max_depth' : [3, 5,8, 10,16,50,60,80,100]}
# tscv = TimeSeriesSplit(n_splits=3)
# gsearch = GridSearchCV(estimator=model, scoring="r2", cv=tscv,
#                          param_grid=param_search,verbose=4)
# gsearch.fit(X,y)
#%%
y_pred=gsearch.predict(X_val)
plt.plot(y_val)
plt.plot(y_pred)
print(rmse(y_val,y_pred))
print(r2_score(y_val,y_pred))
#%%
plt.plot(y_val)
plt.plot(y_pred)
#%%
plt.plot(df.PM25_t.resample("15D").max())
plt.plot(df.MAX_TEMPERATURE_C.resample("15D").max())
plt.plot(df.PRECIP_TOTAL_DAY_MM.resample("15D").max())
#%%
mask = (df.index > '2020-05-20') & (df.index <= '2020-05-30')
mask1 = (df.index > '2021-03-01') & (df.index <= '2021-03-31')
xx=df.loc[mask]
#plt.plot(df.loc[mask1].y)
#%%
date_form = DateFormatter("%Y-%m")
fig, ax = plt.subplots(figsize=(15, 12))
ax.xaxis.set_major_formatter(date_form)
plt.plot(df.PM25_t.resample("15D").max())
#%%
rng = np.random.RandomState(1)
DTC = DecisionTreeRegressor(max_depth=120,criterion="mae")
DTC.fit(X_train, y_train)
y_pred=DTC.predict(X_val)
print(rmse(y_val, y_pred))