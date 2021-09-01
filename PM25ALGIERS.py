import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
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

from datetime import datetime
# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
#%%
dt_string="01-09-2021-09-07-35"
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

    #df[' pm25'].interpolate(method='nearest', inplace=True)

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
    
    models_mae=np.array([
    mean_absolute_error(y_val,DTC.predict(X_val)),
    mean_absolute_error(y_val,AdaBoostDT.predict(X_val)),
    mean_absolute_error(y_val,mlp.predict(X_val)),
    mean_absolute_error(y_val,rndmForest.predict(X_val)),
    mean_absolute_error(y_val,xgb.predict(X_val)),
    mean_absolute_error(y_val,svmModel.predict(X_val)),
    mean_absolute_error(y_val,LightGBM.predict(X_val)),
    mean_absolute_error(y_val,ctmodel.predict(X_val)),
    mean_absolute_error(y_val,dummy_clf.predict(X_val))])
    
    modelNames=['DecisionTree',
                'AdaBoost',
                'MLP',
                'RandomForest',
                'XGB',
                'SVM',
                'LightGBM',
                'catBoost',
                'Dummy Classifier']
    
    model_collection={'DecisionTree':DTC,
                'AdaBoost':AdaBoostDT,
                'MLP':mlp,
                'RandomForest':rndmForest,
                'XGB':xgb,
                'SVM':svmModel,
                'LightGBM':LightGBM,
                'catBoost':ctmodel,
                'Dummy Classifier':dummy_clf}
    
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
    return pd.DataFrame([modelNames,modelsrmse,models_r2,models_mae]),model_collection

def get_builtIn_features_importance(modelnameStr):
    tt=np.vstack([df.columns,model_collection[modelnameStr].feature_importances_])
    return pd.DataFrame(tt.T)


for lag in range(28,31):
    #lag=3
    print("lags value is "+str(lag))
    print('Loading data')
    df=loadData()
    print('Cleaning Data removing unused features')
    #df.drop(columns=["TOTAL_SNOW_MM"],inplace=True)
    #df['OPINION'] =df['OPINION'].astype('category')
    #df['OPINION']=df['OPINION'].cat.codes
    df=cleanData(df=df)
    df.rename(columns = {' pm25':'y'}, inplace=True)
    y=df.y
    df['PM25_t']=y
    #form a vector with only PM25 peaks of the "window" size days
    y=shiftWeekllyMax(y,window=7)
    #Wat lagged value should be introduced as inputs.
    df=shifData(df,STEPS=lag)
    y=np.array(y)
    df.drop(columns=['y'],inplace=True)
    X=df[0:y.shape[0]]
    y_val=y[500:600]
    X_val=X[500:600]
    y_train=y[0:500]
    X_train=X[0:500]
    performance,model_collection=trainModels(df)
    performance['lag']=lag
    if lag==1:
        performance.to_csv("ALL_Metrics"+dt_string+".csv",sep=';',mode='w')
    else:
        performance.to_csv("ALL_Metrics"+dt_string+".csv",sep=';',mode='a')
    
    def save_y_hatForAllModels(model_collection):
        res=pd.DataFrame()
        res['y_truth']=y_val
        for cle, model in model_collection.items():
            y_hat=model.predict(X_val)
            res[cle]=y_hat
        res.to_csv("y_hat_"+str(lag)+".csv",sep=';')
    
    save_y_hatForAllModels(model_collection)
    df_featuresImportance=get_builtIn_features_importance('RandomForest')
    df_featuresImportance.to_csv(str(lag)+"FIMP.csv",sep=';')
#%%
# model = RandomForestRegressor()
# param_search = {'max_depth' : [3, 5,8, 10,16,50,60,80,100]}
# tscv = TimeSeriesSplit(n_splits=3)
# gsearch = GridSearchCV(estimator=model, scoring="r2", cv=tscv,
#                          param_grid=param_search,verbose=4)
# gsearch.fit(X,y)
#%%
#y_pred=gsearch.predict(X_val)
#plt.plot(y_val)
#plt.plot(y_pred)
#print(rmse(y_val,y_pred))
#print(r2_score(y_val,y_pred))
#%%
#plt.plot(y_val)
#plt.plot(y_pred)
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
#%%
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
#%%
plot_acf(df.PM25_t,lags=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
plt.figsize=(6, 4)
plt.xlabel('lags in days')
plt.ylabel('correlation')
plt.title('Autocorrelation of PM2.5 in Algiers')
plt.show()
#%%
import statsmodels.api as sm
#%%
gdp_cycle, gdp_trend = sm.tsa. filters.hpfilter(df.PM25_t)
#%%
gdp_decomp = df[['PM25_t']].copy()
gdp_decomp["cycle"] = gdp_cycle
gdp_decomp["trend"] = gdp_trend
#%%
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
gdp_decomp[["trend"]].plot(ax=ax, fontsize=16);
legend = ax.get_legend()
legend.prop.set_size(20)
#%%
from genetic_selection import GeneticSelectionCV
estimator = DecisionTreeRegressor()
selector = GeneticSelectionCV(estimator,
                              cv=5,
                              verbose=1,
                              scoring="accuracy",
                              max_features=5,
                              n_population=50,
                              crossover_proba=0.5,
                              mutation_proba=0.2,
                              n_generations=10,
                              crossover_independent_proba=0.5,
                              mutation_independent_proba=0.05,
                              tournament_size=3,
                              n_gen_no_change=10,
                              caching=True,
                              n_jobs=-1)
selector = selector.fit(X, y[:723])
print(selector.support_)

