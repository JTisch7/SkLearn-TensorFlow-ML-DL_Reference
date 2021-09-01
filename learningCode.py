# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 15:55:47 2020

@author: Jonathan
"""
import os
import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
import numpy as np
import pandas as pd
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)





  
'''creating data set'''
#import data
dfVix = pd.read_csv('C:/Users/Jonathan/.spyder-py3/data/VIX_ALL_yahoo.csv')
dfSpy = pd.read_csv('C:/Users/Jonathan/.spyder-py3/data/SPY_ALL_yahoo.csv')
dfVix = dfVix[779:].reset_index(drop=True)

#create/merge dataframes
dfComb = dfSpy.copy()
dfComb['vixOpen'], dfComb['vixClose'], dfComb['vixHigh'], dfComb['vixLow'] = dfVix['Open'], dfVix['Close'], dfVix['High'], dfVix['Low']

#create return-based features
dfComb['returnDuringDay'] = (dfComb['Close']-dfComb['Open'])/dfComb['Open']
dfComb['highDuringDay'] = (dfComb['High']-dfComb['Open'])/dfComb['Open']
dfComb['lowDuringDay'] = (dfComb['Open']-dfComb['Low'])/dfComb['Open']
dfComb['vixReturnDuringDay'] = (dfComb['vixClose']-dfComb['vixOpen'])/dfComb['vixOpen']
dfComb['vixHighDuringDay'] = (dfComb['vixHigh']-dfComb['vixOpen'])/dfComb['vixOpen']
dfComb['vixLowDuringDay'] = (dfComb['vixOpen']-dfComb['vixLow'])/dfComb['vixOpen']
dfComb['overnightReturn'] = (dfComb['Open'].shift(-1)-dfComb['Close'])/dfComb['Close']
dfComb['vixOvernightReturn'] = (dfComb['vixOpen'].shift(-1)-dfComb['vixClose'])/dfComb['vixClose']
dfComb['volReturn'] = (dfComb['Volume']-dfComb['Volume'].shift(1))/dfComb['Volume'].shift(1)
dfComb = dfComb.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'vixOpen', 'vixClose', 'vixHigh', 'vixLow'], axis=1)

#create targets
dfComb['spyTarget'] = dfComb['returnDuringDay'].shift(-1)
dfComb['vixTarget'] = dfComb['vixReturnDuringDay'].shift(-1)
dfComb = dfComb.dropna()
dfComb.loc[(dfComb['spyTarget'] >= 0), 'spyTargetBin'] = 1
dfComb.loc[(dfComb['vixTarget'] >= 0), 'vixTargetBin'] = 1
dfComb.loc[(dfComb['spyTarget'] < 0), 'spyTargetBin'] = 0
dfComb.loc[(dfComb['vixTarget'] < 0), 'vixTargetBin'] = 0
    





'''quick look at data'''
dfComb.head()
dfComb.info()
dfComb['spyTargetBin'].value_counts()
dfComb['vixTargetBin'].value_counts()
dfComb.describe()
dfComb.hist(bins=100, figsize=(30,20))






'''sampling methods'''
#train_test_split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dfComb, test_size=0.2, random_state=42)

#sampling data using stratified method(keep test set rep of various categories of a feature)
dfComb["returnDuringDay"].hist(bins=100)
dfComb["returnDuringDay_cat"] = pd.cut(dfComb["returnDuringDay"],
                               bins=[-np.inf, -0.05, -0.025, 0., 0.025, 0.05, np.inf],
                               labels=[1, 2, 3, 4, 5, 6])
dfComb["returnDuringDay_cat"].value_counts()
dfComb["returnDuringDay_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dfComb, dfComb["returnDuringDay_cat"]):
    strat_train_set = dfComb.iloc[train_index]
    strat_test_set = dfComb.iloc[test_index]
    
strat_test_set["returnDuringDay_cat"].value_counts() / len(strat_test_set)






'''REGRESSION/CLASSIFICATION SPLIT HERE'''





'''data exploration'''
practice = strat_train_set.copy()

#historgram
practice.hist(bins=100, figsize=(30,20))

#cool plotting
practice.plot(kind="scatter", x="returnDuringDay", y="vixReturnDuringDay", alpha=0.2,
    s=practice["volReturn"], label="volReturn", figsize=(10,7),
    c="overnightReturn", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()

#correlation matrix
corr_matrix = practice.corr()

#correlation lists from matrix
corr_matrix["spyTarget"].sort_values(ascending=False)
corr_matrix["vixTarget"].sort_values(ascending=False)

#scatter matrix
from pandas.plotting import scatter_matrix
attributes = ["spyTarget", "vixTarget", "returnDuringDay",
              "vixReturnDuringDay"]
scatter_matrix(practice[attributes], figsize=(12, 8))

#scatter plot
practice.plot(kind="scatter", x="vixTarget", y="returnDuringDay", alpha=0.05)
plt.axis([-0.4, 0.4, -0.05, 0.05])






'''data preparation / Build Pipeline'''
practiceSpy = strat_train_set.drop(['Date', 'spyTargetBin', 'vixTargetBin', 'vixTarget', 'spyTarget'], axis=1)
practiceVix = practiceSpy.copy()
practiceSpy_Labels = strat_train_set['spyTarget'].copy()
practiceVix_Labels = strat_train_set['vixTarget'].copy()

#check for and fix NA values in 3 different ways
practiceSpy.info()
sample_incomplete_rows = practiceSpy[practiceSpy.isnull().any(axis=1)].head()

sample_incomplete_rows.dropna(subset=["returnDuringDay"])  #drop rows
sample_incomplete_rows.drop("returnDuringDay", axis=1)  #drop columns
median = practiceSpy["returnDuringDay"].median()  #replace values
sample_incomplete_rows["returnDuringDay"].fillna(median, inplace=True) 

#use imputer to replace missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(practiceSpy)
imputer.statistics_
practiceSpy.median().values
imputer.strategy
X = imputer.transform(practiceSpy)  #or fit_transform
practiceSpy_train = pd.DataFrame(X, columns=practiceSpy.columns, index=practiceSpy.index)

#Ordinal Encoder - convert from category to number 
from sklearn.preprocessing import OrdinalEncoder
return_cat = practiceSpy_train[["returnDuringDay_cat"]]
return_cat.head(10)

ordinal_encoder = OrdinalEncoder()
return_cat_encoded = ordinal_encoder.fit_transform(return_cat)
return_cat_encoded[:10]
ordinal_encoder.categories_

#One-Hot encoder
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
return_cat_1hot = cat_encoder.fit_transform(return_cat)
return_cat_1hot
return_cat_1hot.toarray()
#OR
cat_encoder = OneHotEncoder(sparse=False)
return_cat_1hot = cat_encoder.fit_transform(return_cat)
return_cat_1hot
cat_encoder.categories_

#Custom Transformer for pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, spy_as_target=False, vix_as_target=False): # no *args or **kargs
        self.spy_as_target = spy_as_target
        self.vix_as_target = vix_as_target
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        if self.spy_as_target:
            vixReturnPerSpyReturn = X[:, 2] / (X[:, 5]+0.01)
            X = np.delete(X, 4, axis=1)
            return np.c_[X, vixReturnPerSpyReturn]
        else:
            return np.c_[X]

attr_adder = CombinedAttributesAdder(spy_as_target=True)
pract_extra_attribs = attr_adder.transform(practiceSpy.values)

#Pipeline for numerical columns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

practiceSpy_num = practiceSpy.drop('returnDuringDay_cat', axis=1)
practice_num_tr = num_pipeline.fit_transform(practiceSpy_num)

#Pipeline for combined Numerical/Categorical columns
from sklearn.compose import ColumnTransformer

num_attribs = list(practiceSpy_num.columns)
cat_attribs = ["returnDuringDay_cat"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

practiceSpy_prepared = full_pipeline.fit_transform(practiceSpy)

#Full pipeline with prediction
from sklearn.linear_model import LinearRegression
full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])

full_pipeline_with_predictor.fit(practiceSpy, practiceSpy_Labels)
full_pipeline_with_predictor.predict(practiceSpy)






'''REGRESSION - build/train models with Cross Validation'''
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

practiceSpy_prepared = full_pipeline.fit_transform(practiceSpy)

#function to train models
def regressionFunc(estimator, X=practiceSpy_prepared, y=practiceSpy_Labels):
    est = estimator
    est.fit(X,y)
    pred = est.predict(X)
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, pred)
    scores = cross_val_score(est, X, y, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)
    return rmse, mae, est

#training different models
regressionFunc(LinearRegression())
regressionFunc(DecisionTreeRegressor(random_state=(42)))
regressionFunc(RandomForestRegressor(n_estimators=10, random_state=42))
regressionFunc(SVR(kernel="linear", epsilon=1.5))
regressionFunc(SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma="scale"))
regressionFunc(LinearSVR(epsilon=1.5, random_state=42))
regressionFunc(SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42))
regressionFunc(GradientBoostingRegressor(max_depth=2, n_estimators=120, learning_rate=0.1, random_state=42))

#Polynomial Regression
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(practiceSpy_prepared)
X_poly[0]

regressionFunc(LinearRegression(),X=X_poly)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, practiceSpy_Labels)
lin_reg.intercept_, lin_reg.coef_

'''Regularized REGRESSION models'''
from sklearn.linear_model import Ridge, Lasso, ElasticNet

#Ridge 
regressionFunc(Ridge(alpha=1, solver="cholesky", random_state=42))
regressionFunc(Ridge(alpha=1, solver="sag", random_state=42))
regressionFunc(SGDRegressor(penalty="l2", max_iter=1000, tol=1e-3, random_state=42))

#Lasso
regressionFunc(Lasso(alpha=0.1))
regressionFunc(SGDRegressor(penalty="l1", max_iter=1000, tol=1e-3, random_state=42))

#ElasticNet
regressionFunc(ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))
regressionFunc(SGDRegressor(penalty="elasticnet", max_iter=1000, tol=1e-3, random_state=42))

#Polynomial Regression with Ridge Regularization
model = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("std_scaler", StandardScaler()),
    ("regul_reg", Ridge(alpha=1, solver="cholesky", random_state=42)),
    ])
model.fit(practiceSpy, practiceSpy_Labels)
model.predict([practiceSpy.iloc[0]])
#OR
regressionFunc(model)





 








'''Fine Tuning and finalizing'''
#Grid Search
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(practiceSpy_prepared, practiceSpy_Labels)

grid_search.best_params_
grid_search.best_estimator_

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
pd.DataFrame(grid_search.cv_results_)

#Randomized Grid Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(practiceSpy_prepared, practiceSpy_Labels)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
#Feature Importance using Random Forest Regressor
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

#choose best model and check performance on test set
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop(['Date', 'spyTargetBin', 'vixTargetBin', 'vixTarget', 'spyTarget'], axis=1)
y_test = strat_test_set['spyTarget'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
    
#confidence interval
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))  

#Save/reload models
import joblib

joblib.dump(final_model, 'myPracModel.pkl')
my_model_loaded = joblib.load('myPracModel.pkl')

#Plot learning curves
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14)              # not shown
    
X = dfComb.drop(['Date', 'spyTargetBin', 'vixTargetBin', 'vixTarget', 'spyTarget', 'returnDuringDay_cat'], axis=1)
y = dfComb['spyTarget'] 
   
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
plt.axis([0, 5000, 0, .02])                         # not shown in the book

#Early Stopping
from copy import deepcopy

poly_scaler = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("std_scaler", StandardScaler())
    ])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                       penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)

minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train)  # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = deepcopy(sgd_reg)


















'''CLASSIFICATION CLASSIFICATION CLASSIFICATION '''

#setup binary targets and feature datasets from train_test_split 
spyTargetTrain_Bin = (train_set['spyTarget'] >= 0)
spyTargetTest_Bin = (test_set['spyTarget'] >= 0)
train_set_noTarget = train_set.copy().drop(['Date', 'spyTargetBin', 'vixTargetBin', 'vixTarget', 'spyTarget', 'returnDuringDay_cat'], axis=1)
test_set_noTarget = test_set.copy().drop(['Date', 'spyTargetBin', 'vixTargetBin', 'vixTarget', 'spyTarget', 'returnDuringDay_cat'], axis=1)

#function to fit and evaluate
def classiFunc(est, X=train_set_noTarget, y=spyTargetTrain_Bin, X_test=test_set_noTarget, y_test=spyTargetTest_Bin):
    scores = cross_validate(est, X, y, cv=10, scoring=["accuracy", 'f1'])
    print('Accuracy', scores['test_accuracy'])
    print('F1', scores['test_f1'])
    est.fit(X,y)
    y_pred = est.predict(X_test)
    print('Test')
    print(confusion_matrix(spyTargetTest_Bin, y_pred))
    y_train_pred = cross_val_predict(est, X, y, cv=10)
    print('CV_predict')
    print(confusion_matrix(spyTargetTrain_Bin, y_train_pred))

#SGDClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_validate

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(train_set_noTarget, spyTargetTrain_Bin)

cross_val_score(sgd_clf, train_set_noTarget, spyTargetTrain_Bin, cv=10, scoring="accuracy")

scores = cross_validate(sgd_clf, train_set_noTarget, spyTargetTrain_Bin, cv=10, scoring=["accuracy", 'f1'])
scores['test_f1']
scores['test_accuracy']

classiFunc(SGDClassifier(max_iter=1000, tol=1e-3, random_state=42))

#cross_val_predict / Confustion Matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, train_set_noTarget, spyTargetTrain_Bin, cv=10)
confusion_matrix(spyTargetTrain_Bin, y_train_pred)

#Decision Function - SGD
y_scores = sgd_clf.decision_function(train_set_noTarget[1:2])
y_scores

threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

threshold = 0.72
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

y_scores = cross_val_predict(sgd_clf, train_set_noTarget, spyTargetTrain_Bin, cv=3,
                             method="decision_function")

#Precision/Recall/F1 Scores
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve

precision_score(spyTargetTrain_Bin, y_train_pred)
recall_score(spyTargetTrain_Bin, y_train_pred)
f1_score(spyTargetTrain_Bin, y_train_pred)

precisions, recalls, thresholds = precision_recall_curve(spyTargetTrain_Bin, y_scores)

#PLOT - precision_recall VS threshold
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    plt.axis([-8, 3, 0, 1])             # Not shown

recall_90_precision = recalls[np.argmax(precisions >= 0.60)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.60)]

plt.figure(figsize=(8, 4))                                                                  # Not shown
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.6], "r:")                 # Not shown
plt.plot([-8, threshold_90_precision], [0.6, 0.6], "r:")                                # Not shown
plt.plot([-8, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")# Not shown
plt.plot([threshold_90_precision], [0.6], "ro")                                             # Not shown
plt.plot([threshold_90_precision], [recall_90_precision], "ro")                             # Not shown                                            # Not shown
plt.show()

#PLOT - Precision VS Recall
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([recall_90_precision, recall_90_precision], [0., 0.6], "r:")
plt.plot([0.0, recall_90_precision], [0.6, 0.6], "r:")
plt.plot([recall_90_precision], [0.6], "ro")
plt.show()

#Adjust threshold for specified precision/recall
threshold_90_precision = thresholds[np.argmax(precisions >= 0.65)]
y_train_pred_90 = (y_scores >= threshold_90_precision)

precision_score(spyTargetTrain_Bin, y_train_pred_90)
recall_score(spyTargetTrain_Bin, y_train_pred_90)
confusion_matrix(spyTargetTrain_Bin, y_train_pred_90)

#PLOT - roc_curve / AUC score
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds1 = roc_curve(spyTargetTrain_Bin, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown

plt.figure(figsize=(8, 6))                                    # Not shown
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]           # Not shown
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   # Not shown
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  # Not shown
plt.plot([fpr_90], [recall_90_precision], "ro")               # Not shown                                 # Not shown
plt.show()

roc_auc_score(spyTargetTrain_Bin, y_scores)

#ROC curve and auc score with random Forest Classifier using predict_proba
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, train_set_noTarget, spyTargetTrain_Bin, cv=3,
                                    method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(spyTargetTrain_Bin,y_scores_forest)

recall_for_forest = tpr_forest[np.argmax(fpr_forest >= fpr_90)]

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
plt.plot([fpr_90], [recall_90_precision], "ro")
plt.plot([fpr_90, fpr_90], [0., recall_for_forest], "r:")
plt.plot([fpr_90], [recall_for_forest], "ro")
plt.grid(True)
plt.legend(loc="lower right", fontsize=16)

plt.show()

roc_auc_score(spyTargetTrain_Bin, y_scores_forest)

y_train_pred_forest = cross_val_predict(forest_clf, train_set_noTarget, spyTargetTrain_Bin, cv=3)
precision_score(spyTargetTrain_Bin, y_train_pred_forest)

recall_score(spyTargetTrain_Bin, y_train_pred_forest)

#Split Target for multiClass
train_set['spyTarget_Bin'] = 0
train_set.loc[train_set['spyTarget'] >= -0.0025, ['spyTarget_Bin']] = 1
train_set.loc[train_set['spyTarget'] >= 0.0025, ['spyTarget_Bin']] = 2

test_set['spyTarget_Bin'] = 0
test_set.loc[test_set['spyTarget'] >= -0.0025, ['spyTarget_Bin']] = 1
test_set.loc[test_set['spyTarget'] >= 0.0025, ['spyTarget_Bin']] = 2

train_set_noTarget = train_set.copy().drop(['Date', 'spyTargetBin', 'vixTargetBin', 'vixTarget', 'spyTarget', 'spyTarget_Bin'], axis=1)
test_set_noTarget = test_set.copy().drop(['Date', 'spyTargetBin', 'vixTargetBin', 'vixTarget', 'spyTarget', 'spyTarget_Bin'], axis=1)

train_set.hist('spyTarget_Bin', bins=100)
train_set['spyTarget_Bin'].hist()

#SVC MultiClass One VS One 
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto", random_state=42) # SVC Automatically uses one vs one for multiClass
svm_clf.fit(train_set_noTarget, train_set['spyTarget_Bin']) # y_train, not y_train_5
svm_clf.predict(train_set_noTarget[3:800])

some_digit_scores = svm_clf.decision_function(train_set_noTarget[4:500])
some_digit_scores

np.argmax(some_digit_scores)
svm_clf.classes_
svm_clf.classes_[2]

cross_val_score(svm_clf, train_set_noTarget, train_set['spyTarget_Bin'], cv=3, scoring="accuracy")

#OneVsRestClassifier SVC multiClass
from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC(gamma="auto", random_state=42))
ovr_clf.fit(train_set_noTarget, train_set['spyTarget_Bin'])
ovr_clf.predict(train_set_noTarget[3:800])

some_digit_scores = ovr_clf.decision_function(train_set_noTarget[4:55])
some_digit_scores

len(ovr_clf.estimators_)

cross_val_score(ovr_clf, train_set_noTarget, train_set['spyTarget_Bin'], cv=3, scoring="accuracy")

#SGD multiClass
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(train_set_noTarget, train_set['spyTarget_Bin'])
sgd_clf.predict(train_set_noTarget[4:555])

sgd_clf.decision_function(train_set_noTarget[4:55])

cross_val_score(sgd_clf, train_set_noTarget, train_set['spyTarget_Bin'], cv=3, scoring="accuracy")

#Standardize / Normalize / Confusion Matrix
from sklearn.preprocessing import StandardScaler, Normalizer
scaler = StandardScaler()
scaler2 = Normalizer()
X_train_scaled = scaler.fit_transform(train_set_noTarget.astype(np.float64))
cross_val_score(ovr_clf, X_train_scaled, train_set['spyTarget_Bin'], cv=3, scoring="accuracy")

y_train_pred = cross_val_predict(ovr_clf, X_train_scaled, train_set['spyTarget_Bin'], cv=3)
conf_mx = confusion_matrix(train_set['spyTarget_Bin'], y_train_pred)
conf_mx

#Plot confusion matrix
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

#Plot confusion matrix errors
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="lbfgs", random_state=42)
log_reg.fit(train_set_noTarget, spyTargetTrain_Bin)

y_pred = log_reg.predict(train_set_noTarget)
log_reg.predict_proba(train_set_noTarget.iloc[0:500])

confusion_matrix(spyTargetTrain_Bin, y_pred)

#Softmax Regression using Logistic Regression
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
softmax_reg.fit(train_set_noTarget, spyTargetTrain_Bin)


'''SVM'''

#linear SVC - multiple ways
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier

#LinearSVC
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
    ])

svm_clf.fit(train_set_noTarget, spyTargetTrain_Bin)
y_pred = svm_clf.predict(train_set_noTarget)
confusion_matrix(spyTargetTrain_Bin, y_pred)

#SVC Kernel-Linear
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", SVC(kernel='linear', C=1)),
    ])

svm_clf.fit(train_set_noTarget, spyTargetTrain_Bin)
y_pred = svm_clf.predict(train_set_noTarget)
confusion_matrix(spyTargetTrain_Bin, y_pred)

#SVC SGDClassifier-Linear
m=len(train_set_noTarget)
C=1

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", SGDClassifier(loss='hinge', alpha=1/(m*C))),
    ])

svm_clf.fit(train_set_noTarget, spyTargetTrain_Bin)
y_pred = svm_clf.predict(train_set_noTarget)
confusion_matrix(spyTargetTrain_Bin, y_pred)

#polynomial SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])

polynomial_svm_clf.fit(train_set_noTarget, spyTargetTrain_Bin)
y_pred = svm_clf.predict(train_set_noTarget)
confusion_matrix(spyTargetTrain_Bin, y_pred)

#SVC - Polynomial Kernel
from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
poly_kernel_svm_clf.fit(train_set_noTarget, spyTargetTrain_Bin)
y_pred = svm_clf.predict(train_set_noTarget)
confusion_matrix(spyTargetTrain_Bin, y_pred)

#SVC Gaussian RBF Kernel
from sklearn.svm import SVC

rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ])
rbf_kernel_svm_clf.fit(train_set_noTarget, spyTargetTrain_Bin)
y_pred = svm_clf.predict(train_set_noTarget)
confusion_matrix(spyTargetTrain_Bin, y_pred)


'''DECISION TREE'''

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(train_set_noTarget, spyTargetTrain_Bin)

tree_clf.predict_proba(train_set_noTarget.iloc[0:500])
tree_clf.predict(train_set_noTarget.iloc[0:500])

#plot decision tree sklearn
from sklearn.tree import plot_tree

plot_tree(tree_clf, feature_names=train_set_noTarget.columns, rounded=True, fontsize=5)

#plot decision tree GRAPHVIZ
from graphviz import Source
from sklearn.tree import export_graphviz

dot_data = export_graphviz(tree_clf, out_file=None, 
                           feature_names=train_set_noTarget.columns,  
                           filled=True)

graph = Source(dot_data, format="png") 
graph


'''ENSEMBLES / RANDOM FORESTS'''

#Voting Classifier - Hard
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')

voting_clf.fit(train_set_noTarget, spyTargetTrain_Bin)

#Voting Classifier - Soft
log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", random_state=42, probability=True)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')

voting_clf.fit(train_set_noTarget, spyTargetTrain_Bin)

#evaluate voting classifier
from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(train_set_noTarget, spyTargetTrain_Bin)
    y_pred = clf.predict(test_set_noTarget)
    print(clf.__class__.__name__, accuracy_score(spyTargetTest_Bin, y_pred))

#Bagging/Pasting Classifier - change bootstrap=False for Pasting Classifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, random_state=42)
bag_clf.fit(train_set_noTarget, spyTargetTrain_Bin)
y_pred = bag_clf.predict(test_set_noTarget)

print(accuracy_score(spyTargetTest_Bin, y_pred))

#Out of Bag (oob) Evaluation
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    bootstrap=True, oob_score=True, random_state=40)
bag_clf.fit(train_set_noTarget, spyTargetTrain_Bin)
bag_clf.oob_score_

bag_clf.oob_decision_function_

from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(test_set_noTarget)
accuracy_score(spyTargetTest_Bin, y_pred)

#Random Subspaces (max_samples=1, bootstrap=False, setting bootstrap_features=True AND/OR max_features= <1)
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=1.0, bootstrap=False, bootstrap_features=True, max_features=0.5, random_state=42)
bag_clf.fit(train_set_noTarget, spyTargetTrain_Bin)
y_pred = bag_clf.predict(test_set_noTarget)

print(accuracy_score(spyTargetTest_Bin, y_pred))

#Random Patches (setting bootstrap_features=True AND/OR max_features= <1)
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, bootstrap_features=True, max_features=0.5, random_state=42)
bag_clf.fit(train_set_noTarget, spyTargetTrain_Bin)
y_pred = bag_clf.predict(test_set_noTarget)

print(accuracy_score(spyTargetTest_Bin, y_pred))

classiFunc(bag_clf)

#Random Forest
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
rnd_clf.fit(train_set_noTarget, spyTargetTrain_Bin)

y_pred_rf = rnd_clf.predict(test_set_noTarget)

#Random Forest using BaggingClassifier
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
    n_estimators=500, max_samples=1.0, bootstrap=True, random_state=42)

bag_clf.fit(train_set_noTarget, spyTargetTrain_Bin)
y_pred = bag_clf.predict(test_set_noTarget)

np.sum(y_pred == y_pred_rf) / len(y_pred)  # almost identical predictions

#ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier

rnd_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
rnd_clf.fit(train_set_noTarget, spyTargetTrain_Bin)

y_pred_rf = rnd_clf.predict(test_set_noTarget)

classiFunc(rnd_clf)

#Feature Importance using RandomForest
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(train_set_noTarget, spyTargetTrain_Bin)
for name, score in zip(train_set_noTarget.columns, rnd_clf.feature_importances_):
    print(name, score)
      
rnd_clf.feature_importances_

#AdaBoost
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(train_set_noTarget, spyTargetTrain_Bin)

classiFunc(ada_clf)

#Plot Decision Boundary with AdaBoost (must use only two features)
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(train_set_noTarget[['returnDuringDay', 'highDuringDay']], spyTargetTrain_Bin)

from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.45, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    
plot_decision_boundary(ada_clf, train_set_noTarget[['returnDuringDay', 'highDuringDay']], spyTargetTrain_Bin)

#GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier

gbrt = GradientBoostingClassifier(max_depth=2, n_estimators=200 , learning_rate=0.1, random_state=42)
gbrt.fit(train_set_noTarget, spyTargetTrain_Bin)

classiFunc(gbrt)

#Gradient Boosting with early stopping to find optimal number of trees (with lookback)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(train_set_noTarget, spyTargetTrain_Bin, random_state=49)

gbrt = GradientBoostingClassifier(max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X_train, y_train)

errors = [accuracy_score(y_val, y_pred)
          for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmax(errors) + 1

gbrt_best = GradientBoostingClassifier(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(X_train, y_train)

classiFunc(gbrt_best)

#Gradient Boosting with early stopping to find optimal number of trees (by actually stopping early)
gbrt = GradientBoostingClassifier(max_depth=2, warm_start=True, random_state=42)

max_val_acc = 0
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = accuracy_score(y_val, y_pred)
    if val_error > max_val_acc:
        max_val_acc = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break  # early stopping
            
print(gbrt.n_estimators)
print("Max validation Acc:", max_val_acc)

#XGBoost
#conda install -c anaconda py-xgboost

import xgboost

xgb_reg = xgboost.XGBClassifier(random_state=42)
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_val)
val_acc = accuracy_score(y_val, y_pred) # Not shown
print("Validation ACC:", val_acc) 

#XGBoost Early stopping
xgb_reg = xgboost.XGBClassifier(random_state=42)
xgb_reg.fit(X_train, y_train,eval_set=[(X_val, y_val)], early_stopping_rounds=2)
y_pred = xgb_reg.predict(X_val)
val_acc = accuracy_score(y_val, y_pred) # Not shown
print("Validation ACC:", val_acc) 


''''DIMENSIONALITY REDUCTION'''

#PCA by num of components
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
transPCA = pca.fit_transform(train_set_noTarget)

pca.explained_variance_ratio_
t = pca.components_

inv_PCA = pca.inverse_transform(transPCA) # Decompress back to orig num of comp (info loss due to initial transformation)

#PCA by percent of variance
pca = PCA(n_components = 0.999)
transPCA = pca.fit_transform(train_set_noTarget)

pca.explained_variance_ratio_

#PCA all components - plot variance/component curve
pca = PCA()
pca.fit(train_set_noTarget)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.999) + 1

pca.explained_variance_ratio_

plt.plot(cumsum, linewidth=2)

#PCA randomized solver
pca = PCA(n_components = 2, svd_solver='randomized')
transPCA = pca.fit_transform(train_set_noTarget)

pca.explained_variance_ratio_

#PCA - Incremental
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=3)
for X_batch in np.array_split(train_set_noTarget, n_batches):
    print(".", end="") # not shown in the book
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(train_set_noTarget)

pca.explained_variance_ratio_

#kPCA - different kernels
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(train_set_noTarget)

lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
X_reduced = lin_pca.fit_transform(train_set_noTarget)

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(train_set_noTarget)

sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)
X_reduced = sig_pca.fit_transform(train_set_noTarget)

#Use Grid Search to find best kernel and gamma values for kPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression(solver="lbfgs"))
    ])

param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
    }]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(train_set_noTarget, spyTargetTrain_Bin)

print(grid_search.best_params_)

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(train_set_noTarget)

#Reconstruction error technique to find best kernel and gamma values by minimizing mse
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)

X_reduced = rbf_pca.fit_transform(train_set_noTarget) #Compress
X_preimage = rbf_pca.inverse_transform(X_reduced) # Decompress

from sklearn.metrics import mean_squared_error
mean_squared_error(train_set_noTarget, X_preimage) # reconstruction error

#LLE - local linear embedding - manifold technique
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_reduced = lle.fit_transform(train_set_noTarget)

#Other Dimensionality Reduction techniques (Multidimensional Scaling / Isomap / t-SNE / LDA)
from sklearn.manifold import MDS
mds = MDS(n_components=2, random_state=42)
X_reduced_mds = mds.fit_transform(train_set_noTarget)

from sklearn.manifold import Isomap
isomap = Isomap(n_components=2)
X_reduced_isomap = isomap.fit_transform(train_set_noTarget)

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
X_reduced_tsne = tsne.fit_transform(train_set_noTarget)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(train_set_noTarget, spyTargetTrain_Bin)
X_reduced_lda = lda.transform(train_set_noTarget)


'''UNSUPERVISED LEARNING'''

#K-means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(train_set_noTarget)

kmeans.labels_
kmeans.cluster_centers_
kmeans.inertia_
kmeans.score(train_set_noTarget)

kmeans.predict(train_set_noTarget.iloc[0:2]) # Hard Clustering
kmeans.transform(train_set_noTarget.iloc[0:2]) # Soft Clustering

#MiniBatchKMeans
from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, n_init=10)
minibatch_kmeans.fit_predict(train_set_noTarget)

minibatch_kmeans.inertia_

#Plot - Inertia vs k
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(train_set_noTarget)
                for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.axis([1, 10, 0, 8000])
plt.show()

#Plot - silhouette score
from sklearn.metrics import silhouette_score

silhouette_score(train_set_noTarget, kmeans.labels_)

silhouette_scores = [silhouette_score(train_set_noTarget, model.labels_)
                     for model in kmeans_per_k[1:]]

plt.figure(figsize=(8, 3))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.axis([0, 11, 0.0, 1.7])
plt.show()

#Image Segmentation - k-Means
from matplotlib.image import imread
image = imread('C:/Users/Jonathan/.spyder-py3/scripts/sashPic.jpg')
image.shape

X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=4).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

t = segmented_img.astype(int)

plt.imshow(image)
plt.axis('off')

plt.imshow(t)
plt.axis('off')

#Preprocessing data using k-Means
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(train_set_noTarget, spyTargetTrain_Bin)

log_reg.score(test_set_noTarget, spyTargetTest_Bin)

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50, random_state=42)),
    ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)),
])

pipeline.fit(train_set_noTarget, spyTargetTrain_Bin)

pipeline.score(test_set_noTarget, spyTargetTest_Bin)

from sklearn.model_selection import GridSearchCV

param_grid = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(train_set_noTarget, spyTargetTrain_Bin)

grid_clf.best_params_
i = grid_clf.cv_results_
grid_clf.score(test_set_noTarget, spyTargetTest_Bin)

#Semi-Supervised learning using clustering (k-Means)
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", random_state=42)
log_reg.fit(train_set_noTarget[:50], spyTargetTrain_Bin[:50])
log_reg.score(test_set_noTarget, spyTargetTest_Bin)

kmeans = KMeans(n_clusters=50, random_state=42)
X_digits_dist = kmeans.fit_transform(train_set_noTarget)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = train_set_noTarget.iloc[representative_digit_idx]
y_representative_digits = spyTargetTrain_Bin.iloc[representative_digit_idx]

log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_representative_digits, y_representative_digits)
log_reg.score(test_set_noTarget, spyTargetTest_Bin)

y_train_propagated = np.empty(len(train_set_noTarget), dtype=np.int32)
for i in range(50):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i:i+1]
    
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(train_set_noTarget, y_train_propagated)
log_reg.score(test_set_noTarget, spyTargetTest_Bin)

percentile_closest = 20
X_cluster_dist = X_digits_dist[np.arange(len(train_set_noTarget)), kmeans.labels_]
for i in range(50):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1
    
partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = train_set_noTarget[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
log_reg.score(test_set_noTarget, spyTargetTest_Bin)

#DBSCAN
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(train_set_noTarget)

len(dbscan.labels_)
len(dbscan.core_sample_indices_)
dbscan.core_sample_indices_[:100]
len(dbscan.components_)
np.unique(dbscan.labels_)

dbscan2 = DBSCAN(eps=0.2)
dbscan2.fit(train_set_noTarget)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

i = knn.predict_proba(test_set_noTarget)

y_dist, y_pred_idx = knn.kneighbors(test_set_noTarget, n_neighbors=1)
y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
y_pred[y_dist > 0.2] = -1
i = y_pred.ravel()

#SpectralClustering
from sklearn.cluster import SpectralClustering

sc1 = SpectralClustering(n_clusters=2, gamma=100, random_state=42)
sc1.fit(train_set_noTarget)

np.percentile(sc1.affinity_matrix_, 95)

#AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering

X = np.array([0, 2, 5, 8.5]).reshape(-1, 1)
agg = AgglomerativeClustering(linkage="complete").fit(X)

def learned_parameters(estimator):
    return [attrib for attrib in dir(estimator)
            if attrib.endswith("_") and not attrib.startswith("_")]

learned_parameters(agg)

agg.children_

#GaussianMixture
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(train_set_noTarget)

gm.weights_
gm.means_
gm.covariances_
gm.converged_
gm.n_iter_

i = gm.predict(test_set_noTarget)
i = gm.predict_proba(test_set_noTarget)

X_new, y_new = gm.sample(10)
X_new

gm.score_samples(train_set_noTarget[0:1])

gm_full = GaussianMixture(n_components=3, n_init=10, covariance_type="full", random_state=42)
gm_tied = GaussianMixture(n_components=3, n_init=10, covariance_type="tied", random_state=42)
gm_spherical = GaussianMixture(n_components=3, n_init=10, covariance_type="spherical", random_state=42)
gm_diag = GaussianMixture(n_components=3, n_init=10, covariance_type="diag", random_state=42)
gm_full.fit(X)
gm_tied.fit(X)
gm_spherical.fit(X)
gm_diag.fit(X)

#Anomaly Detection with Gaussian Mixtures
densities = gm.score_samples(train_set_noTarget)
density_threshold = np.percentile(densities, 4)
anomalies = train_set_noTarget[densities < density_threshold]

#find and plot best k using bic/aic 
gm.bic(train_set_noTarget)
gm.aic(train_set_noTarget)

gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=42).fit(train_set_noTarget)
             for k in range(1, 20)]

bics = [model.bic(train_set_noTarget) for model in gms_per_k]
aics = [model.aic(train_set_noTarget) for model in gms_per_k]

plt.plot(range(1, 20), bics, "bo-", label="BIC")
plt.plot(range(1, 20), aics, "go--", label="AIC")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Information Criterion", fontsize=14)
plt.axis([1, 20, np.min(aics) - 50, np.max(aics) + 50])
plt.legend()

#find best combination of k and covariance_type
min_bic = np.infty

for k in range(1, 20):
    for covariance_type in ("full", "tied", "spherical", "diag"):
        bic = GaussianMixture(n_components=k, n_init=10,
                              covariance_type=covariance_type,
                              random_state=42).fit(train_set_noTarget).bic(train_set_noTarget)
        if bic < min_bic:
            min_bic = bic
            best_k = k
            best_covariance_type = covariance_type
            
best_k
best_covariance_type
            
#BayesianGaussianMixture
from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(train_set_noTarget)
np.round(bgm.weights_, 2)

bgm_low = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,
                                  weight_concentration_prior=0.01, random_state=42)
bgm_high = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,
                                  weight_concentration_prior=10000, random_state=42)

bgm_low.fit(train_set_noTarget[:100])
bgm_high.fit(train_set_noTarget[:100])

np.round(bgm_low.weights_, 2)
np.round(bgm_high.weights_, 2)







'''NEURAL NETWORKS'''




'''BASICS'''
#Perceptrom using sklearn
import numpy as np
from sklearn.linear_model import Perceptron

per_clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
per_clf.fit(train_set_noTarget, spyTargetTrain_Bin)

y_pred = per_clf.predict(train_set_noTarget[0:50])

#Classification MLP
import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_train_full.shape
X_train_full.dtype

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

plt.imshow(X_train[0], cmap="binary")
plt.axis('off')

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
#or
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model.layers
model.summary()

keras.utils.plot_model(model, show_shapes=True)

hidden1 = model.layers[1]
hidden1.name
model.get_layer(hidden1.name) is hidden1

weights, biases = hidden1.get_weights()
weights
weights.shape
biases

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
#or
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=[keras.metrics.sparse_categorical_accuracy])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

history.params
print(history.epoch)
history.history.keys()

import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)

model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

y_pred = model.predict_classes(X_new)
y_pred

#Regression MLP
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_pred = model.predict(X_new)
y_pred

plt.plot(pd.DataFrame(history.history))
plt.grid(True)
plt.gca().set_ylim(0, 1)

#Functional API
np.random.seed(42)
tf.random.set_seed(42)

input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input_], outputs=[output])

model.summary()

model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
y_pred = model.predict(X_new)

np.random.seed(42)
tf.random.set_seed(42)

#two inputs with deep and wide model
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
                    validation_data=((X_valid_A, X_valid_B), y_valid))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A, X_new_B))

#add auxilliary output for regularization
np.random.seed(42)
tf.random.set_seed(42)

input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="main_output")(concat)
aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
model = keras.models.Model(inputs=[input_A, input_B],
                           outputs=[output, aux_output])

model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(lr=1e-3))

history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20,
                    validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))

total_loss, main_loss, aux_loss = model.evaluate(
    [X_test_A, X_test_B], [y_test, y_test])
y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])

#Subclassing API
class WideAndDeepModel(keras.models.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)
        
    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

model = WideAndDeepModel(30, activation="relu")

model.compile(loss="mse", loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(lr=1e-3))
history = model.fit((X_train_A, X_train_B), (y_train, y_train), epochs=10,
                    validation_data=((X_valid_A, X_valid_B), (y_valid, y_valid)))
total_loss, main_loss, aux_loss = model.evaluate((X_test_A, X_test_B), (y_test, y_test))
y_pred_main, y_pred_aux = model.predict((X_new_A, X_new_B))

#Saving and restoring models
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
    ])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)

model.save("my_keras_model.h5")
model = keras.models.load_model("my_keras_model.h5")

model.save_weights("my_keras_weights.ckpt") # for subclassing API
model.load_weights("my_keras_weights.ckpt") # for subclassing API

#using callbacks during training
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
    ])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb])
model = keras.models.load_model("my_keras_model.h5") # rollback to best model
mse_test = model.evaluate(X_test, y_test)

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])
mse_test = model.evaluate(X_test, y_test)

class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))
        
    
val_train_ratio_cb = PrintValTrainRatioCallback()
history = model.fit(X_train, y_train, epochs=50,
                    validation_data=(X_valid, y_valid),
                    callbacks=[val_train_ratio_cb])

'''Tensorboard - PROBLEMATIC!!!!!!! NEEEDS FIXED!!!!!'''
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
run_logdir

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
    ])    

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=40,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, tensorboard_cb])

#%load_ext tensorboard  # in console
#%tensorboard --logdir=./my_logs --port=6006  # in console
#OR
#tensorboard --logdir=./my_logs --port=6006 # in command prompt

#in browser go to localhost:6006


#using Grid Search in Keras
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

keras_reg.fit(X_train, y_train, epochs=20,
              validation_data=(X_valid, y_valid),
              callbacks=[keras.callbacks.EarlyStopping(patience=5)])

mse_test = keras_reg.score(X_test, y_test)

y_pred = keras_reg.predict(X_new)

np.random.seed(42)
tf.random.set_seed(42)

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

param_distribs = {
    "n_hidden": [0, 1, 4],
    "n_neurons": [1, 2, 10],
    "learning_rate": [3e-2, 3e-1]
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=2, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=20,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])
#OR
rnd_search_cv = GridSearchCV(keras_reg, param_distribs, cv=2, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=20,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])

rnd_search_cv.best_params_
rnd_search_cv.best_score_
rnd_search_cv.best_estimator_
rnd_search_cv.score(X_test, y_test)

model = rnd_search_cv.best_estimator_.model
model

model.evaluate(X_test, y_test)





'''DEEP NEURAL NETWORKS'''


'''Vanishing/Exploding Gradient - Initializers, Activation Funcs, Batch Norm.'''
#setting initializer
keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")

#Setting initializer using fan-average vs fan-in
init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg', distribution='uniform')
keras.layers.Dense(10, activation="relu", kernel_initializer=init)

#Leaky ReLU 
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

tf.random.set_seed(42)
np.random.seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

#PReLU
tf.random.set_seed(42)
np.random.seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(100, kernel_initializer="he_normal"),
    keras.layers.PReLU(),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))

#ELU
keras.layers.Dense(10, activation="elu")

#SELU (must standardize inputs)
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="selu",
                             kernel_initializer="lecun_normal"))
for layer in range(99):
    model.add(keras.layers.Dense(100, activation="selu",
                                 kernel_initializer="lecun_normal"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"])

pixel_means = X_train.mean(axis=0, keepdims=True) #standardize inputs
pixel_stds = X_train.std(axis=0, keepdims=True)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds

history = model.fit(X_train_scaled, y_train, epochs=5,
                    validation_data=(X_valid_scaled, y_valid))

#ReLU
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu", kernel_initializer="he_normal"))
for layer in range(99):
    model.add(keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"])

history = model.fit(X_train_scaled, y_train, epochs=5,
                    validation_data=(X_valid_scaled, y_valid))

#Batch Normalization (added after activation func)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
    ])

model.summary()

bn1 = model.layers[1]
[(var.name, var.trainable) for var in bn1.variables]

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))

#Batch Normalization (added before activation func)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(100, use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))

#Gradient clipping
optimizer = keras.optimizers.SGD(clipvalue=1.0)
optimizer = keras.optimizers.SGD(clipnorm=1.0)

'''Using Pretrained Layers'''
#Transfer learning
def split_dataset(X, y):
    y_5_or_6 = (y == 5) | (y == 6) # sandals or shirts
    y_A = y[~y_5_or_6]
    y_A[y_A > 6] -= 2 # class indices 7, 8, 9 should be moved to 5, 6, 7
    y_B = (y[y_5_or_6] == 6).astype(np.float32) # binary classification task: is it a shirt (class 6)?
    return ((X[~y_5_or_6], y_A),
            (X[y_5_or_6], y_B))

(X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)
(X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid)
(X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)
X_train_B = X_train_B[:200]
y_train_B = y_train_B[:200]

X_train_A.shape
X_train_B.shape

tf.random.set_seed(42)
np.random.seed(42)

model_A = keras.models.Sequential()
model_A.add(keras.layers.Flatten(input_shape=[28, 28]))
for n_hidden in (300, 100, 50, 50, 50):
    model_A.add(keras.layers.Dense(n_hidden, activation="selu"))
model_A.add(keras.layers.Dense(8, activation="softmax"))

model_A.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.SGD(lr=1e-3),
                metrics=["accuracy"])

history = model_A.fit(X_train_A, y_train_A, epochs=20,
                    validation_data=(X_valid_A, y_valid_A))

model_A.save("my_model_A.h5")

model_B = keras.models.Sequential()
model_B.add(keras.layers.Flatten(input_shape=[28, 28]))
for n_hidden in (300, 100, 50, 50, 50):
    model_B.add(keras.layers.Dense(n_hidden, activation="selu"))
model_B.add(keras.layers.Dense(1, activation="sigmoid"))

model_B.compile(loss="binary_crossentropy",
                optimizer=keras.optimizers.SGD(lr=1e-3),
                metrics=["accuracy"])

history = model_B.fit(X_train_B, y_train_B, epochs=20,
                      validation_data=(X_valid_B, y_valid_B))

model.summary()

model_A = keras.models.load_model("my_model_A.h5")
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))
#clone model to make copy so original model wont be affected
model_A_clone = keras.models.clone_model(model_A) 
model_A_clone.set_weights(model_A.get_weights())

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

model_B_on_A.compile(loss="binary_crossentropy",
                     optimizer=keras.optimizers.SGD(lr=1e-3),
                     metrics=["accuracy"])

history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
                           validation_data=(X_valid_B, y_valid_B))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

model_B_on_A.compile(loss="binary_crossentropy",
                     optimizer=keras.optimizers.SGD(lr=1e-3),
                     metrics=["accuracy"])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
                           validation_data=(X_valid_B, y_valid_B))

model_B.evaluate(X_test_B, y_test_B)
model_B_on_A.evaluate(X_test_B, y_test_B)

'''OPTIMIZERS'''
#Momentum
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)

#Nesterov Accelerated Gradient (NAG)
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

#AdaGrad
optimizer = keras.optimizers.Adagrad(lr=0.001)

#RMSProp
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)

#Adam
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#AdaMax
optimizer = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999)

#Nadam
optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)

'''Learning Rate Scheduling'''
#Power scheduling
optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)

#Exponential scheduling per epoch
def exponential_decay_fn(epoch):       # Using epoch as argument
    return 0.01 * 0.1**(epoch / 20)
#OR
def exponential_decay(lr0, s):          # More general function
    def exponential_decay_fn(epoch):    # If you don't want to hard code lr0 and s
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn
#OR
def exponential_decay_fn(epoch, lr):   # Using lr as argument instead of epoch
    return lr * 0.1**(1 / 20)          # Easier for reloading model than epoch as argument

exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
#THEN
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
n_epochs = 25

lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[lr_scheduler])

#Exponential scheduling per batch
K = keras.backend

class ExponentialDecay(keras.callbacks.Callback):
    def __init__(self, s=40000):
        super().__init__()
        self.s = s

    def on_batch_begin(self, batch, logs=None):
        # Note: the `batch` argument is reset at each epoch
        lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, lr * 0.1**(1 / s))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(10, activation="softmax")
])
lr0 = 0.01
optimizer = keras.optimizers.Nadam(lr=lr0)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
n_epochs = 25

s = 20 * len(X_train) // 32 # number of steps in 20 epochs (batch size = 32)
exp_decay = ExponentialDecay(s)
history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[exp_decay])

#PieceWise Constant Scheduling
def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001
#OR
def piecewise_constant(boundaries, values):    # More general function
    boundaries = np.array([0] + boundaries)
    values = np.array(values)
    def piecewise_constant_fn(epoch):
        return values[np.argmax(boundaries > epoch) - 1]
    return piecewise_constant_fn

piecewise_constant_fn = piecewise_constant([5, 15], [0.01, 0.005, 0.001])
#THEN
lr_scheduler = keras.callbacks.LearningRateScheduler(piecewise_constant_fn)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
n_epochs = 25
history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[lr_scheduler])

#Performance Scheduling
tf.random.set_seed(42)
np.random.seed(42)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(10, activation="softmax")
])
optimizer = keras.optimizers.SGD(lr=0.02, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
n_epochs = 25
history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[lr_scheduler])

#Learning Rate scheduling using tf.keras schedulers
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(10, activation="softmax")
])
s = 20 * len(X_train) // 32 # number of steps in 20 epochs (batch size = 32)
learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
optimizer = keras.optimizers.SGD(learning_rate)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
n_epochs = 25
history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                    validation_data=(X_valid_scaled, y_valid))
#OR for piecewise constant
n_steps_per_epoch = 16
learning_rate = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[5. * n_steps_per_epoch, 15. * n_steps_per_epoch],
    values=[0.01, 0.005, 0.001])

#1cyle scheduling - HAS PROBLEMS!
class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0
    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)
    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
            rate = max(rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)

batch_size = 128
n_epochs = 25
onecycle = OneCycleScheduler(len(X_train) // batch_size * n_epochs, max_rate=0.05)
history = model.fit(X_train_scaled, y_train, epochs=n_epochs, batch_size=batch_size,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[onecycle])

'''REGULARIZATION'''
#L1 and L2
layer = keras.layers.Dense(100, activation="elu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01))
# or l1(0.1) for ℓ1 regularization with a factor or 0.1
# or l1_l2(0.1, 0.01) for both ℓ1 and ℓ2 regularization, with factors 0.1 and 0.01 respectively
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="elu",
                       kernel_initializer="he_normal",
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(100, activation="elu",
                       kernel_initializer="he_normal",
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(10, activation="softmax",
                       kernel_regularizer=keras.regularizers.l2(0.01))
    ])

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
n_epochs = 2
history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                    validation_data=(X_valid_scaled, y_valid))

#To prevent repeating same arguments for every layer
from functools import partial

RegularizedDense = partial(keras.layers.Dense,
                           activation="elu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    RegularizedDense(300),
    RegularizedDense(100),
    RegularizedDense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
n_epochs = 2
history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                    validation_data=(X_valid_scaled, y_valid))

#Dropout
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
    ])

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
n_epochs = 2
history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                    validation_data=(X_valid_scaled, y_valid))

#Alpha Dropout
tf.random.set_seed(42)
np.random.seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
])
optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
n_epochs = 20
history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                    validation_data=(X_valid_scaled, y_valid))

#Monte Carlo Dropout
tf.random.set_seed(42)
np.random.seed(42)

y_probas = np.stack([model(X_test_scaled, training=True)
                     for sample in range(100)])
y_proba = y_probas.mean(axis=0)
y_std = y_probas.std(axis=0)

y_pred = np.argmax(y_proba, axis=1)

accuracy = np.sum(y_pred == y_test) / len(y_test)
accuracy
 
#Monte Carlo using MCDropout/MCAlphaDropout class (use for models with special layers)
class MCDropout(keras.layers.Dropout):               
    def call(self, inputs):
        return super().call(inputs, training=True)

class MCAlphaDropout(keras.layers.AlphaDropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
    
tf.random.set_seed(42)
np.random.seed(42)

mc_model = keras.models.Sequential([
    MCAlphaDropout(layer.rate) if isinstance(layer, keras.layers.AlphaDropout) else layer
    for layer in model.layers
])

optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
mc_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

mc_model.set_weights(model.get_weights())

np.round(np.mean([mc_model.predict(X_test_scaled[:1]) for sample in range(100)], axis=0), 2)

#Max-Norm Regularization
layer = keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal",
                           kernel_constraint=keras.constraints.max_norm(1.))

MaxNormDense = partial(keras.layers.Dense,
                       activation="selu", kernel_initializer="lecun_normal",
                       kernel_constraint=keras.constraints.max_norm(1.))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    MaxNormDense(300),
    MaxNormDense(100),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
n_epochs = 2
history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                    validation_data=(X_valid_scaled, y_valid))

 

'''lower-level TensorFlow'''
#tensors
t = tf.constant([[1., 2., 3.], [4., 5., 6.]]) # matrix
tf.constant(42) # scalar
t.shape
t.dtype

#variables
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
v.assign(2 * v)
v[0, 1].assign(42)
v[:, 2].assign([0., 1.])

#sparse tensor
s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],
                    values=[1., 2., 3.],
                    dense_shape=[3, 4])
print(s)
tf.sparse.to_dense(s)

#indexing
t[:, 1:]
t[..., 1, tf.newaxis]

#Ops
t + 10
tf.square(t)
t @ tf.transpose(t)

#using keras.backend
from tensorflow import keras
K = keras.backend
K.square(K.transpose(t)) + 10

#From/To Numpy
a = np.array([2., 4., 5.])
tf.constant(a)
t.numpy()
np.array(t)
tf.square(a)
np.square(t)

#Also - other data types - strings, string arrays, ragged tendsors, sparse tensors, sets, tensor arrays, etc.
#Also - custom loss funcs, act funcs, initializers, regularizers, constraints, metrics, layers, models, training loops, etc.
#https://github.com/ageron/handson-ml2/blob/master/12_custom_models_and_training_with_tensorflow.ipynb
#Chapter 12




'''Preprocessing Data with TensorFlow'''

#Data API OPS (batching/unbatching, mapping, filtering, shuffling)
X = tf.range(10)
X = [0,1,2,3,4,5,6,7,8,9]
dataset = tf.data.Dataset.from_tensor_slices(X)
dataset
#OR
dataset = tf.data.Dataset.range(10)
#THEN
for item in dataset:
    print(item)
    
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)
    
dataset = dataset.map(lambda x: x * 2)
for item in dataset:
    print(item)
    
dataset = dataset.unbatch()

dataset = dataset.filter(lambda x: x < 10)  # keep only items < 10

for item in dataset.take(3):
    print(item)
    
tf.random.set_seed(42)
dataset = tf.data.Dataset.range(10).repeat(3)
dataset = dataset.shuffle(buffer_size=3, seed=42).batch(7)
for item in dataset:
    print(item)

#custom preprocessing layer
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class Standardization(keras.layers.Layer):
    def adapt(self, data_sample):
        self.means_ = np.mean(data_sample, axis=0, keepdims=True)
        self.stds_ = np.std(data_sample, axis=0, keepdims=True)
    def call(self, inputs):
        return (inputs - self.means_) / (self.stds_ + keras.backend.epsilon())

standardization = Standardization(input_shape=[28, 28])
# or perhaps soon:
#standardization = keras.layers.Normalization()

standardization.adapt(X_train)

model = keras.models.Sequential([
    standardization,
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="nadam", metrics=["accuracy"])




'''CNNs'''

#Convolution using lower level TensorFlow
import numpy as np
from sklearn.datasets import load_sample_image

china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower])
batch_size, height, width, channels = images.shape

filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # vertical line
filters[3, :, :, 1] = 1  # horizontal line

outputs = tf.nn.conv2d(images, filters, strides=2, padding="SAME")

plt.imshow(outputs[0, :, :, 1], cmap="gray") # plot 1st image's 2nd feature map
plt.axis("off") # Not shown in the book
plt.show()

#Conv layer using keras
conv = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                           padding="SAME", activation="relu")

#Pooling layer (MaxPool)
max_pool = keras.layers.MaxPool2D(pool_size=1)
output = max_pool(images)
plt.imshow(output[0])

#Pooling layer (AvgPool)
avg_pool = keras.layers.AvgPool2D(pool_size=2)
output_avg = avg_pool(images)
plt.imshow(output_avg[0])

#Pooling layer (GlobalAvgPool)
global_avg_pool = keras.layers.GlobalAvgPool2D()
global_avg = global_avg_pool(images)
#OR
output_global_avg2 = keras.layers.Lambda(lambda X: tf.reduce_mean(X, axis=[1, 2]))
output_global_avg2(images)

#Depth-wise pooling
class DepthMaxPool(keras.layers.Layer):
    def __init__(self, pool_size, strides=None, padding="VALID", **kwargs):
        super().__init__(**kwargs)
        if strides is None:
            strides = pool_size
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
    def call(self, inputs):
        return tf.nn.max_pool(inputs,
                              ksize=(1, 1, 1, self.pool_size),
                              strides=(1, 1, 1, self.pool_size),
                              padding=self.padding)
    
depth_pool = DepthMaxPool(3)
with tf.device("/cpu:0"): # there is no GPU-kernel yet
    depth_output = depth_pool(images)
depth_output.shape
plt.imshow(depth_output[0])

#Depth-wise pooling using lambda layer
depth_pool = keras.layers.Lambda(lambda X: tf.nn.max_pool(
    X, ksize=(1, 1, 1, 3), strides=(1, 1, 1, 3), padding="VALID"))
with tf.device("/cpu:0"): # there is no GPU-kernel yet
    depth_output = depth_pool(images)
depth_output.shape
plt.imshow(depth_output[0])

#Simple CNN example
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

from functools import partial

DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")

model = keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=10, activation='softmax'),
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=2, validation_data=(X_valid, y_valid))
score = model.evaluate(X_test, y_test)
X_new = X_test[:10] # pretend we have new images
y_pred = model.predict(X_new)

#ResNet-34 example
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="SAME", use_bias=False)

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
    
model = keras.models.Sequential()
model.add(DefaultConv2D(64, kernel_size=7, strides=2,
                        input_shape=[224, 224, 3]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation="softmax"))

model.summary()

#Using pretrained model with tf.keras.applications
model = keras.applications.resnet50.ResNet50(weights="imagenet")

images_resized = tf.image.resize(images, [224, 224])
plt.imshow(images_resized[0])

images_resized = tf.image.resize_with_pad(images, 224, 224, antialias=True)
plt.imshow(images_resized[0])

images_resized = tf.image.resize_with_crop_or_pad(images, 224, 224)
plt.imshow(images_resized[0])

china_box = [0, 0.03, 1, 0.68]
flower_box = [0.19, 0.26, 0.86, 0.7]
images_resized = tf.image.crop_and_resize(images, [china_box, flower_box], [0, 1], [224, 224])
plt.imshow(images_resized[0])

inputs = keras.applications.resnet50.preprocess_input(images_resized * 255)
Y_proba = model.predict(inputs)

top_K = keras.applications.resnet50.decode_predictions(Y_proba, top=3)
for image_index in range(len(images)):
    print("Image #{}".format(image_index))
    for class_id, name, y_proba in top_K[image_index]:
        print("  {} - {:12s} {:.2f}%".format(class_id, name, y_proba * 100))
    print()
    
#pretrained model for transfer learning 
import tensorflow_datasets as tfds

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname='flower_photos', 
                                   untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

import os
import PIL
import PIL.Image

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[2]))

train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir)
for images, labels in train_ds.take(10):
    plt.imshow(images[20].numpy().astype("uint8"))

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(224, 224),
  batch_size=32)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(224, 224),
  batch_size=32)

train_ds.element_spec
list(train_ds.as_numpy_iterator())

class_names = train_ds.class_names
print(class_names)
n_classes = len(class_names)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
    
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label
#OR
def central_crop(image):
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]])
    top_crop = (shape[0] - min_dim) // 4
    bottom_crop = shape[0] - top_crop
    left_crop = (shape[1] - min_dim) // 4
    right_crop = shape[1] - left_crop
    return image[top_crop:bottom_crop, left_crop:right_crop]

def random_crop(image):
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]]) * 90 // 100
    return tf.image.random_crop(image, [min_dim, min_dim, 3])

def preprocess(image, label, randomize=False):
    if randomize:
        cropped_image = random_crop(image)
        cropped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        cropped_image = central_crop(image)
    resized_image = tf.image.resize(cropped_image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label
#THEN
from functools import partial
train_set = train_ds.shuffle(1000).repeat()
train_set = train_ds.map(partial(preprocess, randomize=False))
valid_set = val_ds.map(preprocess).prefetch(1)

plt.figure(figsize=(10, 10))
for images, labels in train_set.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i] / 2 + 0.5)
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
plt.figure(figsize=(10, 10))
for images, labels in valid_set.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i] / 2 + 0.5)
    plt.title(class_names[labels[i]])
    plt.axis("off")
    

base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)

for index, layer in enumerate(base_model.layers):
    print(index, layer.name)
    

for layer in base_model.layers:
    layer.trainable = False

optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(train_set,
                    steps_per_epoch=int(0.80 * image_count / 32),
                    validation_data=valid_set,
                    validation_steps=int(0.20 * image_count / 32),
                    epochs=1)

for layer in base_model.layers:
    layer.trainable = True

optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9,
                                 nesterov=True, decay=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(train_set,
                    steps_per_epoch=int(0.80 * image_count / 32),
                    validation_data=valid_set,
                    validation_steps=int(0.20 * image_count / 32),
                    epochs=1)




'''RNN'''

#Simple RNN or Deep RNN predicting one value
from numpy import array

train_size = len(dfComb)*75 // 100
train = dfComb[:train_size]
val = dfComb[train_size:]

def window(features, target, window_length=10):
    input_data = features[:-window_length]
    targets = target[window_length:]
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        input_data, targets, sequence_length=window_length, batch_size=32)
    return dataset

train_ds = window(train['returnDuringDay'], train['spyTargetBin']).unbatch()
val_ds = window(val['returnDuringDay'], val['spyTargetBin']).unbatch()

train_ds = np.stack(list(train_ds))
train_feat = np.stack(list(train_ds[:,0])).reshape((len(train_ds),10,1))
train_ds = np.stack(list(train_ds))
train_targ = np.stack(list(train_ds[:,1])).reshape((len(train_ds),1))

val_ds = np.stack(list(val_ds))
val_feat = np.stack(list(val_ds[:,0])).reshape((len(val_ds),10,1))
val_ds = np.stack(list(val_ds))
val_targ = np.stack(list(val_ds[:,1])).reshape((len(val_ds),1))

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1]),
    keras.layers.Dense(1, activation='sigmoid')
    ])
#OR
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(1, activation='sigmoid')
])
#THEN
optimizer = keras.optimizers.Adam(lr=0.005)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

history = model.fit(x=train_feat, y=train_targ, epochs=30, validation_data=(val_feat, val_targ))
                    
def plot_learning_curvesLoss(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    
def plot_learning_curvesAcc(accuracy, val_accuracy):
    plt.plot(np.arange(len(accuracy)) + 0.5, accuracy, "b.-", label="Training Acc")
    plt.plot(np.arange(len(val_accuracy)) + 1, val_accuracy, "r.-", label="Validation Acc")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    
plot_learning_curvesLoss(history.history["loss"], history.history["val_loss"])
plot_learning_curvesAcc(history.history["accuracy"], history.history["val_accuracy"])

#Deep RNN predicting one value but using ret_seq w/ all values
train_size = len(dfComb)*75 // 100
train = dfComb[:train_size]
val = dfComb[train_size:]

def windowFeat(features, window_length=10):
    input_data = features[:-window_length]
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        input_data, targets=None, sequence_length=window_length, batch_size=32)
    return dataset

def windowTarg(features, window_length=10):
    input_data = features[1:-(window_length-1)]
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        input_data, targets=None, sequence_length=window_length, batch_size=32)
    return dataset

train_ds_Feat = windowFeat(train['returnDuringDay']).unbatch()
val_ds_Feat = windowTarg(val['returnDuringDay']).unbatch()
train_ds_Targ = windowFeat(train['spyTargetBin']).unbatch()
val_ds_Targ = windowTarg(val['spyTargetBin']).unbatch()

train_ds_Feat = np.stack(list(train_ds_Feat))
train_ds_Feat = np.stack(list(train_ds_Feat)).reshape((len(train_ds_Feat),10,1))
train_ds_Targ = np.stack(list(train_ds_Targ)).reshape((len(train_ds_Feat),10,1))
val_ds_Feat = np.stack(list(val_ds_Feat))
val_ds_Feat = np.stack(list(val_ds_Feat)).reshape((len(val_ds_Feat),10,1))
val_ds_Targ = np.stack(list(val_ds_Targ)).reshape((len(val_ds_Feat),10,1))

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid'))
])

def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.binary_accuracy(Y_true[:, -1], Y_pred[:, -1])

optimizer = keras.optimizers.Adam(lr=0.005)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[last_time_step_mse])

history = model.fit(x=train_ds_Feat, y=train_ds_Targ, epochs=40, validation_data=(val_ds_Feat, val_ds_Targ))
                    
def plot_learning_curvesLoss(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    
def plot_learning_curvesAcc(accuracy, val_accuracy):
    plt.plot(np.arange(len(accuracy)) + 0.5, accuracy, "b.-", label="Training Acc")
    plt.plot(np.arange(len(val_accuracy)) + 1, val_accuracy, "r.-", label="Validation Acc")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

plot_learning_curvesLoss(history.history["loss"], history.history["val_loss"])
plot_learning_curvesAcc(history.history["last_time_step_mse"], history.history["val_last_time_step_mse"])

#BatchNormalization RNN (doesn't yield great results)
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.BatchNormalization(),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.BatchNormalization(),
    keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid'))
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(x=train_ds_Feat, y=train_ds_Targ, epochs=20, validation_data=(val_ds_Feat, val_ds_Targ))

#Layer Normalization
from tensorflow.keras.layers import LayerNormalization

class LNSimpleRNNCell(keras.layers.Layer):
    def __init__(self, units, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.output_size = units
        self.simple_rnn_cell = keras.layers.SimpleRNNCell(units,
                                                          activation=None)
        self.layer_norm = LayerNormalization()
        self.activation = keras.activations.get(activation)
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        return [tf.zeros([batch_size, self.state_size], dtype=dtype)]
    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs, [norm_outputs]
    
    
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.RNN(LNSimpleRNNCell(10), return_sequences=True,
                     input_shape=[None, 1]),
    keras.layers.RNN(LNSimpleRNNCell(10), return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid'))
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(x=train_ds_Feat, y=train_ds_Targ, epochs=40, validation_data=(val_ds_Feat, val_ds_Targ))

#LSTM
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid'))
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(x=train_ds_Feat, y=train_ds_Targ, epochs=30, validation_data=(val_ds_Feat, val_ds_Targ))

#GRU
np.random.seed(42)
tf.random.set_seed(42)
    
model = keras.models.Sequential([
    keras.layers.GRU(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid'))
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(x=train_ds_Feat, y=train_ds_Targ, epochs=30, validation_data=(val_ds_Feat, val_ds_Targ))

#Conv1D 
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",
                        input_shape=[None, 1]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid'))
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(train_ds_Feat, train_ds_Targ[:, 3::2], epochs=60,
                    validation_data=(val_ds_Feat, val_ds_Targ[:, 3::2]))

#WaveNet
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[None, 1]))
for rate in (1, 2, 4, 8) * 2:
    model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal",
                                  activation="relu", dilation_rate=rate))
model.add(keras.layers.Conv1D(filters=1, kernel_size=1, activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(x=train_ds_Feat, y=train_ds_Targ, epochs=30, validation_data=(val_ds_Feat, val_ds_Targ))






'''NLP'''
