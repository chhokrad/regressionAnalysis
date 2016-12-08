
# coding: utf-8

# In[1]:

import pandas
from collections import namedtuple
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn import ensemble

XYdata = namedtuple('XYdata',['X','Y'])

excel_file = pandas.read_excel('matched.xlsx')
excel_col_names = list(excel_file.columns)
print(excel_col_names)

filtered_data_col_names = excel_col_names[excel_col_names.index('benchmark_id'): excel_col_names.index('latency')+1]
filtered_data = pandas.DataFrame(excel_file, columns = filtered_data_col_names)
app_names = list(set(filtered_data['benchmark_id'].values))
#app_names.remove('benchmark_id')
app_data = dict()
for k in app_names:
    app_data[k] = filtered_data[filtered_data['benchmark_id']==k]
app_data_XY = dict()
X_col_names = filtered_data_col_names[filtered_data_col_names.index('benchmark_id')+1: filtered_data_col_names.index('latency')]
print('Xcols',X_col_names)
Y_col_names = list()
Y_col_names.append(filtered_data_col_names[-1])
print('Ycols',Y_col_names)
for k in app_data:
    app_data_XY[k] = XYdata(pandas.DataFrame(app_data[k], columns = X_col_names), pandas.DataFrame(app_data[k], columns = Y_col_names))

import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import ShuffleSplit
from itertools import combinations
from scipy.stats.stats import pearsonr
FeatureSelection = namedtuple('FeatureSelection',['Sorted_feature_vector', 'Score'])
alpha = 0.95
params = {'n_estimators':500, 'max_depth':4, 'min_samples_split':9, 'learning_rate':0.01,'loss':'ls'}
FestureSelectionData = dict()
for app_name in app_data_XY:
    print(app_name)
    X_data = app_data_XY[app_name].X.values
    Y_data = app_data_XY[app_name].Y.values[:,-1]
    myEst = ensemble.GradientBoostingRegressor(**params)
    #myEst = SVR(kernel= 'linear')
    mySel = RFECV(estimator=myEst, scoring='neg_mean_absolute_error',  cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0), step=1)
    mySel = mySel.fit(X_data, Y_data)
    selected_features = [X_col_names[i] for i,k in enumerate(mySel.ranking_) if k == 1]
    selected_features_ = zip(selected_features, mySel.estimator_.feature_importances_)
    sorted_selected_features = sorted(selected_features_, key = lambda feature : feature[1])
    FestureSelectionData[app_name] = FeatureSelection(sorted_selected_features, max(mySel.grid_scores_))
print(FestureSelectionData)

pearsonData = dict()
for app_name in app_data_XY:
    nc2 = list(combinations([k[0] for k in FestureSelectionData[app_name].Sorted_feature_vector], 2))
    temp = dict()
    for com in nc2:
        temp[tuple(com)] = pearsonr((app_data_XY[app_name].X[com[0]].values).tolist(),(app_data_XY[app_name].X[com[1]].values).tolist())
    pearsonData[app_name] = temp


from copy import deepcopy
FeatureCorrData = dict()
for app_name in FestureSelectionData:
    print app_name
    feature_vectors = [feature[0] for feature in FestureSelectionData[app_name].Sorted_feature_vector]
    app_pearson_keys = pearsonData[app_name].keys()
    upp_thresh = 0.8
    low_thresh = -0.8
    feature_copy = set(deepcopy(feature_vectors))
    for feature in reversed(feature_vectors):
        if feature in feature_copy:
            for rest_feature in feature_copy.difference(set([feature])):
                if (feature, rest_feature) in app_pearson_keys:
                    if pearsonData[app_name][feature, rest_feature][0] <= upp_thresh and pearsonData[app_name][feature, rest_feature][0] >= low_thresh:
                       continue
                    else:
                        feature_copy = feature_copy.difference(set([rest_feature]))
                elif (rest_feature, feature) in app_pearson_keys:
                    if pearsonData[app_name][rest_feature, feature][0] <= upp_thresh and pearsonData[app_name][rest_feature, feature][0] >= low_thresh:
                       continue
                    else:
                        feature_copy = feature_copy.difference(set([rest_feature]))
    FeatureCorrData[app_name] = list(feature_copy)

for app_name in FeatureCorrData:
    print app_name
    print FeatureCorrData[app_name]
    print len(FeatureCorrData[app_name])


from sklearn import metrics
from sklearn.model_selection import cross_val_score
EstimatorData = dict()
for app_name in FeatureCorrData:
    print app_name
    cols = FeatureCorrData[app_name]
    X_data = pandas.DataFrame(app_data_XY[app_name].X, columns=cols)
    X_data_ = X_data.values
    Y_data_ = app_data_XY[app_name].Y.values[:,-1]
    scores = cross_val_score(myEst, X_data_, Y_data_, cv= ShuffleSplit(n_splits=10, random_state=0,test_size=0.2), scoring = 'neg_mean_absolute_error')
    print scores.mean()
    print FestureSelectionData[app_name].Score

alpha = 0.95
params = {'n_estimators':500, 'max_depth':4, 'min_samples_split':9, 'learning_rate':0.01,'loss':'huber'}
myEst1 = ensemble.GradientBoostingRegressor(**params)
from sklearn import metrics
from sklearn.model_selection import cross_val_score
EstimatorData = dict()
for app_name in FeatureCorrData:
    print app_name
    cols = FeatureCorrData[app_name]
    X_data = pandas.DataFrame(app_data_XY[app_name].X, columns=cols)
    X_data_ = X_data.values
    Y_data_ = app_data_XY[app_name].Y.values[:,-1]
    scores = cross_val_score(myEst1, X_data_, Y_data_, cv= ShuffleSplit(n_splits=5, random_state=0,test_size=0.2), scoring = 'neg_mean_absolute_error')
    print scores.mean()
    print FestureSelectionData[app_name].Score


import pickle
pickle.dump(app_data_XY, open('app_data_XY.p', 'wb'))
pickle.dump(FestureSelectionData, open('FeatureSelectionData.p', 'wb'))
pickle.dump(pearsonData, open('pearsonData.p', 'wb'))
pickle.dump(FeatureCorrData, open('FeatureCorrData.p', 
for app_name in FeatureCorrData:
    print app_name
    cols = FeatureCorrData[app_name]
    X_data = pandas.DataFrame(app_data_XY[app_name].X, columns=cols)
    X_data_ = X_data.values
    Y_data_ = app_data_XY[app_name].Y.values[:,-1]
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'huber'}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X_data_, Y_data_)
    EstimatorData[app_name] = clf.estimators_
