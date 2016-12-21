import pandas
from my_namedtuples import FeatureSelection, XYdata
import sys
import numpy
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.model_selection import ShuffleSplit
from itertools import combinations
from scipy.stats.stats import pearsonr
from copy import deepcopy
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
import pickle
import warnings
from sklearn import linear_model
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# arg 0: excel file path
# arg 1: application id column name
# arg 2: input feature label list
# arg 3: output feature label list
# arg 4: [prev, curr] state label list
# arg 5: Correlaton Threshold upper
# arg 6: Correlation Threshold lower

filenames  = ['precision_server.csv', 'isis2.csv', 'meta2.csv', 'isislab9.csv']
app_id_label = 'benchmark_id'
complete_input_feature_list = [u'host_IPC', u'host_IPS', u'host_cache_misses', u'host_cache_ref', 
                               u'host_cpupercent', u'host_cs', u'host_disk_io', u'host_kvm_exit', 
                               u'host_llc_bw', u'host_mem_bw', u'host_memory', u'host_net_io', 
                               u'host_sched_iowait', u'host_sched_switch', u'host_sched_wait']
complete_output_feature_list = [u'latency']
prev_state_label = 'index_prev'
curr_state_label = 'index'
thres_upper = 0.8
thresh_lower = -0.8

alpha = 0.95
params = {'n_estimators':500, 'max_depth':4, 'min_samples_split':9, 'learning_rate':0.01,'loss':'ls'}
CompleteData = dict()

for filename in filenames:
    excel_file = pandas.read_csv(filename)
    excel_file_ = excel_file[numpy.isfinite(excel_file[prev_state_label])]
    app_names = list(set(excel_file[app_id_label].values))
    app_data = dict()
    for app_name in app_names:
        temp = excel_file_[excel_file_[app_id_label] == app_name]
        X_new = pandas.DataFrame(temp, columns=complete_input_feature_list)
        previous_indices = temp[prev_state_label].values.tolist()
        X_prev = pandas.DataFrame()
        for index in previous_indices:
            X_prev = X_prev.append(pandas.DataFrame(excel_file[excel_file[curr_state_label] == int(index)], 
                                       columns = complete_input_feature_list))
        Y = pandas.DataFrame(temp, columns=complete_output_feature_list)
        assert len(X_prev) == len(X_new) == len(Y), 'Data of unequal length'
        app_data[app_name] = XYdata(X_prev, X_new, Y)

    FestureSelectionData = dict()
    for app_name in app_names:
        X_data = app_data[app_name].X_new_frame.values
        Y_data = app_data[app_name].Y_frame.values[:,-1]
        assert len(X_data) == len(Y_data), 'Length is not same'
        myEst = ensemble.GradientBoostingRegressor(**params)
        mySel = RFECV(estimator=myEst, scoring='neg_mean_squared_error',  cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0), step=1)
        mySel = mySel.fit(X_data, Y_data)
        selected_features = [complete_input_feature_list[i] for i,k in enumerate(mySel.ranking_) if k == 1]
        selected_features_ = zip(selected_features, mySel.estimator_.feature_importances_)
        sorted_selected_features = sorted(selected_features_, key = lambda feature : feature[1])
        FestureSelectionData[app_name] = FeatureSelection(sorted_selected_features, max(mySel.grid_scores_))
    print(FestureSelectionData)
    
    for app_name in app_names:
        print app_name
        print len(FestureSelectionData[app_name].Sorted_feature_vector)
        for pair in FestureSelectionData[app_name].Sorted_feature_vector:
            print pair
        
    pearsonData = dict()
    for app_name in app_names:
        nc2 = list(combinations([k[0] for k in FestureSelectionData[app_name].Sorted_feature_vector], 2))
        temp = dict()
        for com in nc2:
            temp[tuple(com)] = pearsonr((app_data[app_name].X_new_frame[com[0]].values).tolist(),(app_data[app_name].X_new_frame[com[1]].values).tolist())
        pearsonData[app_name] = temp
    print pearsonData
    
    FeatureCorrData = dict()
    for app_name in app_names:
        feature_vectors = [feature[0] for feature in FestureSelectionData[app_name].Sorted_feature_vector]
        app_pearson_keys = pearsonData[app_name].keys()
        feature_copy = set(deepcopy(feature_vectors))
        for feature in reversed(feature_vectors):
            if feature in feature_copy:
                for rest_feature in feature_copy.difference(set([feature])):
                    if (feature, rest_feature) in app_pearson_keys:
                        if pearsonData[app_name][feature, rest_feature][0] <= thres_upper and pearsonData[app_name][feature, rest_feature][0] >= thresh_lower:
                           continue
                        else:
                            feature_copy = feature_copy.difference(set([rest_feature]))
                    elif (rest_feature, feature) in app_pearson_keys:
                        if pearsonData[app_name][rest_feature, feature][0] <= thres_upper and pearsonData[app_name][rest_feature, feature][0] >= thresh_lower:
                           continue
                        else:
                            feature_copy = feature_copy.difference(set([rest_feature]))
        FeatureCorrData[app_name] = list(feature_copy)
        
    for app_name in FeatureCorrData:
        print app_name
        print FeatureCorrData[app_name]
        print len(FeatureCorrData[app_name])
        

    EstimatorData = dict()
    for app_name in app_names:
        print app_name
        cols = FeatureCorrData[app_name]
        X_data_ = pandas.DataFrame(app_data[app_name].X_new_frame, columns=cols).values
        Y_data_ = app_data[app_name].Y_frame.values[:,-1]
        scores = cross_val_score(myEst, X_data_, Y_data_, cv= ShuffleSplit(n_splits=10, random_state=0,test_size=0.2), scoring = 'neg_mean_squared_error')
        clf = ensemble.GradientBoostingRegressor(**params)
        clf.fit(X_data_, Y_data_)
        
        print scores.mean()
        print FestureSelectionData[app_name].Score
        
        X_new_frame = pandas.DataFrame(app_data[app_name].X_new_frame, columns=cols)
        X_prev_frame = pandas.DataFrame(app_data[app_name].X_prev_frame, columns=cols)
        
        mul_clf_dict = dict()
        for feature in cols:
            print "   " + feature
            X_new = X_new_frame[feature].values
            X_prev = X_prev_frame[feature].values
            assert len(X_new) == len(X_prev), 'Vectors Don\'t match' + str(len(X_new)) + ' ' + str(len(X_prev))
            clf1_ = SVR('rbf')
            scores = cross_val_score(clf1_ , numpy.asarray(X_prev).reshape(len(X_prev),1), X_new, cv= ShuffleSplit(n_splits=10, random_state=0,test_size=0.2), scoring = 'neg_mean_squared_error')
            print "   " + str(scores.mean())
            mul_clf_ = linear_model.Lasso(alpha=.1)
            mul_clf_.fit(numpy.asarray(X_prev).reshape(len(X_prev),1), X_new)
            mul_clf_dict[feature] = mul_clf_
        EstimatorData[app_name] = (clf, mul_clf_dict, FeatureCorrData[app_name])
    
    CompleteData[filename] = {'EstimatorData': EstimatorData, 'FeatureCorrelationData': FeatureCorrData, 
                              'FeatureSelectionData': FestureSelectionData, 'pearsonData': pearsonData}    

pickle.dump(CompleteData, open('AllHardwareData.p', 'wb'))

