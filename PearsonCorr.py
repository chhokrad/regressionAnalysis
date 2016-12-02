'''
Created on Nov 13, 2016

@author: ajaychhokra
'''




import pandas
from itertools import combinations
from scipy.stats.stats import pearsonr, mode
from _collections import defaultdict
from numpy import median

if __name__ == '__main__':
    #g = FrameGrabber.FrameGrabber('Demo2016_VU_NCSU_WideScreen.mp4', False)
    excel_file = pandas.read_excel('results4.xlsx')
    benchmark_ids = list(set(excel_file['benchmark_id']))
    app_data_filtered = dict()
    feature_names = list(excel_file.columns)[2:17]
    myData = dict()
    for k in benchmark_ids:
        app_data_filtered[k] = excel_file[excel_file['benchmark_id'] ==k]
        temp = dict()
        for l in feature_names:
            temp[l] = app_data_filtered[k][l]
        myData[k] = temp
    
    nc2 = list(combinations(feature_names,2))
    result = dict()
    
    for app in myData:
        temp = dict()
        for my_combination in nc2:
            temp[my_combination] = pearsonr(myData[app][my_combination[0]], myData[app][my_combination[1]])
        result[app] = temp
    
    result_ = defaultdict(list)
    
    for k in result:
        for l in result[k]:
            result_[l].append(result[k][l][0])
    
    for k in result:
        print('------------------------------------------------------------------')
        print('Name of the app is', k)
        for l in result[k]:
            print('**************************************************************')
            print('Correlation between')
            print(l)
            print('is :', result[k][l])
            print('**************************************************************')
        print('------------------------------------------------------------------')
    print('#################################################################')    
    for k in result_:
        print('------------------------------------------------------------------')
        print('Combination is :')
        print(k)
        print('coefts are for different apps are :')
        print(result_[k])
        print('average is :', sum(result_[k])/float(len(result_[k])))
        print('median is : ', median(result_[k]))
        print('mode is: ', mode(result_[k]))
        
    
    
    

        
        
    
    
        
        