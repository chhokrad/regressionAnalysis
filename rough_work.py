import random
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.cluster.vq import  kmeans2, whiten


class myApp(object):
    def __init__(self, app_name,degree = None, equation_coeff = None, x = None, y = None):
        '''
        degree is the max degree of the underlying model
        equation_coeff is a list of coeff of underlying math equation
        '''
        assert app_name != '', 'Name cannot be empty'
        self.app_name = app_name
        if degree is None:
            valid_degrees = [0,1,2]
            degree = random.sample(valid_degrees,1)[0]

        self.equation_params = list()
        if equation_coeff is None:
            # equation params are x^0, x^1 ... x^degree
            for coeff in range(0, degree+1):
                self.equation_params.append(random.randrange(0, 100, 1))
        else:
            assert len(equation_coeff) == degee + 1, 'Supplied Equation Coeff Data is inconsistent with degree'
            self.equation_params = equation_coeff
            self.equation_params_estimate = equation_coeff


        if x is None:
            self.x = range(0, 50)
        else:
            self.x = x

        if y is None:
            self.y = [ self.getyval(self.equation_params, k) for k in self.x ]
            self.y_with_noise = self.addNoise(self.y)

        else:
            self.y = y
            self.y_with_noise = y
            self.y_estimate = y

        self.equation_params_estimate = self.regressionanalysis(self.x, self.y_with_noise)[::-1]
        self.y_estimate = [self.getyval(self.equation_params_estimate, k) for k in self.x ]

        assert len(self.x) == len(self.y), 'Supplied X and Y data is inconsistent'

        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.set_title(app_name)
        ax.set_xlabel('X values')
        ax.set_ylabel('Y Values')
        ax.text(0.3, 0.9, self.equation_params, horizontalalignment='center',
                verticalalignment='center', transform = ax.transAxes)
        ax.plot(self.x, self.y, 'o' )

        ax = fig.add_subplot(212)
        ax.set_xlabel('X values')
        ax.set_ylabel('Y with noise')
        ax.plot(self.x,self.y_with_noise, '*')
        ax.plot(self.x,self.y_estimate, '--')
        ax.text(0.3, 0.9, [round(k,2) for k in self.equation_params_estimate] ,
                horizontalalignment='center', verticalalignment='center',
                transform = ax.transAxes)
        fig.savefig(app_name + '.png')

    def getyval(self, equation_params, xval):
        result = 0.0
        for k in range(len(equation_params)):
            result = result + equation_params[k]*math.pow(xval,k)
        return result

    def addNoise(self, data, mean =  None, standard_deviation = None, ):
        if mean is None:
            mean = 0
        if standard_deviation is None:
            standard_deviation = 10.0
        numsamples = len(data)
        noise_vec = np.random.normal(mean, standard_deviation, numsamples)
        return [sum(x) for x in zip(noise_vec, data)]

    def Distance(self, AppObject):
        diff = [a - b for (a , b ) in zip([self.getyval(self.equation_params, k) for k in AppObject.getXYDataEst()[0]], AppObject.getXYDataEst()[1])]
        # calculating L2 norm
        return(np.linalg.norm(diff))


    def regressionanalysis(self, x_data, y_data):
        return list(np.polyfit(x_data, y_data, 2))

    def getXYData(self):
        return (self.x, self.y_with_noise)

    def getXYDataEst(self):
        return(self.x,self.y_estimate)

    def getEquationParam(self):
        return self.equation_params

    def getEquationParamEst(self):
        return self.equation_params_estimate

    def get_app_name(self):
        return self.app_name

if __name__ == '__main__':
    training_data_app = dict()
    for k in range(0,10):
        training_data_app[k] = myApp('App'+str(k))

    # generating x data for testing app
    x_new = xrange(45, 600, 35)
    test_app = myApp('TestingApp', degree = None, equation_coeff = None, x = x_new, y = None)

    result = [training_data_app[k].Distance(test_app) for k in training_data_app]
    print('The closest one is ', training_data_app[result.index(min(result))].get_app_name(), min(result))

    print('--Sanity Check---')

    for a in training_data_app:
        print('Testing' , training_data_app[a].get_app_name())
        result_ = [training_data_app[k].Distance(training_data_app[a]) for k in training_data_app]
        print('The closest one is ', training_data_app[result_.index(min(result_))].get_app_name(), min(result_))

    print('-- Clustering --')
    training_data_app_dict = dict()
    feacture_vec = list()
    for k in training_data_app:
        training_data_app_dict[tuple(training_data_app[k].getEquationParamEst())] = training_data_app[k].get_app_name()
        feacture_vec.append(training_data_app[k].getEquationParamEst())
    (centroid, label) = kmeans2(whiten(feacture_vec), 4)
    print([(training_data_app_dict[tuple(feacture_vec[k])], label[k])  for k in range(0, len(feacture_vec))])
