import random
import matplotlib.pyplot as plt
import math



class myApp(object):
    def __init__(self, app_name,degree = None):
        if degree is None:
            valid_degrees = [0,1,2]
            degree = random.sample(valid_degrees,1)[0]
            equation_params = list()
            # equation params are x^0, x^1 ... x^degree
            for coeff in range(0, degree+1):
                equation_params.append(random.randrange(0, 1000, 1))
            y = [ self.getyval(equation_params, k) for k in range(0, 50) ]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(app_name)
            ax.set_xlabel('X values')
            ax.set_ylabel('Y Values')
            ax.plot(range(0,50), y, '--', )
            fig.savefig(app_name + '.png')

    def getyval(self, equation_params, xval):
        result = 0.0
        for k in range(len(equation_params)):
            result = result + equation_params[k]*math.pow(xval,k)
        return result

if __name__ == '__main__':
    app = dict()
    for k in range(0,10):
        app[k] = myApp('App'+str(k))
