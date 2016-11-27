# Gradient Descent and Linear Regression
# -- compare the co-efficient and intercept parameters obtained by the two approaches
#    they are expected to be same or very close

import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

def step_gradient(b_current, m_current, xs, ys, learningRate, epsilon):
    b_gradient = 0
    m_gradient = 0
    stop = 0
    N = float(len(xs))
    for i in range(0, len(xs)):
        b_gradient += -(2/N) * (ys[i][0] - ((m_current*xs[i][0]) + b_current))
        m_gradient += -(2/N) * xs[i][0] * (ys[i][0] - ((m_current * xs[i][0]) + b_current))

    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    if( np.abs(b_gradient) < epsilon and np.abs(m_gradient) < epsilon ):
       stop=1

    return [new_b, new_m, stop]

def gradient_descent_run( x_data, y_data, start_b, start_m, learning_rate, epsilon ):
   b = start_b
   m = start_m
   while True:
      b,m,stop = step_gradient( b,m, x_data, y_data, learning_rate, epsilon)
      if( stop==1 ):
         break;

   return [b,m]

df = pd.read_csv("dataset_simpleRegression.csv")
x_train = df[df.columns[0]].values.reshape(-1,1)
y_train = df[df.columns[1]].values.reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit( x_train, y_train )
score = regr.score(x_train,y_train)

print regr
print 'Coefficients:', regr.coef_
print 'Intercept:', regr.intercept_
print 'Score: ', score

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regr.predict(x_train),color='red',linewidth=3)
plt.xticks(())
plt.yticks(())
#plt.show()

learning_rate=0.001
epsilon=0.00001
[b,m] = gradient_descent_run(x_train,y_train,0,0,learning_rate,epsilon)

print "b=",b,"m=",m
