import numpy as np
import matplotlib.pyplot as plt
import time
from collections import Counter
import pandas as pd

def load_data(train_file, test_file):
    df = pd.read_csv(train_file)
    x_train = df.drop('label', axis=1)
    y_train = df['label']
    x_test = pd.read_csv(test_file)
    #print x_train.head()
    #print y_train.head()
    #print x_test.head()
    return x_train, y_train, x_test

def show_sample( sample_df, y_train, digits, num_sample ):
    num_digits = len(digits)
    for y, cls in enumerate(digits):
       idxs = np.nonzero([i==y for i in y_train ])
       idxs = np.random.choice(idxs[0], num_sample, replace=False )
       for i, idx in enumerate(idxs):
           plt_idx = i * num_digits + y + 1
           plt.subplot( num_sample, num_digits, plt_idx )
           plt.imshow( sample_df.iloc[idx].values.reshape((28,28)))
           plt.axis("off")
           if i==0: 
              plt.title(cls)


    plt.show()

class simple_knn:
   def __init__(self):
      pass

   def train(self, x, y):
      self.X_train = x
      self.Y_train = y

   def predict(self, X, k=1):
       dists = self.comp_euclidian_dist(X)
       num_test = dists.shape[0]
       y_pred = np.zeros(num_test)
       for i in range(num_test):
           k_closest_y=[]
           labels = self.Y_train[np.argsort(dists[i,:])].flatten()
           k_closest_y = labels[:k]
           c = Counter(k_closest_y)
           y_pred[i] = c.most_common(1)[0][0]

       return y_pred

   def comp_euclidian_dist(self, X):
       num_test = X.shape[0]
       num_train = self.X_train.shape[0]
       dot_pro = np.dot(X, self.X_train.T)
       
       sum_square_test = np.square(X).sum(axis=1)
       sum_square_train = np.square(self.X_train).sum(axis=1)
       dists = np.sqrt(-2*dot_pro + sum_square_train + np.matrix(sum_square_test).T )
       
       return dists


X_train, Y_train, X_test = load_data("data/digit_recognizer/train.csv", "data/digit_recognizer/test.csv")
classes = ["0","1","2","3","4","5","6","7","8","9"]
sample_per_class=8
show_sample( X_train, Y_train, classes, sample_per_class)
classifier = simple_knn()
classifier.train( X_train, Y_train )
predictions = classifier.predict( X_test.head(10).values, classifier )
#for i in range(len(predictions))
print predictions.head(10)
