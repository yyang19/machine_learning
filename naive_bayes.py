#
# Single feature formula
#
#           P(x|Y) * P(Y)
# P(Y|x) = ---------------
#               P(x)
#
# Multiple feature formula, X=(x1,x2,..., xn)
#
#           P(x1|Y) * P(x2|Y) * ... P(xn|Y) * P(Y)
# P(Y|X) = ---------------------------------------
#              P(x1) * P(x2) * ... * P(xn)
#

# dataset: Pima Indians Diabetes problem

import pandas as pd
import sklearn
import math

def loaddata(filepath, cols):
    df=pd.read_csv(filepath, names=cols )
    print "File", filepath, "loaded"
    return df

def df_dump( df ):
    print "Data frame shape:", df.shape 
    print "Data frame describe: \n", df.describe() 
    print "Data preview: \n", df.head()

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def dev(numbers):
    avg = mean(numbers)
    variance = sum( [pow(x-arg,2) for x in numbers])/float(len(numbers)-1)
    return sqrt(variance)

#We can use a Gaussian function to estimate the probability of a 
#given attribute value, given the known mean and standard deviation 
#for the attribute estimated from the training data. 
#The return value is the conditional probability of a given 
#attribute value given a class value.

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def buildStat( df, column_names ):
    # statistics table preparation
    cols = [ "class","feature", "stat" ]
    df_stat = pd.DataFrame( columns = cols )
    feature_count = len(column_names)-1
    for i in range( 0, feature_count ):
        df_stat.loc[len(df_stat)] = [0, column_names[i], [df[column_names[i]][df['result']==0].mean(), df[column_names[i]][df['result']==0].std()]] 
        df_stat.loc[len(df_stat)] = [1, column_names[i], [df[column_names[i]][df['result']==1].mean(), df[column_names[i]][df['result']==1].std()]]

    return df_stat

def calStat( cls, df, feature ):
    avg = df["stat"].loc[(df['class']==cls) & (df['feature']==feature)].values[0][0]
    std = df["stat"].loc[(df['class']==cls) & (df['feature']==feature)].values[0][1]
    return avg, std

def calClassProbability( cls, X, df_stat, features ):
    p = 1
    for i in range(len(X)):
        avg, std = calStat( cls, df_stat, features[i] )
        p *= calculateProbability(X[i], avg, std)

    return p

def predict( X, df_stat, features, prior_prob_pos, prior_prob_neg ):
    return calClassProbability( 1, X, df_stat, features ) * prior_prob_pos >  calClassProbability( 0, X, df_stat, features ) * prior_prob_neg

def getPredictions(Xs, df_stat, features, prior_prob_pos, prior_prob_neg):
    predictions = []
    for i in range(len(Xs)):
        result = predict(Xs[i], df_stat, features, prior_prob_pos, prior_prob_neg)
        predictions.append(result)
    
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i] == predictions[i]:
            correct += 1

    return (correct/float(len(testSet))) * 100.0

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def splitData( df, ratio ):
    df_head = df.head( int(len(df.index) * ratio) )
    df_tail = df.tail( int(len(df.index) * (1-ratio)) )

    return df_head, df_tail

if __name__ == "__main__":
    column_names=["col1","col2","col3","col4","col5","col6","col7","col8","result"]
    df = loaddata("data/pima-indians-diabetes.csv", column_names)
    #split dataset into two part for traning and test
    train_portion=0.8
    df_train, df_test = splitData( df, train_portion )
    prior_prob_pos = df_train['result'].sum()/float(len(df_train))
    prior_prob_neg = 1- prior_prob_pos
    df_stat = buildStat( df_train, column_names )
    X=[6, 148, 72, 35, 0, 33.6, 0.627, 50]
    print X, predict( X, df_stat, column_names, prior_prob_pos, prior_prob_neg )
    X=[1, 85, 66, 29, 0, 26.6, 0.351, 31 ]
    print X, predict( X, df_stat, column_names, prior_prob_pos, prior_prob_neg )
    predicts = getPredictions( df_test.drop(["result"], axis=1).values, df_stat, column_names, prior_prob_pos, prior_prob_neg )
    print "Accuracy",getAccuracy(df_test["result"].values, predicts)
    

# prediction
