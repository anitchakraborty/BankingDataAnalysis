import pandas as pd 
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder
import random
from sklearn.linear_model import LinearRegression

def bank(path, val_ratio):
    df=pd.read_csv(path)
    ls = df.education.unique().tolist()
    enc_ed = np.array(list(map(lambda x:ls.index(x),df.education.tolist()))).reshape(-1,1)
    enc_h = np.array(list(map(lambda x:0 if x=='yes' else 1,df.housing.tolist()))).reshape(-1,1)
    y = np.array(list(map(lambda x: 1 if x=='nonexistent' else (2 if x=='success' else 3),df.poutcome.tolist()))).reshape(-1,1)
    x = np.hstack((enc_ed,enc_h))
    idx = list(range(x.shape[0]))
    random.shuffle(idx)
    valamount = int(x.shape[0]*val_ratio)
    xtrain = x[idx][valamount:]
    xtest = x[idx][:valamount]
    ytrain = y[idx][valamount:]
    ytest = y[idx][:valamount]
#    import pdb;pdb.set_trace()
    lr = LinearRegression().fit(xtrain,ytrain)
    pred = lr.predict(xtest)
    pred = np.round(pred,0)
    error = np.mean(np.power(np.abs(np.power(pred,2) - np.power(ytest,2)),0.5))
    accuracy = 1 - error
    return accuracy
    
path = 'D:\\Executable\\banking.csv'
print(bank(path,0.1))