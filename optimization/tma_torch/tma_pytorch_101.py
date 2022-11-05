# this script is a first version of using pytorch to optimize TMA envelope, worfklow is following:
#   1. get data from csv - aggregate it to desired timeframes 
#   2. calculate some stds and smas on data without lookahead bias 
#   3. get good entry/exit points via lookahead data processing 
#   4. pytorch train test split 
#   5. plot model results 
#
#


import torch 
import sys 
import numpy as np 
from sklearn.model_selection import train_test_split 
sys.path.append("../..")
from myfuns import myfuns as mf
from mymath.mymath import myMath
import pandas as pd 
import matplotlib.pyplot as plt 
from lookahead_entries import *
import torch 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
import torch.nn as nn 
# 1.  get data 
if 1:
    filename='BTC-USD2022-08-24_2022-08-27'
#    filename='BTC-USD2021-06-05_2022-06-05'
    filename=f'../../data/{filename}.csv'
    raw_cols=['open','close','low','high','timestamp']
    df=mf.read_csv(filename=filename,add_index=True,cols=raw_cols)
    m=myMath()
    m.math_df=df
    df=m.aggregate(df=m.math_df, scale =5, src_col='timestamp')
    df['candle']=df['close']-df['open']
        
# 2. add features to data  -> smas and stds on various timeframes 
    # get relevant timeframes 
timeframes=mf.get_timeframes(dist_fun='exp',    timeframes_ranges=[1,150],N=4)
for tf in timeframes:
    df[f'sma-{tf}']=m.calculate_fun(df=df,fun_name='sma',col='close',window=tf,inplace=False)
    df[f'std-{tf}']=m.calculate_fun(df=df,fun_name='std',col='candle',window=tf,inplace=False)

# 3. add  LONG, SHORT  lookahead signals 
df, p_longs,p_shorts,percentile_longs,percentile_shorts,lowest_longs,highest_shorts = example_workflow(df)

# 4. plot lookahead signals to marvel at my marvelous plot of lookahead signals :
if 0:
    shorts_ser=df['high'][highest_shorts]
    longs_ser=df['low'][lowest_longs]
    f1,ax2=mf.plot_candlestick(df=df,shorts_ser=shorts_ser,longs_ser=longs_ser)
    plt.show()
    
# 5 training loop ! 
df.dropna(how='any',inplace=True)
features_cols=['sma-2','sma-7','sma-20','sma-54','sma-148','std-2','std-7','std-20','std-54','std-148']
labels_cols=['SHORT']

features=df[features_cols].to_numpy()
labels=df[labels_cols].astype(int).to_numpy()

# make tensors 
X=torch.from_numpy(features.astype(np.float32))
Y=torch.from_numpy(labels.astype(np.float32))
n_samples, n_features = X.shape 
# train test split 
X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=1234)
# scaling 
sc=StandardScaler() # Z-scoring our features - this line uses lookahead most likely, z scoring should be done on lookback windows
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test  = torch.from_numpy(X_test.astype(np.float32))

class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear=nn.Linear(n_input_features,1)
        
    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x) )
        return y_predicted
    
model=LogisticRegression(n_features)
# 2) loss and optimizer
criterion = nn.BCELoss() # binary cross entropy 
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(100):
    # prediction 
    y=model(X_train) # calling the module class invokes forward function fren ! loss calculation
    l=criterion(y,Y_train) # order here is important ! 
    # graduebt 
    l.backward()    # using backward method to calcylate dl/dw gradient 
    # update weights 
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 ==0:
        print(epoch)
    
    
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    print(acc)
    
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)
exit(1)
    
    

exit(1)
if 1:
    x1y1= [
        [ df['index'],df['close']],
        [ df['index'][percentile_longs], df['close'][percentile_longs] ],
         [ df['index'][percentile_shorts], df['close'][percentile_shorts] ]

    ] 
    x2y2= [ 
        [ df['index'],df['close']],
         [ df['index'][lowest_longs], df['close'][lowest_longs] ],
          [ df['index'][highest_shorts], df['close'][highest_shorts] ]
    
           ]

    print(df[lowest_longs | highest_shorts])
    print(df.columns)
    
    mf.basic_plot(x1y1=x1y1,x2y2=x2y2,linez=['-xk','^g','vr'])



#plt.plot(timeframes,'o')
#plt.show()

