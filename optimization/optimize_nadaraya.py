# this script optimizes nadaraya watson envelope to perform better on training set 
# currently:
#   it preps df to be optimized 
#       it gets columns used for statistical model from math df 
#       it brings lookahead signals from lookahead 

# goal function:
    # maximize catching long entry points 
    # maximize catching short entry points 
    # minimize catching non signal points 

# model parameters :
    # nadaraya watson's bandwidth
    # backwindow std 
    # backwindow ema distance 
    # backwindow ema integral 
import sys
from turtle import right
sys.path.append("..") # makes you able to run this from parent dir 
from myfuns import myfuns as mf 
mf.how_long=mf.how_long()
import numpy as np 
import pandas as pd 
print(mf.how_long(), ' imports')
from mymath.mymath import myMath    
print(mf.how_long(), ' mymath import')
from lookahead_entries import lookahead_score,example_workflow
print(mf.how_long(), ' lookahead import')

# ------------------------------------------------------ funs  
# wrapper for passing iterable as X and getting rid of lookahead calculations 
def nadaraya_wrapper(X,Xi,Yi,h=1.0,window=50):
    Y=[]
    for no,x in enumerate(X):
        start=0
        if no>window:
            start=no-window 
        Xii=Xi.iloc[start:no+1]
        Yii=Yi.iloc[start:no+1]
        y=yet_another_nadaraya(x,Xi=Xii,Yi=Yii,h=1.0)
        
        Y.append(y)
    return Y
# returrns y(x) nadaraya watson function estimation
def yet_another_nadaraya(x,Xi,Yi,h=1.0):
    norm_z = lambda x,mu,sigma : 1/(sigma * np.sqrt (2* np. pi )) * np.exp ( -1/2 * ( (x-mu)/sigma  )**2  )
    norm= lambda x,mu : norm_z(x,mu,1)     
    foo=np.sum([norm(x,xj) for xj in Xi ]) # cant seem to think of obvious way to optimize this step - new point has new values of norms
    w=[]
    y=[]
    for xi,yi in zip(Xi,Yi): # syntax from python engineer channel, youtube is my teacher ! 
        wi=norm(x/h,xi)/foo
        w.append(wi)
        yi=wi*yi/h
        y.append(yi)
    return np.sum(y)
    
# ------------------------------------------------------ stuff 
if 0:
    df=mf.test_population()
    df['close']=df['Yi']
    df['index']=df['Xi']
    m=myMath()
    df['nad']=nadaraya_wrapper(X=df['index'], Xi=df['index'],Yi=df['close'],h=1.0,window=50)
    df['nad2']=m.rolling_nadaraya(df=df,xcol='index',ycol='close',h=1)
    import matplotlib.pyplot as plt 
    plt.plot(df['index'],df['nad'],'--b')
    plt.plot(df['index'],df['nad2'],'or')
    plt.show()
    exit(1)
go_through_workflow_instead_of_reading_backed_up_file=False                     # true -> goes through workflow to generate signals instead of reading backed up file 
training_df='training_df'
raw_data='BTC-USD2022-05-29_2022-06-05'
m=myMath()
if go_through_workflow_instead_of_reading_backed_up_file:
    print('doing whole workflow ')    
    df=pd.read_csv(f'../data/{raw_data}.csv')
    #df=df.iloc[:500]
    print(mf.how_long(),' read csv')
    df=m.aggregate(df=df,scale=15)
    # use test population as df 
    if 1:
        df=mf.test_population()
        df['close']=df['Yi']
        df['index']=df['Xi']
    # get lookahead based entries and exits
    print(mf.how_long(), ' aggregate') 
    df, p_longs,p_shorts,percentile_longs,percentile_shorts,lowest_longs,highest_shorts = example_workflow(df,percentile_score=90)
    df.to_csv(f'{training_df}.csv')
    print(mf.how_long(),' lookahead workflow')
else:
    print('reading dumped training_df')
    df=pd.read_csv(f'./{training_df}.csv')
m.math_df=df
# calcualte nadaraya watson 
df['nad']=nadaraya_wrapper(X=df['index'], Xi=df['index'],Yi=df['close'],h=1.0,window=50) # both return same values, good 
df['nad2']=m.rolling_nadaraya(df=df,xcol='index',ycol='close',h=1)
# rolling std of candles 
df['candle']=df['close']-df['open']
df['std']=m.calculate_fun(df=df,fun_name='std',col='candle',window=50,inplace=False)
# calculate ema and it's gradient 
df['ema']=m.calculate_fun(df=df,fun_name='ema',col='close',window=50,inplace=False)
df['ema-grad']=m.calculate_fun(df=df,fun_name='grad',col='ema',window=10,inplace=False)
df['ema-grad-smooth']=m.calculate_fun(df=df,fun_name='ema',col='ema-grad',window=10,inplace=False)
# calculate ema_distance 
df['ema-dist']=df['close']-df['ema']
# calculate ema distance integral 
df['ema-dist-area']=m.calculate_fun(df=df,fun_name='cumdiff',col1='close',col2='ema',window=25,inplace=True).replace(np.nan,df.iloc[0]['ema'])

# plot some stuff 
input_cols=['close','lowest_longs','highest_shorts']
opti_cols=['nad','std','ema-grad-smooth','ema-dist','ema-dist-area']
score_col='close'
# df to numpy 
df.dropna(how='any',inplace=True)
df.reset_index(inplace=True)
df['index']=df.index
print(df)
long_entry=df['long_entry']==True 
long_exit=df['long_exit']==True 
x1y1= [

        [ df['index'],df['close']]
        ,[ df[long_entry]['index'],df[long_entry]['close']]
        ,[ df[long_exit]['index'],df[long_exit]['close']]

 
    ]
x2y2= [ 
        [ df['index'],df['close']]
        ,[ df[long_entry]['index'],df[long_entry]['close']]
        ,[ df[long_exit]['index'],df[long_exit]['close']]
        ,[ df['index'],df['nad']]
        ,[ df['index'],df['nad2']]


 

           ]
mf.basic_plot(x1y1=x1y1,x2y2=x2y2,linez=['-xk','^g','vr','--r','--b'])

print(df.to_markdown())
#yet_another_nadaraya(Xi=Xi,Yi=Yi,x=3)


 # 3. get lookahead nadaraya just to have it  
Y=[]
X=df['index']
for x in X:
    yi=yet_another_nadaraya(Xi=df['index'],Yi=df['close'],x=x,h=1.0)
    Y.append(yi) 
