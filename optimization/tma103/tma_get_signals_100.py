# this is versio of second script but better 
# this code outputs dataframes to csv files with features - z scores of price action and labels - long/short signals
from fileinput import filename
from torch.utils.data import Dataset, DataLoader 
import pandas as pd 
import torch 
import matplotlib.pyplot as plt 
import sys 
sys.path.append("../..")
sys.path.append("..")
from myfuns import myfuns as mf
from myfuns import lookahead_entries as la
import torchvision 
import torch.nn as nn 
from sklearn.model_selection import train_test_split 
import numpy as np 




class prep_data(): # transformation 
    def __init__(self,n=-1, path = None, filename = None ) -> None:
        self.scale=5                                                                                                                            # aggregation timescale 
        self.df=self.read_csv(path=path,filename=filename,scale=self.scale)                                                                     # df to aggregate 
        self.df['index']=self.df.index                                                                                                          # add index column 
        if n!=-1:                                                                                                                               # n!=-1 -> read first n rows  
            self.df=self.df.iloc[:n].copy(deep=True)

        # dictionary with functions to get labels for pytorch 
        self.fun_d={}
        self.fun_d['sma']=lambda df,n,col: df[col].rolling(window=n).mean() # rolling sma 
        self.fun_d['max']=lambda df,n,col: df[col].rolling(window=n).max() # rolling sma 
        self.fun_d['std']=lambda df,n,col: df[col].rolling(window=n).std() # rolling std 
        self.fun_d['nrzscore']=lambda df,n,col: self.zscore(n,col)                # not rolling  
        self.fun_d['zscore']=lambda df,n,col : df[col].rolling(window=n).apply(self.rolling_zscore) # rolling mean based normalization
        self.fun_d['norm']=lambda df,n,col : df[col].rolling(window=n).apply(self.rolling_normalize) # rolling mean based normalization
        
    # read csv from file and aggregate 
    def read_csv(self,path : str = None,filename : str = None ,scale:int = 5 ):
        if not path:
            path='../../data/'
        if not filename:
            filename='BTC-USD2022-01-10_2022-01-18'
            #filename='BTC-USD2021-06-05_2022-06-05'
            print(f'reading default {filename}')
        df=pd.read_csv(f'{path}/{filename}.csv', lineterminator=';')
        return mf.aggregate(df=df, scale=scale, src_col='timestamp')
    
    # zscore function 
    def rolling_zscore(self,ser):
        mu=ser.mean()
        std=ser.std()
        return (ser.iloc[-1] - mu ) / std 
    # not rolling z score 
    def zscore(self,window,col): # rolling zscore 
        r = self.df[col].rolling(window=window)
        m = r.mean().shift(1)
        s = r.std(ddof=0).shift(1)
        z = (self.df[col]-m)/s
        return z
    # rolling normalization function 
    def rolling_normalize(self,ser): # rolling normalize based on mean 
        return ser.iloc[-1]/ser.mean()

    # workflow computing things - returns dataframe with variables  
    def workflow2(self) -> pd.DataFrame: 
        self.df['candle']=self.df['close']-self.df['open']
        timeframes=mf.get_timeframes(dist_fun='exp', timeframes_ranges=[2,200],N=5) # get few timeframes for labels wigh dist_fun distribution ( 2,4,8,16,32 or 2,4,16,169 or sth )
        z_keys=[]
        for tf in timeframes: # compute variables on various timeframes 
            self.df[f'sma-{tf}']=self.fun_d['sma'](self.df,tf,'close')
            self.df[f'std-{tf}']=self.fun_d['std'](self.df,tf,'close') 
            self.df[f'z-sma-{tf}']=self.fun_d['zscore'](self.df,tf,f'sma-{tf}')
            self.df[f'z-std-{tf}']=self.fun_d['zscore'](self.df,tf,f'std-{tf}')
            z_keys.append(f'z-sma-{tf}')
            z_keys.append(f'z-std-{tf}')
        return self.df[z_keys].copy(deep=True) # return df with variables 

    
    def get_labels(self,zscore_df): # adds price action and signals columns to zscore_df 
        lookahead_df, p_longs,p_shorts,percentile_longs,percentile_shorts,lowest_longs,highest_shorts,lh_columns = la.example_workflow(df=self.df,percentile_score=90,best_of_breed=10)
        signal_cols=['LONGS_SIGNAL','SHORTS_SIGNAL','index','open','close','low','high'] # close columns is in df
        zscore_df=zscore_df.merge(lookahead_df[signal_cols].astype(float),left_index=True,right_index=True)
        return zscore_df , p_longs,p_shorts,percentile_longs,percentile_shorts,lowest_longs,highest_shorts,signal_cols

    
    

    

def feature_engineering(filename=None,csv_prefix=''): # dumps csv for pytorch 
    # prep data 
    
    t=prep_data(n=-1,filename=filename)
    zscore_df=t.workflow2().dropna(how='any')
    df, p_longs,p_shorts,percentile_longs,percentile_shorts,lowest_longs,highest_shorts,signal_cols =t.get_labels(zscore_df==zscore_df)

   # df=df.dropna(how='any')
    features_df=df[ [c for c in df.columns if c not in signal_cols] ]                   # df with features for pytorch 
    labels_df=df[['LONGS_SIGNAL','SHORTS_SIGNAL']]                                    # df for labels for pytorch 
    plot_df=df[signal_cols].copy(deep=True)
    plot_df['LONGS_SIGNAL_PA']=plot_df.apply(lambda x: x['LONGS_SIGNAL']*x['low'],axis=1 )
    plot_df['SHORTS_SIGNAL_PA']=plot_df.apply(lambda x: x['SHORTS_SIGNAL']*x['high'],axis=1 )
    
    mf.dump_csv(df=features_df,filename=f'{csv_prefix}features_df.csv')
    mf.dump_csv(df=labels_df,filename=f'{csv_prefix}labels_df.csv')
    mf.dump_csv(df=plot_df,filename=f'{csv_prefix}plot_df.csv')
    mf.dump_csv(df=df,filename=f'{csv_prefix}features_labels.csv' )
    plot_flg=True
    if plot_flg:
        mf.plot_candlestick(df=plot_df, shorts_ser=plot_df['SHORTS_SIGNAL_PA'],longs_ser=plot_df['LONGS_SIGNAL_PA'])
        plt.show()
        

    return features_df,labels_df
    
    
class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear=nn.Linear(n_input_features,1)
        
    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x) )
        return y_predicted
    
if __name__=='__main__':
    # make features and labels from scratch or read csv if you did it altready 
    dump_csvs=True
    filename='BTC-USD2022-01-10_2022-01-18'
    if dump_csvs:
        print('calculating signals and dumpting output data')
        features_df,labels_df = feature_engineering(csv_prefix=filename+'_',filename=filename)
    else:
        print('reading signals from csv files ')
        features_df=pd.read_csv(filename+'_features_df.csv')
        labels_df=pd.read_csv(filename+'_labels_df.csv')
        plot_df=pd.read_csv(filename+'_plot_df.csv')
        
    if features_df.isnull().values.any():
        raise #  'nans in feature df '
    if labels_df.isnull().values.any():
        raise # 'nans in label df '
    
    exit(1) # stuff from below moved to dma_torch.y
    X=torch.from_numpy(features_df.to_numpy().astype(np.float32))
    Y=torch.from_numpy(labels_df['LONGS_SIGNAL'].to_numpy().astype(np.float32))
    Y=Y.view(len(Y),1)
    print(X.shape)
    print(Y.shape)
    
    X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=1234)
    
    
    n_samples, n_features = X_train.shape 
    model=LogisticRegression(n_features)
    criterion = nn.BCELoss() # binary cross entropy 
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    
    
for epoch in range(1000):
    # prediction 
    y=model(X_train) # calling the module class invokes forward function fren ! loss calculation
    l=criterion(y,Y_train) # order here is important ! 
    # graduebt 
    l.backward()    # using backward method to calcylate dl/dw gradient 
    # update weights 
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 100 ==0:
        print(epoch)
    
    
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    print('acc:', acc)
    
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)
        
    
    
    y_out=model(X).detach()
    y_out=pd.Series(y_out[:,0]/max(y_out[:,0]) )
    y_out=y_out*plot_df['low']
    plt.plot(plot_df['high']/plot_df['high'].mean(),'o' )
    plt.plot(plot_df['LONGS_SIGNAL'],'or' )
    plt.plot(y_out/y_out.mean(),'--x')
    plt.show()