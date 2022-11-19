from cProfile import label
from heapq import merge
import sys 
sys.path.append("../..")
sys.path.append("..")
import pandas as pd 
import matplotlib.pyplot as plt
from myfuns import myfuns as mf 
import numpy as np 

from data.coinbase_api import coinbase_api, ApiUtils, CoinbaseExchangeAuth




# 1.  read_raw_data_from_api
def  read_raw_data_from_api(start_date : str = '10-01-2022', end_date : str ='18-01-2022', path : str = './data/' ):
    """ reads data from api """
    print('reading raw data from api ')
    utils=ApiUtils(CoinbaseExchangeAuth,coinbase_api)
    api=utils.init(config_filepath='../../credentials/api_config.json')
    filename=utils.download_by_dates(api=api,start_date=start_date ,end_date=end_date,path='./data/')
    return pd.read_csv(path+filename)

def read_df_from_file(relpath='./data/',filename='BTC-USD2022-01-10_2022-01-18',lineterminator=None) -> pd.DataFrame:
    return pd.read_csv(relpath+filename+'.csv',lineterminator=lineterminator)

def dump_df(df :pd.DataFrame, filename='raw_df',relpath='./data/') -> None:
    df.to_csv(relpath+filename+'.csv',index=False)
    
def aggregate_df(df : pd.DataFrame, scale : int = 5, src_col : str = 'timestamp') -> pd.DataFrame:
    return mf.aggregate(df=df, scale=scale, src_col='timestamp')

def make_labels(df) -> pd.DataFrame:
    df=df.copy(deep=True)
    df['index']=df.index
    original_columns=list(df.columns) # gotta write down original  columns 
    mf.example_workflow(df=df)       # this mutates provided df :( 
    labels_df, p_longs,p_shorts,percentile_longs,percentile_shorts,lowest_longs,highest_shorts=mf.example_workflow(df=df,percentile_score=80)
    labels_cols=['LONG_ENTRY','LONG_EXIT'] # columns of interests 
    #merge_df=pd.merge(df,labels_df)        # joining, maybe unnecessary 
    merge_df=labels_df [original_columns+labels_cols]
    # checking if no of signals make sense
    if 0: 
        msk1=merge_df['LONG_ENTRY']==True 
        msk2=merge_df['LONG_EXIT']==True
        print(len(merge_df[msk1]), len(merge_df[msk1])/len(merge_df))
        print(len(merge_df[msk2]), len(merge_df[msk2])/len(merge_df))
    print(original_columns+labels_cols)
    return merge_df
     

def make_features(df): 
    df=df.copy(deep=True)
    df['index']=df.index
    def rolling_zscore(ser):
        mu=ser.mean()
        std=ser.std()
        return (ser.iloc[-1] - mu ) / std 
    def rolling_normalize(ser): # rolling normalize based on mean 
        return ser.iloc[-1]/ser.mean()
    fun_d={}
    fun_d['sma']=lambda df,n,col: df[col].rolling(window=n).mean() # rolling sma 
    fun_d['max']=lambda df,n,col: df[col].rolling(window=n).max() # rolling sma 
    fun_d['std']=lambda df,n,col: df[col].rolling(window=n).std() # rolling std  
    fun_d['zscore']=lambda df,n,col : df[col].rolling(window=n).apply(rolling_zscore) # rolling mean based normalization
    fun_d['norm']=lambda df,n,col : df[col].rolling(window=n).apply(rolling_normalize) # rolling mean based normalization
    
    
    fun_d['distance']=lambda df,col1,col2 : df[col1]-df[col2]
    fun_d['cumdiff']=lambda df,n,col1,col2: df['index'].rolling(window=n).apply(lambda x: np.sum (df.iloc[x][col1]-df.iloc[x][col2]))
    
    
    timeframes=mf.get_timeframes(dist_fun='exp',timeframes_ranges=[5,150],N=6) # timeframes 
    print(df.columns)
    # get features - various calcs on various timeframes 
    for tf in timeframes:
        print(tf)
        df[f'sma-{tf}']=fun_d['sma'](df,tf,'close')
        df[f'std-{tf}']=fun_d['std'](df,tf,'close') 
        df[f'rsi-{tf}']=mf.calc_rsi(over = df['close'],fn_roll=lambda s: s.ewm(span=tf).mean())

#    df['oindex']=df.index
    df.dropna(how='any',inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={'level_0':'oindex'},inplace=True)
 
    return df 
        
        

# workflow d for controlling the workflow 
workflow_d={
    'read_df_from_file':(read_df_from_file,True)      # done - 1
    ,'read_raw_data_from_api':(read_raw_data_from_api,False) # done - 0 
    ,'dump_df':(dump_df,False)          # done    
    ,'aggregate_df':(read_raw_data_from_api,False)           # done 
}
    
    
    
if __name__=='__main__':
    raw_df = read_df_from_file(filename='BTC-USD2022-01-10_2022-01-18',lineterminator=';')
    agg_df=aggregate_df(df=raw_df)    
    feature_df=make_features(df=agg_df)
    labels_df=make_labels(df=feature_df)
    dump_df(df=labels_df,filename='labels_df_long')
    
    exit(1)
    df=labels_df

    
    l1=['close']
    l2=['sma-7','sma-13','sma-24','sma-81','sma-148']
    l3=['rsi-7','rsi-13','rsi-24','rsi-81','rsi-148']
    x1y1=[[df.index,df[i]] for i in l2 ]
    x2y2=[[df.index,df[i]] for i in l3 ]
    
    longs_ser=df['LONG_ENTRY']
    shorts_ser=df['LONG_EXIT']
    mf.plot_candlestick2(candles_df=df,x1y1=x1y1,x2y2=x2y2,longs_ser=longs_ser,shorts_ser=shorts_ser)
    plt.show()

    
    exit(1)
    
    agg_df=read_df_from_file(filename='agg_df_small')
    feature_df=make_features(agg_df)
    exit(1)
    
if __name__=='__main__':
    df=pd.read_csv('./data/agg_df.csv')
    #sin=[2+np.sin(x/10) for x in range(len(df))]
    #df['close']=sin

    fun_d={}
    fun_d['distance']=lambda df,col1,col2 : df[col1]-df[col2]
    fun_d['cumdiff']=lambda df,n,col1,col2: df['index'].rolling(window=n).apply(lambda x: np.sum (df.iloc[x][col1]-df.iloc[x][col2]))
    
    
    
    fun_d['sma']=lambda df,n,col : df[col].rolling(window=n).mean()
    
    
    df['sma5']=fun_d['sma'](df,5,'close')
    df['sma100']=fun_d['sma'](df,100,'close')
    df['dist']=fun_d['distance'](df,'sma5','sma100')
    df['index']=df.index
    df['cumdiff']=fun_d['cumdiff'](df,100,'sma100','close')
    
    
    
    fig,ax=plt.subplots(3,1)
    ax[0].plot(df.index,df['close'],'--r')
    ax[0].plot(df.index,df['sma100'],'--b')
    ax[1].plot(df.index,df['dist'],'-r')
    ax[1].plot(df.index,df['close']-df['sma100'],'-b')
    ax[2].plot(df.index,df['cumdiff'])
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    plt.show()