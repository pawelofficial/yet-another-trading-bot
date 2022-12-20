import logging 
import pandas as pd 
import datetime
from typing import Callable
import numpy as np 
import matplotlib.pyplot as plt 
# logs variable 
def log_variable(var,msg='',wait=False):
    if var is None:
        var='None'
    s= f' {msg} \n{var}'
    logging.info(s)
    if wait:
        print(s)
        input('waiting in log ')
        
        
def plot_candlestick2(candles_df : pd.DataFrame
                      ,x1y1 : list =[]
                      , x2y2 : list = []
                      ,longs_ser : pd.Series = pd.Series({},dtype=np.float64)
                      ,shorts_ser : pd.Series = pd.Series({},dtype=np.float64)
                      ):
    df=candles_df
    plt.rcParams['axes.facecolor'] = 'y'
    low=df['low']
    high=df['high']
    open=df['open']
    close=df['close']
    # mask for candles 
    green_mask=df['close']>=df['open']
    red_mask=df['open']>df['close']
    up=df[green_mask]
    down=df[red_mask]
    # colors
    col1='green'
    black='black'
    col2='red'

    width = .4
    width2 = .05

    fig,ax=plt.subplots(2,1)
    ax[0].bar(up.index,up['high']-up['close'],width2,bottom=up['close'],color=col1,edgecolor=black)
    ax[0].bar(up.index,up['low']-up['open'],width2, bottom=up['open'],color=col1,edgecolor=black)
    ax[0].bar(up.index,up['close']-up['open'],width, bottom=up['open'],color=col1,edgecolor=black)
    ax[0].bar(down.index,down['high']- down['close'],width2,bottom=down['close'],color=col2,edgecolor=black)
    ax[0].bar(down.index,down['low']-  down['open'],width2,bottom=down['open'],color=col2,edgecolor=black)
    ax[0].bar(down.index,down['close']-down['open'],width,bottom=down['open'],color=col2,edgecolor=black)

    for xy in x1y1:
        ax[0].plot(xy[0],xy[1])
        
        
    for xy in x2y2:
        ax[1].plot(xy[0],xy[1])

    if not longs_ser.empty:
        msk=longs_ser==True
        ax[0].plot(longs_ser[msk].index, df[msk]['low']*longs_ser[msk].astype(int),'^g')

    if not shorts_ser.empty:
        msk=shorts_ser==True
        ax[0].plot(shorts_ser[msk].index, df[msk]['high']*shorts_ser[msk].astype(int),'vr')


    
def aggregate(df :pd.DataFrame = pd.DataFrame({}), 
              scale : int = 5, 
              src_col : str = 'timestamp',
              cols : list = ['open','close','low','high','volume'],
              timestamp_name='timestamp'):
    tformat='%Y-%m-%dT%H:%M:%S.%fZ'
#    for col in cols:
#        df[col].apply(pd.to_numeric)
    
    floor_dt =  lambda df,str_col, scale: df[str_col].apply(lambda x: datetime.datetime.strptime(x,tformat) -
                    datetime.timedelta(
                    minutes=( datetime.datetime.strptime(x,tformat).minute) % scale,
                    seconds= datetime.datetime.strptime(x,tformat).second,
                    microseconds= datetime.datetime.strptime(x,tformat).microsecond))
    
    
    agg_funs_d={  'open':lambda ser: ser.iloc[0],
                 'close':lambda ser: ser.iloc[-1],
                 'high':lambda ser: ser.max(),
                 'low': lambda ser: ser.min(),
                 'volume': lambda ser: ser.sum()
                }  
    
    if src_col not in df.columns:
        print('src col not in df columns')
        raise 
    dt_col = '-'.join(['ts',str(scale) ])
    agg_df=pd.DataFrame({})
    
    df[dt_col]=floor_dt(df,'timestamp',scale)
#    df[dt_col]=calculate_fun(df=df,fun_name='floor_dt',str_col='timestamp',scale=scale) # need to write floor column to df  to later groupby it 
    agg_df[dt_col]=df[dt_col].unique().copy()
    for col in cols:
        g=df[[col,dt_col]].groupby([dt_col])
        ser=g.apply(agg_funs_d[col])[col].reset_index(name=col)
        agg_df=agg_df.merge(ser,left_on=dt_col,right_on=dt_col)


    agg_df.rename(columns={dt_col:timestamp_name},inplace=True)
    return agg_df


if __name__=='__main__':
    df=pd.read_csv('./data/BTC-USD2022-01-10_2022-01-18.csv')
    
    for index,row in df[:100].iterrows():
        print(index)