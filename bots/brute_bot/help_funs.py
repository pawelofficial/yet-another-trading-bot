import logging 
import pandas as pd 
import datetime
from typing import Callable
import numpy as np 
# logs variable 
def log_variable(var,msg='',wait=False):
    if var is None:
        var='None'
    s= f' {msg} \n{var}'
    logging.info(s)
    if wait:
        print(s)
        input('waiting in log ')
        
        
    
def aggregate(df :pd.DataFrame = pd.DataFrame({}), 
              scale : int = 5, 
              src_col : str = 'timestamp',
              cols : list = ['open','close','low','high']):
    tformat='%Y-%m-%dT%H:%M:%S.%fZ'
    floor_dt =  lambda df,str_col, scale: df[str_col].apply(lambda x: datetime.datetime.strptime(x,tformat) -
                    datetime.timedelta(
                    minutes=( datetime.datetime.strptime(x,tformat).minute) % scale,
                    seconds= datetime.datetime.strptime(x,tformat).second,
                    microseconds= datetime.datetime.strptime(x,tformat).microsecond))
    
    
    agg_funs_d={  'open':lambda ser: ser.iloc[0],
                 'close':lambda ser: ser.iloc[-1],
                 'high':lambda ser: ser.max(),
                 'low': lambda ser: ser.min(),
                 'volume': lambda ser: ser.mean()
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
    #agg_df['index']=agg_df.index
    return agg_df

def rsi(ser, periods = 14, ema = True):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = ser.diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi


pairs=[('ema-25', 'ema-15'), ('ema-50', 'ema-25'), ('ema-20', 'ema-25'), ('ema-10', 'ema-25'), ('ema-15', 'ema-25'), ('ema-5', 'ema-25'), ('ema-25', 'ema-25'), ('ema-50', 'ema-20'), ('ema-50', 'ema-10'), ('ema-20', 'ema-20'), ('ema-20', 'ema-10'), ('ema-50', 'ema-50'), ('ema-20', 'ema-50'), ('ema-15', 'ema-20'), ('ema-10', 'ema-20'), ('ema-10', 'ema-10'), ('ema-15', 'ema-10'), ('ema-50', 'ema-5'), ('ema-5', 'ema-20'), ('ema-5', 'ema-10'), ('ema-10', 'ema-50'), ('ema-5', 'ema-50'), ('ema-15', 'ema-50'), ('ema-20', 'ema-5'), ('ema-25', 'ema-10'), ('ema-25', 'ema-20'), ('ema-15', 'ema-5'), ('ema-10', 'ema-5'), ('ema-50', 'ema-15'), ('ema-5', 'ema-5'), ('ema-25', 'ema-50'), ('ema-20', 'ema-15'), ('ema-25', 'ema-5'), ('ema-15', 'ema-15'), ('ema-10', 'ema-15'), ('ema-5', 'ema-15')]



def unique_pairs(pairs):
    # input: [(1,2),(2,1)]
    # output: -> [[1,2]]
    unique_pairs=[]
    f= lambda s: int(s.split('-')[-1])
    ordered_pairs=[ sorted(list(s),key=f) for s in pairs ]
    for pair in ordered_pairs:
        if pair in unique_pairs:
            continue
        elif pair[0]!=pair[1]:
            unique_pairs.append(pair)
            
    return unique_pairs

