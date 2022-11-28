
import pandas as pd 
import help_funs as hf
import ta.momentum as tam 
# this scripty script puts together cool indicators, for now the goal is to have 
# 3 momentum indicators
# 3 volume indicators 
# 3 volatility indicators 
# 3 other indicators 

# once achieved all of those will be normalized on their history
#   0-1 method 
#   z-score 

# once this is achieved this df will be put to pytorch and vectorbot to find good strategy for specific day/week

# once this is achieved strategies will be connected with each other with pytorch again to have a strategy of strategies

class indicators:
    def __init__(self,df) -> None:
        self.df=df
    # dummy 
    def fun_dummy(self, window : 0, src_col : 0, inplace = True ):
        colname = f'dummy-{src_col}-{window}'
        f= lambda df,col,window : df[col] - window 
        if inplace:
            self.df[colname]=f(df=self.df,col=src_col,window=window)
            return 
        return f(df=self.df,col=src_col,window=window)
        
    #  ema 
    def fun_ema(self,src_col='close', window : int = 25 , inplace = True ):
        colname=f'ema-{src_col}-{window}'
        f = lambda df,col,window : df[col].ewm(span=window).mean()
        if inplace:
            self.df[colname]=f(df=self.df,col=src_col,window=window)
            return 
        return f(df=self.df,col=src_col,window=window)

# momentum indicators 
    #  rsi 
    def fun_rsi(self,src_col = 'close', window: int = 25 , inplace = True ):
        colname = f'rsi-{src_col}-{window}'
        f= lambda df,col,window : tam.rsi(close=df[col],window=window)
        if inplace:
            df[colname]=f(df=self.df,col=src_col,window=window)
            return 
        return f(df=self.df,col=src_col,window=window)
    
    # kama ema 
    def fun_kama(self, src_col='close', window :int =  10, pow1 : int = 2, pow2 : int = 30,  inplace = True ):
        colname = f'kama-{src_col}-{window}-{pow1}-{pow2}'
        f= lambda df,col,window,pow1,pow2: tam.kama(close=df[col], window=window,pow1=pow1,pow2=pow2) 
        if inplace:
            self.df[colname]=f(df=self.df,col=src_col,window=window,pow1=pow1,pow2=pow2)
            return 
        return f(df=self.df,col=src_col,window=window,pow1=pow1,pow2=pow2)

    # awesome indicator 
    def fun_ao(self,window1:int =5 , window2=34, inplace = True ):
        colname=f'ao-{window1}-{window2}'
        f= lambda df,window1,window2 : tam.awesome_oscillator(high=df['high'],
                                                              low=df['low'],
                                                              window1=window1,
                                                              window2=window2  )
        if inplace:
            df[colname]=f(df=self.df,window1=window1,window2=window2)
            return 
        return f(df=self.df,window1=window1,window2=window2)
# volume indicators 
        
        
        
        
        
        
if __name__=='__main__':
    filename='BTC-USD2022-01-10_2022-01-18'
    df=pd.read_csv(f'./data/{filename}.csv')
    i=indicators(df=df)
    i.fun_ema()
    i.fun_rsi()
    i.fun_ao()
    i.fun_kama()

#    print(i.fun_kama(inplace=False))
    #print(i.df)