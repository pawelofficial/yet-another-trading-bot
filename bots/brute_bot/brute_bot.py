from binance_api2 import BApi
import pandas as pd 
import datetime 
import logging
logging.basicConfig(level=logging.INFO,filename='bot.log',filemode='w')
import time 
import help_funs as hf 
import random 
import numpy as np 
import itertools

class mybot:
    def __init__(self,binance_config='./configs/binance_api.json') -> None:
        self.b=BApi(api_config=binance_config)          # binance api object 
        self.df=None                                    # dataframe with data 
        self.scale='5min'
        self.interval='1hour'
        self.loop_frequency=30                          # should be lower than end of candle cutoff
        self.dollar_amo=15
        self.time_format='%Y-%m-%d %H:%M:%S'
        self.dt_fun = lambda t1,t2: int((datetime.datetime.strptime(t1,self.time_format)-datetime.datetime.strptime(t2,self.time_format)).total_seconds())

        self.trades_d={                           # dictionary with api response 
                     'symbol': None               # 'ADABUSD', 
                     ,'orderId': None             #  int(datetime.datetime.now().timestamp()),
                     ,'orderListId':None          # -1,
                     ,'clientOrderId':None        # 'N0OWokxZgkcR9jwf1WdIJP',
                     ,'transactTime':None         #  1668283104455,
                     ,'price':None                #  str(price),
                     ,'origQty': None             #  '57.60000000',
                     ,'executedQty': None         #  '57.60000000', 
                     ,'cummulativeQuoteQty': None #  '19.79712000',
                     ,'status': None              # 'FILLED',
                     ,'timeInForce': None         #  'GTC',
                     ,'type': None                # 'MARKET',
                     ,'side': None                # 'SELL', 
                     ,'fills': None               #  [{'price': '0.34370000', 'qty': '57.60000000', 
                     }                            #'commission': '0.01979712', #'commissionAsset': 'BUSD', 'tradeId': 92820107}]}
        self.trades_df=pd.DataFrame(self.trades_d,index=[0])
        # dictionary with assertions 
        # pnl structures for stop losses and take profits 
        self.pnl_sl=0.995        # stop loss 
        self.pnl_tp=1.005        # take profit 
        self.tr_sl=0.995         # trailing stop loss 
        self.pnl_d={'tradeid':None,
                    'price':None,
                    'status':None,
                    'cur_pnl':None,             # cur pnl regardless of order status 
                    'max_pnl':None or -1,       # max pnl when order was open  
                    'min_pnl':None or 999 ,     # min pnl when order was open 
                    'last_pnl':None or 0,       # last pnl when order was open 
                    'symbol': None,             # symbol used to update pnl data 
#                    'sl_price':None,            # to do 
#                    'tp_price':None,            # to do 
#                    'trailing_tp_price':None,   # to do 
                    'comment':None}
        # pnl statuses:
            # FILLED -> filled market sell order 
            # OPEN-LONG -> filled market buy order 
            # CLOSED-LONG -> closed market buy 
        self.pnl_df=pd.DataFrame(self.pnl_d,index=[0])
        self.trading_symbol='ADAUSDT' # let's do only one symbol now 
        # structure for actions 

    # puts nones to a traded d 
    def clear_d(self,d=None):
        if d is None: 
            d=self.trades_d
        d={k:None for k in d.keys()}
        
    # inserts d to a trade_df
    def save_to_df(self,d=None,df=None):
        if d is None: # defaults
            d=self.trades_d
        if df is None: # defaults to trades df 
            df=self.trades_df
        df.loc[len(df)]=d        

    def get_df_from_api(self,scale,interval):
        scale=self.scale 
        interval=self.interval
        klines=self.b.get_recent_candles(symbol=self.trading_symbol,scale=scale,interval=interval)
        self.df=self.b.parsed_kline_list_to_df(parsed_kline_list=klines)
        if 0:
            print('zump it')
            self.df.to_csv('./data/test_df.csv',index=False)

    def market_buy(self,dollar_amo=15,pnl_comment=''):
        response,status=self.b.market_buy_dollar_amo(symbol=self.trading_symbol,dollar_amo=dollar_amo)
        self.trades_d=response
        self.save_to_df()
        self.clear_d()
        # update pnl data  if order got filled 
        print('check order statuses for partial filled here ')
        if status =='FILLED':
            self.pnl_d['symbol']=response['symbol']
            self.pnl_d['tradeid']=response['orderId']
            self.pnl_d['price']=response['price']
            self.pnl_d['status']='OPEN-LONG'
            self.pnl_d['comment']=pnl_comment
            self.save_to_df(d=self.pnl_d,df=self.pnl_df)
            self.clear_d(d=self.pnl_d)
        return response 
    
    # market sells by order id and updates pnl df 
    def market_sell_tradeid(self,orderid,pnl_comment=''):
        response,status = self.b.market_sell_orderid(orderid=orderid,symbol=self.trading_symbol)      
        self.trades_d=response 
        self.save_to_df()
        self.clear_d()
        
        if status =='FILLED':
            self.pnl_d['symbol']=response['symbol']
            self.pnl_d['tradeid']=response['orderId']
            self.pnl_d['price']=response['price']
            self.pnl_d['status']='FILLED'
            self.pnl_d['comment']=pnl_comment
            self.save_to_df(d=self.pnl_d,df=self.pnl_df)
            self.clear_d(d=self.pnl_d)
            
            msk=self.pnl_df['tradeid']==orderid # close long in pnl_df
            index=self.pnl_df[msk].index.values
            self.pnl_df.loc[index,'status']='CLOSED-LONG'
        return response
    
    def update_pnl_df(self):
        response=self.b.check_current_price(symbol=self.trading_symbol)  # check price for a row this shouldn't be done in a loop but whatever.
        key='close' # current price assumption 
        for index,row in self.pnl_df.iterrows():
            if row['tradeid'] is None: # first row is full of nones
                continue
            if row['status'] !='OPEN-LONG':
                continue
            current_price=response[key]
            cur_pnl=current_price/float(row['price']) # one endpoint returns stuff as float another as str meh 

            self.pnl_df.loc[index,'cur_pnl']=cur_pnl # cur pnl regardless of order status 
            if row['status']=='OPEN-LONG':           # when orders are open: 
                self.pnl_df.loc[index,'last_pnl']=cur_pnl
                if cur_pnl>row['max_pnl']:
                    self.pnl_df.loc[index,'max_pnl']=cur_pnl=cur_pnl
                if cur_pnl<row['min_pnl']:
                   self.pnl_df.loc[index,'min_pnl']=cur_pnl=cur_pnl
    
    # iterates over pnl df and closes rows which have pnl < cutoff 
    def execute_sl(self,force=False, pnl_comment =''):
        for index,row in self.pnl_df.iterrows():
            if index==0: # skipping first row 
                continue
            if row['status']!='OPEN-LONG': # skipping pnl rows that are not open-long status 
                continue
            pnl=row['cur_pnl']
            if pnl<self.pnl_sl or force:
                self.market_sell_tradeid(orderid=row['tradeid'],pnl_comment=f'SL  {pnl_comment}')

    
    # iterates over pnl df and closes rows which have pnl > cutoff 
    def execute_tp(self,force=False,pnl_comment=''):
        for index,row in self.pnl_df.iterrows():
            if index==0: # skipping first row 
                continue
            if row['status']!='OPEN-LONG':
                continue
            pnl=row['cur_pnl'] # cur pnl in a row calculated in update_pnl_df 
            if pnl>self.pnl_tp or force:
                self.market_sell_tradeid(orderid=row['tradeid'],pnl_comment=f'TP {pnl_comment}')

    def execute_trtp(self,force=False, pnl_comment=''):
        for index,row in self.pnl_df.iterrows():
            if index==0:
                continue
            if row['status']!='OPEN-LONG':
                continue
            tr_pnl=row['cur_pnl']/row['max_pnl'] 
            if tr_pnl<self.tr_sl or force:
                self.market_sell_tradeid(orderid=row['tradeid'],pnl_comment=f'TRTP {pnl_comment}')
    # dummy strategy on random to check if things work 
    def dummy_strategy(self,symbol = None ):
        if symbol is None: 
            symbol=self.trading_symbol
        buy_frequency = 50 # how often per sleep time make a trade 
        sell_frequency = 50 
        sleep_time = 5 # loop sleep time 
        n=0
        while True:
            n+=1
            print(f'dummy loop number  {n} !  ')
            if random.randint(0,100) / buy_frequency<1: # buy at random  
                print('buying ! ')        
                self.market_buy(dollar_amo=self.dollar_amo,pnl_comment='dummy loop buy')
            print('sleeping ! ')
            time.sleep(sleep_time)
            
            if random.randint(0,100) / sell_frequency<1: # buy at random  
                print('selling !  ! ')        
                self.b.try_to_close_all_by_symbol(symbol=self.trading_symbol)
    
    def load_df(self,from_api=True, from_file=False,scale=None, interval=None, filename='BTC-USD2022-01-10_2022-01-18' ):
        if scale is None: 
            scale=self.scale # 5min 
        if interval is None: 
            interval = self.interval # 1hour 
        if from_api:
            self.df=self.get_df_from_api(scale=scale,interval=interval)

        if from_file:
            self.df=pd.read_csv(filepath_or_buffer=f'./data/{filename}.csv')
            self.df=hf.aggregate(df=self.df,scale=5)
     #   return self.df
    
    
class brutebot(mybot):
    def __init__(self, binance_config='./configs/binance_api.json') -> None:
        super().__init__(binance_config)
        print('this is brute bot ! ')
        self.scale='5min'
        self.interval='1day'
        self.load_df(from_file=True)

        self.lambda_d={
            'rsi': lambda df,col1,span : hf.rsi(df[col1],periods=span)
            ,'gtoe_const': lambda df,col,const: (df[col]>=const).astype(int)
            ,'gtoe_cols': lambda df,col1,col2: (df[col1]>=df[col2]).astype(int)
            ,'ema': lambda df,col,window : df[col].ewm(span=window).mean()
            ,'grad' : lambda df,col,window : np.gradient(df[col],window) # 1d gradient because gradient is important ! 
        }
        self.brute_d={ # dictionary with column names for all the stuf 
            'ema':[]                    # 1st order - on raw 
            ,'rsi':[]                   # 1st order - on raw 
            ,'grad':{'ema':[],'rsi':[]} # 2nd order - on 1st order
            ,'gtoec': []                # 2.5 order - on constant  
            , 'gtoef':{'ema':[]}         # 3rd order - on two functions 
        }
        self.constant=0 # for gtoc
        
        self.output_cats=['rsi','grad','gtoec','gtoef']
        self.output_subcats={'grad':['rsi','ema']}
        self.output_cols=[]

    def gather_outputs(self):
        for key,value in self.brute_d.items():
            if key in  self.output_cats:
                if type(value)==type([]):
                    self.output_cols+=value
                if type(value)==type({}):
                    for kk,vv in value.items():
                        self.output_cols+=vv
                
                
  

    # f(src_col,span) -> value* 
    def function_ema(self,src_col:str ='close',window:int= 5,short_name: bool = False  ):     # adds ema column on top of src_col
        colname=f'ema-{window}-{src_col}'
        if short_name:
            colname=f'ema-{window}'
        self.df[colname]=self.lambda_d['ema'](self.df,src_col,window)
        return colname 
    
    def function_rsi(self,src_col :'str', window : int ,short_name: bool=False):     # adds rsi column 
        colname=f'rsi-{window}-{src_col}'        
        if short_name:
            colname=f'rsi-{window}'
        self.df[colname]=self.lambda_d['rsi'](self.df,src_col,window)
        return colname 

    # f(src_col,span) -> value* 
    def function_grad(self,src_col,window=3 ):     # adds gradient column on top of src_col
        colname=f'grad-{window}-{src_col}'
        self.df[colname]=self.lambda_d['grad'](self.df,src_col,window)
        return colname 

    # f(src_col,value) ->value 


    # f(src_col,value)  -> bool 
    # f(src_col,tgt_col) -> bool  
    def function_gtoe(self,src_col:'str',tgt_col:'str'=None ,constant:int = None): # gtor than tgt col or constant
        if constant is not None:
            colname=f'gtoec-{constant}-{src_col}'
            self.df[colname]=self.lambda_d['gtoe_const'](self.df,src_col,constant)
        if constant is None:
            colname=f'gtoef-{src_col}-{tgt_col}'
            self.df[colname]=self.lambda_d['gtoe_cols'](self.df,src_col,tgt_col)
            
        return colname

    def mixer(self):
        # 1.  first_order
        first_order_spans=[5,10,15,20,25,50]
        first_order_d={ 
            'ema': self.function_ema        # ema on close 
            ,'rsi': self.function_rsi       # rsi on close 
            }
        src_col='close'
        for span in first_order_spans:
            for key,fun in first_order_d.items():
                colname=fun(src_col=src_col,window=span,short_name=True) 
                self.brute_d[key].append(colname)
                
        #2. 2nd order - gradient on ema and rsi  
        grad_spans=[5]
        grad_cols=list(self.brute_d['ema']) + list(self.brute_d['rsi'])
        for span in grad_spans:
            for col in grad_cols :                                 # for ema and rsi columns 
                col_span=col.split('-',2)[1]                       # ema/rsi column span         
                if span<=int(col_span):                            # dont calculategradient of bigger span on lower span emas
                    colname=self.function_grad(src_col=col,window=span)
                    foo=col.split('-',2)[0]     # ema / rsi  
                    self.brute_d['grad'][foo].append(colname)
        
        #3. gtoc - rsi and ema grads 
        grad_cols=self.brute_d['grad']['ema'] + self.brute_d['grad']['rsi']
        for col in grad_cols:
            #colname=f'gtoec-{col}-{self.const}'
            colname=self.function_gtoe(src_col=col,constant=self.constant)
            foo=col.split('-',2)[0]     # ema / rsi  
            self.brute_d['gtoec'].append(colname)
            
        #3.1 gtoc - rsi absolutes 
        cols=self.brute_d['rsi']
        vals=[10,30,50,90]
        for col in cols:
            for val in vals:
                colname=self.function_gtoe(src_col=col,constant=val)
                self.brute_d['gtoec'].append(colname)
            
        #4. gtoef - emas greater than each other 
        cols=self.brute_d['ema'] # use emas for gtoefs 
        pairs=list(set(list(itertools.product(cols,cols)))) # get product of all emas 
        unique_pairs=hf.unique_pairs(pairs) # unique emas - skippint (ema5,ema10) vs (ema10,ema5)

        for pair in unique_pairs:
            if pair[0]==pair[1]: # skip pairs with the same function 
                continue
            colname=self.function_gtoe(src_col=pair[0],tgt_col=pair[1])
            self.brute_d['gtoef']['ema'].append(colname)
            
            
        


                    


    
    # output
        #  5 rsis 
        #  5 gtor gradients  
        #  N gtor funs  
        
        

    
    
    
    
if __name__=='__main__':
#    m=mybot()
    bb=brutebot()
#    bb.function_ema() 
#    bb.function_grad(src_col='ema-close-5')
#    bb.function_gtoe(src_col='grad-3-ema-close-5',constant=0)
#    bb.function_gtoe(src_col='close',tgt_col='open')
#    bb.function_rsi(src_col='close',span=5)
    bb.mixer()
#    for k,v in bb.brute_d.items():
#        print('\n',k,v)
    
    bb.gather_outputs()
    print(bb.output_cols)
#    print(bb.df)