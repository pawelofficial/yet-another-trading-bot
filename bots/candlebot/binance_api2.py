import requests,json,time
from binance import Client
from binance import *
import numpy as np 
import datetime
import logging 
logging.basicConfig(level=logging.INFO,filename='binance_log.log',filemode='w')
import datetime
import pandas as pd 
import random 
from playground import return_on_failure,retry_on_index_failure



class BApi:
# config methods 
    def __init__(self,api_config='binance_api.json'):
        # symbols are ADAUSDT
        # symbols/assets are ADA 
        print("i am binance api brrrrrrr \n\n")
        self.api_config_json=api_config # json with binance api keys
        self.read_config()
        self.client= Client(self.api_key, self.api_secret)
        self.close_residue=1 # dollar close residue 
        self.order=None
        self.time_format='%Y-%m-%d %H:%M:%S'
        self.system_status=self.check_status()
        self.dust=1 # cant sell 100% :( 
        
        self.kline_d={}
        self.kline_d['1min']=[Client.KLINE_INTERVAL_1MINUTE, "1 minutes ago UTC"]
        self.round=5 # rounding for calculation 
        self.real_bags_symbols=['USDT','BUSD','ADA','BTC','LUNC'] # only those symbols will be shown if get_my_bags is in default
        

        self.kline_d={
            '1min':Client.KLINE_INTERVAL_1MINUTE
            ,'5min':Client.KLINE_INTERVAL_5MINUTE
            ,'1hour':Client.KLINE_INTERVAL_1HOUR
            ,'15min':Client.KLINE_INTERVAL_15MINUTE
            ,'1day':Client.KLINE_INTERVAL_1DAY
        }
        self.scale_d={
            '1min':'1 minutes ago UTC'
            ,'3min':'3 minutes ago UTC'
            ,'5min':'5 minutes ago UTC'
            ,'15min':'15 minutes ago UTC'
            ,'1hour':'1 hour ago UTC'
            ,'1day':'1 day ago UTC'
            , '1week': '1 week ago UTC'
            , '1month':'1 month ago UTC'
        }
        self.order_side_d={'buy':Client.SIDE_BUY,'sell':Client.SIDE_SELL}                        # sides of orders - buy or sell 
        self.order_type_d={'market':Client.ORDER_TYPE_MARKET,'limit':Client.ORDER_TYPE_LIMIT}    # types of orders - market or order 
        self.dummy_price=100 # dummy price to have some control over test orders for testing ! 
        
        
    # reads config json with api keys 
    def read_config(self):  
        api_config_d=json.load(open(self.api_config_json))
        self.api_key=api_config_d['api_key']
        self.api_secret=api_config_d['api_secret']
        
    # checks status of exchange 
    def check_status(self): 
        system_status = self.client.get_system_status()['status']
        self.log_variable(var=system_status,msg='system status')
        return system_status 

    # logs a variable, if wait then waits and prints stuff 
    def log_variable(self,var,msg='',**kwargs):
        if var is None:
            var=''
        s= f' {msg} : {var}'
        for k,v in kwargs.items():
            s  += f' {k} : {v} '
        logging.info(s)


    # gets all availeble symbols on exchange 
    def get_all_symbols(self):  
        data = self.client.get_all_tickers() # yay tickers!
        symbols = sorted([ d['symbol'] for d in data ]) # here it's symbol yeah! 
        self.log_variable(var=symbols, msg='all symbol')
        return symbols

    # returns list of dictionaries with non zero positions 
    def get_my_bags(self,return_only_real_bags=True,specific_asset = None,ignore_assets=['BUSD','USDT'] ) -> list:
        account_info=self.client.get_account() # dictionary with holdings and other info 
        balances=account_info['balances']      # list of positions 
        bags=[]                                # list of non zero positions 
        for d in balances:                     # filters out positions that are equal to zero 
            if float(d['free'])>0:
                bags.append(d)
        # if return_only_real_bags is true then return only those bags that are in self.real_bags variable 
        real_bags=[]
        if return_only_real_bags and specific_asset is None:
            for b in bags:
                if b['asset'] in self.real_bags_symbols : # getting amount for specific asset shouldnt be done for base assets - stabelecoins 
                    real_bags.append(b)
            return real_bags
        # return amount for one specific asset - ADA or ADAUSDT
 
        if specific_asset is not None :
            for d in bags:
                if d['asset'] in specific_asset and d['asset'] not in ignore_assets: # ADA in ADAUSDT:
                    return d['free']
        
        return bags                            # list of positions with non zero amount 
    
    # checks current price of a symbol - returns parsed kline ! 
    @retry_on_index_failure({}) # sometimes you get a bad response from api  and d[0] fails 
    def check_current_price(self,symbol) -> dict:
        kline=self.client.get_historical_klines(symbol, self.kline_d['1min'],self.scale_d['3min'])
        d=self.parse_kline(kline_list=kline)
        print(kline)
        print(d)
        input('wait')
        return d[0] # returns: {'open_epoch': 1668177840000.0, 'open': 16553.21, 'close': 16573.0, 'high': 16553.12, 'low': 16557.4, 'volume': 118.09522, 'open_utc': '2022-11-11 15:44:00'}
    
    # returns recent candles of given scale over given interval 
    def get_recent_candles(self,symbol,scale='5min',interval='1hour') -> list:
        # due to how binance api works number of candles here is not deterministic, don't rely on it 
        # download more candles and truncate them if you want last n candles !
        kline_list=self.client.get_historical_klines(symbol,self.kline_d[scale],self.scale_d[interval])
        parsed_kline_list=self.parse_kline(kline_list=kline_list)
        return parsed_kline_list[1:] # first candle most likely wont fit your scale hence truncating it  
    
    # parses kline returned by binance which is in shitty format ot something human readable!
    def parse_kline(self,kline_list : list ) -> dict :
        parse_d={'open_epoch':'','open':'','high':'','low':'','close':'','volume':'','close_epoch':''}
        # parsing list of lists to a list of human readable dictionaries
        parsed_kline_list=[]
        for kline in kline_list:
            d = dict(zip([k for k,v in parse_d.items()],[float(i) for i in kline[:7] ] ))
            t=str(d['open_epoch'])
            d['open_utc']=datetime.datetime.fromtimestamp(int(t[:10])).strftime(self.time_format)
            t=str(d['close_epoch'])
            d['close_utc']=datetime.datetime.fromtimestamp(int(t[:10])).strftime(self.time_format)
            parsed_kline_list.append(d)
        return parsed_kline_list
    
    # parses list of dicts to a dataframe 
    def parsed_kline_list_to_df(self,parsed_kline_list):
        df=pd.DataFrame(parsed_kline_list)
        return df 
        
    def calculate_from_fills(self,fills=[],key=None):       
        # binance gives you a shitty response that you have to parse to actually check out which price you got omg CZ ima switch to FTX !! 
#        fills=[{'price': '0.33380000', 'qty': '44.90000000', 'commission': '0.04490000', 'commissionAsset': 'ADA', 'tradeId': 92890779},
#                 {'price': '0.33380000', 'qty': '44.90000000', 'commission': '0.04490000', 'commissionAsset': 'ADA', 'tradeId': 92890779}
#                 ]
        if key is None:
            key='price'
        avg_price=sum([ float(d[key])*float(d['qty']) for d in fills   ])/ sum ( [ float(d['qty']) for d in fills ]  )
        
        return avg_price 
        
    # returns amount you can buy of a symbol for provided dollar_amount 
    def calculate_quantity(self,symbol,dollar_amo,asset_amo=None):
        symbol_info=self.client.get_symbol_info(symbol)
        minQty=float(symbol_info['filters'][2]['minQty']) # minimum qty to buy 
        stepSize=float(symbol_info['filters'][2]['stepSize']) # step size
        precision=int(symbol_info['baseAssetPrecision'])
        # rounding asset rather than dollar amount 
        if asset_amo is not None :
            lotSize=float(symbol_info['filters'][2]['stepSize']) 
            real_amo=float(asset_amo) # real amo  i wish to truncate 
            trunc_amo=np.round(np.floor(real_amo/lotSize) * lotSize,precision) # faken gottarefactor this shiet 
            return trunc_amo
        
        key='close' 
        curPrice=float(self.check_current_price(symbol=symbol)[key]) # current price fren 
        real_amo_to_buy = dollar_amo / curPrice # amount of asset i want to buy from dollar terms 
        if real_amo_to_buy < minQty:
            self.log_variable(var=real_amo_to_buy, msg=f' cant buy such small amount of {symbol}, lowest amo possible is {minQty}')
        lot_size_amo = np.floor(real_amo_to_buy/minQty)  * minQty # amount of asset i can buy due to steps 
        lot_size_amo=np.round(lot_size_amo,precision) # amount of assets i can buy due to steps and precision 
        if 0: 
            print(real_amo_to_buy)
            print(lot_size_amo)
        return lot_size_amo
        
    @return_on_failure({})
    def market_buy_dollar_amo(self,symbol,dollar_amo=20):      
        key='high' # least favorable for buying
        quantity=self.calculate_quantity(symbol=symbol,dollar_amo=dollar_amo)
        self.log_variable(var = '', msg=' executing market_buy_dollar_amo ', symbol=symbol,dollar_amo=dollar_amo,quantity=quantity)
        
        self.order=self.client.create_order(symbol=symbol,
                                side=self.order_side_d['buy'],
                                type=self.order_type_d['market'],
                                quantity=quantity
                                )

        if self.order['status']!='FILLED':
            print('dupa, order not fully filled!!')
        self.order['price']=self.calculate_from_fills(self.order['fills'])
        
        self.log_variable(var=self.order, msg = '       after buy order status ')
        return self.order,self.order['status']

    # sells order by it's id 
    def market_sell_orderid(self,symbol,orderid : int ):    
        order=self.client.get_order(symbol=symbol,orderId=orderid) # get order you wish to close 
        self.log_variable(var='', msg = 'executing market_sell_orderid.order ',symbol=symbol,orderid=orderid)
        
        symbol=order['symbol']                                     
        amo=order['executedQty']                                   
        precision=len(amo.split('.')[1])
        amo=np.round(float(amo),precision) 
        # if amo from trade is greater than amount of bags then sell amount of bags 
        # when you make an order you pay a commission, but once you access this order back you don't have commission amo available from binance 
        bags_amo=self.get_my_bags(specific_asset=symbol) # get amount available  
        if float(amo)>float(bags_amo):
            amo=self.calculate_quantity(symbol=symbol,dollar_amo=-1,asset_amo=bags_amo)
            self.log_variable(var=amo,msg=' selling all holdings rather than orderqty ',bags_amo=bags_amo)
             
        self.order=self.client.create_order(symbol=symbol,
                                            side=self.order_side_d['sell'],
                                            type=self.order_type_d['market'],
                                            quantity=amo)
        if self.order['status']!='FILLED':
            print('dupa, order not fully filled!!')
            
        self.log_variable(var=self.order, msg = '       after sell order status ')
        return self.order,self.order['status']
    
    def try_to_close_all_by_symbol(self,symbol):
        f= lambda x:(datetime.datetime.now() -  datetime.datetime.fromtimestamp(int(x)/1000)).total_seconds()
        r=None
        all_orders_list=self.get_historical_orders(symbol=symbol,last_n=5)
        for d in all_orders_list:
            if d['side']!='BUY':
                continue
            if f(d['time'])>60*60*24: # not closing orders older than a day 
                continue
            
            orderid=d['orderId']
            order_symbol=d['symbol']
            try:
                r=self.market_sell_orderid(symbol=order_symbol,orderid=orderid)
            except Exception as er:
                self.log_variable(var=er,msg='error during try_to_close_all_by_symbol')
                self.log_variable(var=r, msg='try_to_close_all_by_symbol response')
                self.log_variable(var=d, msg='order tried to close')
                
    def get_historical_orders(self,symbol,last_n=-1):
        orders=self.client.get_all_orders(symbol=symbol)
        if last_n !=-1:
            return orders[-last_n:]
        return orders 
        
    def get_server_time(self) -> str:
        t=str(self.client.get_server_time()['serverTime'])
        utc_server_time=datetime.datetime.fromtimestamp(int(t[:10])).strftime(self.time_format)
        return utc_server_time



# walkthroughs: 
def show_bags_example(b: BApi):
    r=b.get_my_bags()
    print(r)
    
def market_buy_example(b : BApi):
    symbol='ADAUSDT'
    dollar_amo=20
    test_order=False 
    response=b.market_buy_dollar_amo(symbol=symbol,dollar_amo=dollar_amo,test_order=test_order)
    print(response) 
    
def close_long_orderid(b: BApi):
    symbol='ADAUSDT'
    orderid=3754593233
    r=b.market_sell_orderid(symbol=symbol,orderid=orderid)
    print(r) 
    
def close_all_orders_by_symbol(b: BApi):
    symbol='BUSDUSDT'
    b.try_to_close_all_by_symbol(symbol=symbol)
    
def check_current_price(b: BApi):
    r=b.check_current_price(symbol='ADAUSDT')
    print(r)
    
if __name__=='__main__':
    b=BApi(api_config='./configs/binance_api.json')
    #b.get_historical_orders(symbol='')
 
 #   show_bags_example(b=b)
    
    if 0:
        market_buy_example(b=b)
    if 0: 
        close_long_orderid(b=b)    
    if 0:
        close_all_orders_by_symbol(b=b)
    if 1:
        while True:
            check_current_price(b=b)
            time.sleep(1)
    
#    show_bags_example(b=b)
    
    