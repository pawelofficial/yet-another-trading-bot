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
from binance_api2 import BApi


# walkthroughs: 
def show_bags_example(b: BApi):
    r=b.get_my_bags()
    print(r)
    
def market_buy_example(b : BApi,symbol,dollar_amo):
    #    symbol='BUSDUSDT'
    #test_order=False 
    response=b.market_buy_dollar_amo(symbol=symbol,dollar_amo=dollar_amo)
    print(response) 
    
def close_long_orderid(b: BApi,symbol,orderid : int ):
    #    symbol='BUSDUSDT'
    #    orderid=765402603
    r=b.market_sell_orderid(symbol=symbol,orderid=orderid)
    print(r) 
    
def close_all_orders_by_symbol(b: BApi,symbol):
    #symbol='BUSDUSDT'
    b.try_to_close_all_by_symbol(symbol=symbol)
    
    
def show_historical_orders(b : BApi, symbol='ADAUSDT'):
    r=b.get_historical_orders(symbol=symbol)    
    f= lambda x: datetime.datetime.fromtimestamp(int(x)/1000).isoformat()
    for d in r:
#        print(d)
        print(d['symbol'],d['orderId'],d['side'],f(d['time']),d['executedQty'],d['status'])
    
if __name__=='__main__':
    b=BApi(api_config='./configs/binance_api.json')


    #b.get_historical_orders(symbol='')
    show_bags_example(b=b)
    symbol='ADAUSDT'
    dollar_amo=20
    if 0:
        show_historical_orders(b=b)
    if 0:
        r=market_buy_example(b=b,symbol=symbol,dollar_amo=20)
    if 0: 
        close_long_orderid(b=b,symbol=symbol,orderid=3754732773)    
    if 0:
        close_all_orders_by_symbol(b=b,symbol='ADAUSDT')


