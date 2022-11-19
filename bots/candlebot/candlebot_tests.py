from candlebot import candlebot
import pandas as pd 
import time 


    
def open_and_close_an_order(c : candlebot):
    # opens a dummy order and closes it and checks if assertions are ok 
    response=c.market_buy()
    time.sleep(2)
    response2=c.market_sell_tradeid(orderid=response['orderId'])
    msk1=c.pnl_df['tradeid']==response['orderId']
    msk2=c.pnl_df['tradeid']==response2['orderId']
    
 
    a1=False
    if (c.pnl_df[msk1]['status']=='CLOSED-LONG').all():
        a1=True 
    a2=False
    if (c.pnl_df[msk2]['status']=='FILLED').all():
        a2 = True
          
    return a1&a2 
    
def update_pnls(c:candlebot): # checks if updating pnl works 
    c.market_buy()
    c.market_buy()
    c.update_pnl_df()
    
# forces a stop loss 
def force_sl(c: candlebot):
    response=c.market_buy()
    response=c.market_buy()
    response=c.market_buy()
    c.update_pnl_df()
    print(c.pnl_df)
    c.execute_sl(force=True)
    print(c.pnl_df)
    c.update_pnl_df()

def force_tp(c:candlebot):
    response=c.market_buy()
    response=c.market_buy()
    response=c.market_buy()
    c.update_pnl_df()
    print(c.pnl_df)
    c.execute_tp(force=True)
    print(c.pnl_df)
    
def force_trtp(c:candlebot):
    response=c.market_buy()
    response=c.market_buy()
    response=c.market_buy()
    c.update_pnl_df()
    print(c.pnl_df)
    c.execute_trtp(force=True)
    print(c.pnl_df)
    
# checks stop losses for real real ! 
def real_sl(c:candlebot):
    c.b.dummy_price=100 # let's buy high 
    response=c.market_buy()
    response=c.market_buy()
    response=c.market_buy()
    c.b.dummy_price=0
    print('your buy orders ')
    print(c.pnl_df)
    c.update_pnl_df()
    print('your pnls after price drop ')
    print(c.pnl_df)
    c.execute_sl()
    print('your pnls after stop losses ')
    print(c.pnl_df)
    
def real_tp(c:candlebot):
    c.b.dummy_price=0.001 # let's buy low
    response=c.market_buy()
    response=c.market_buy()
    response=c.market_buy()
    c.b.dummy_price=100 # elon's tweet 
    print('your buy orders ')
    print(c.pnl_df)
    c.update_pnl_df()
    print('your pnls after price drop ')
    print(c.pnl_df)
    c.execute_tp()
    print('your pnls after stop losses ')
    print(c.pnl_df)
    
    
# tak trzeba zyc:  
# icebergs 
# -- thile True 
#   - check for signal 
#   - buy when signal
#   - buy ok 
#   - buy not ok 
#   - check for exit strategy 
#       - SL
#       - TRTP
#       - TP
#   - sell when exist strategy 
    
if __name__=='__main__':
    
    c=candlebot()
#    open_and_close_an_order(c=c)
    #update_pnls(c=c)   
#    force_sl(c=c)
#    force_tp(c=c)
#    force_trtp(c=c)
#    real_sl(c=c)
    real_tp(c=c)