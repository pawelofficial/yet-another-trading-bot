
import pandas as pd 
# funs like that 
from myfuns import myfuns as mf
# instances like that 
from mymath.mymath import * 
from data.coinbase_api import * 



#1.  download data from coinbase from main dir
if 0:   
    utils=ApiUtils(CoinbaseExchangeAuth,coinbase_api)
    api=utils.init(config_filepath='./credentials/api_config.json')
    utils.download_last_n_days(api=api,n=3,path='./data/')

#2. read a csv 
filename='BTC-USD2022-05-29_2022-06-05'
df=pd.read_csv(f'./data/{filename}.csv')

#3. do some math on csv 
m=myMath()
df=m.aggregate(df=df,scale=15)


#4. get signals on df 


#5. check performance 