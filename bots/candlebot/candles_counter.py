# this scripty script shows you that there is a distribution of consecutive numbers of candles of the same color
# it might be shocking to you to realize that 2 green candles in a row happen more often than 8 candles in a row !
# maybe this difficult to grab concept can be used as a basis for a profitable and very advanced trading bot 
import sys 
sys.path.append("../..") # makes you able to run this from parent dir 
from myfuns import myfuns as mf 
from data import coinbase_api
from optimization.NN_capstone1 import NNfuns as nn 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 



def countero2(ser):
    colors=list(ser)            # list of zeros and ones 
    counts=[-1 for i in colors] # declare colors of counts 
    
    for no,(color,count) in enumerate(zip(colors,counts)):
        if no==0:           # first count equals a color  
            counts[no]=color
        else:               # next counts equal previous count + current color 
            counts[no]=counts[no-1]+color
        if color==0:        # if color is zero then reset the count 
            counts[no]=color
#        print(n,color,count)
#    print(f'counts {counts}')   
#    print(f'colors {colors}')   
    return counts 


def read_from_clipboard():
    # copy paste your df from log to clipboard
    # change timestamp formats from 2022-11-20 08:45:00 to 2022-11-20T08:45:00
    # in python shell execute 
    # df=pd.read_clipboard()
    # df.to_csv(path_or_buf='C:\\Users\\zdune\\Documents\\moonboy\\yet-another-trading-bot\\bots\\candlebot\\dobry_df.csv',sep='|',header=True,index=False)
    df=pd.read_csv(filepath_or_buffer='./dobry_df.csv',sep='|')

#    df.drop(labels=['red_cnt','green_cnt'],axis=1,inplace=True)
    df.drop(labels=['volume','open_epoch','close_epoch'],inplace=True,axis=1)
    
    if 1: # drop original counts 
        df.drop(labels=['green_cnt','red_cnt'],axis=1,inplace=True)
    else:
        df.rename(columns={'red_cnt':'or_red_cnt','green_cnt':'or_green_cnt'},inplace=True)
    return df 
    
if __name__=='__main__':
    if 0:  # read data from api, aggregate and dump 
        nn.read_raw_data_from_api()
        #df=nn.read_df_from_file()
        agg_df=nn.aggregate_df(df=df,scale=5)
        nn.dump_df(df=agg_df,filename='agg_df_5')
    
#    df=nn.read_df_from_file(filename='agg_df_5')
    df= read_from_clipboard()

    
    
    df['green']=df['close']>df['open']
    df['red']=df['close']<=df['open'] # <= ?? 
    
    df['green']=df['green'].astype(int)
    df['red']=df['red'].astype(int)
    
    
    counts=countero2(df['green'])
    df['green_counts']=counts
    print(df)
    counts=countero2(df['red'])
    df['red_counts']=counts
    print(df)

    if 0:
        plt.hist(x=(df['green_counts'],df['red_counts']),log=True,bins=20)
        plt.title('histogram of consecutive green/red candles')
        plt.ylabel('no of occurences')
        plt.xlabel('len of streak')
        plt.show()


import binance 