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


def candles_counter(ser): # counts consecutive occurences of value in a series 
    #- i used to be able to write recursion but now not so much apparently :( :( 
    def countero(l,index):
        l= l[:index+1]
        cnt=1
        i=0
        while True:
#            print(l)
#            print(index,i)
#            input()
            ass=l[index-i]==l[index-i-1]
            if ass:
                i=i+1
                cnt=cnt+1
            else:
                break
            if i>=len(l):
                break

        return cnt 
    l=list(ser)
    counts=[]
    index=len(l)-1
    while index>=1:
        counts.append(countero(l,index))
        index=index-1
    counts.append(1)
    counts.reverse()
    return counts

if __name__=='__main__':
    if 0:  # read data from api, aggregate and dump 
        nn.read_raw_data_from_api()
        #df=nn.read_df_from_file()
        agg_df=nn.aggregate_df(df=df,scale=5)
        nn.dump_df(df=agg_df,filename='agg_df_5')
    
    df=nn.read_df_from_file(filename='agg_df_5')
    
    df['green']=df['close']>df['open']
    df['red']=df['close']<=df['open'] # <= ?? 
    
    df['green']=df['green'].astype(int)
    df['red']=df['red'].astype(int)
    

    counts=candles_counter(df['green'])
    df['green_counts']=counts
    counts=candles_counter(df['red'])
    df['red_counts']=counts
    print(df)

    if 0:
        plt.hist(x=(df['green_counts'],df['red_counts']),log=True,bins=20)
        plt.title('histogram of consecutive green/red candles')
        plt.ylabel('no of occurences')
        plt.xlabel('len of streak')
        plt.show()


import binance 