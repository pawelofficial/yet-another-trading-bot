# this is a second version of using pytorch to do tma optimization
# hopefylly better than tma 101 


from torch.utils.data import Dataset, DataLoader 
import pandas as pd 
import torch 
import matplotlib.pyplot as plt 
import sys 
sys.path.append("../..")
sys.path.append("..")
from myfuns import myfuns as mf
from lookahead_entries import *
import torchvision 
import torch.nn as nn 

# class for preparing the data 
class prep_data(): # transformation 
    def __init__(self,path = None, filename = None ) -> None:
        self.scale=5
        self.df=self.read_csv(path=path,filename=filename,scale=self.scale)
        self.df['index']=self.df.index

        # dictionary with pandas functions 
        self.fun_d={}
        self.fun_d['sma']=lambda df,n,col: df[col].rolling(window=n).mean() # rolling sma 
        self.fun_d['max']=lambda df,n,col: df[col].rolling(window=n).max() # rolling sma 
        self.fun_d['std']=lambda df,n,col: df[col].rolling(window=n).std() # rolling std 
        self.fun_d['nrzscore']=lambda df,n,col: self.zscore(n,col)                # not rolling  
        self.fun_d['zscore']=lambda df,n,col : df[col].rolling(window=n).apply(self.rolling_zscore) # rolling mean based normalization
        self.fun_d['norm']=lambda df,n,col : df[col].rolling(window=n).apply(self.rolling_normalize) # rolling mean based normalization
        
    # read csv from file 
    def read_csv(self,path,filename,scale):
        if not path:
            path='../../data/'
        if not filename:
            filename='BTC-USD2022-08-24_2022-08-27'
        df=pd.read_csv(f'{path}/{filename}.csv', lineterminator=';')
        return mf.aggregate(df=df, scale=scale, src_col='timestamp')
    
    # zscore function 
    def rolling_zscore(self,ser):
        mu=ser.mean()
        std=ser.std()
        return (ser.iloc[-1] - mu ) / std 
    # not rolling z score 
    def zscore(self,window,col): # rolling zscore 
        r = self.df[col].rolling(window=window)
        m = r.mean().shift(1)
        s = r.std(ddof=0).shift(1)
        z = (self.df[col]-m)/s
        return z
    # rolling normalization function 
    def rolling_normalize(self,ser): # rolling normalize based on mean 
        return ser.iloc[-1]/ser.mean()

    # worflow stitching things together 
    def workflow1(self): # normalizes stuff with respect to mean 
        out_df=pd.DataFrame({})
        out_df['close']=self.fun_d['norm'](self.df,200,'close')           # normalized close 
        out_df['open']=self.fun_d['norm'](self.df,200,'open')             # normalized open 
        out_df['candle']=out_df['close']-out_df['open'] # normalized candle
        
        timeframes=mf.get_timeframes(dist_fun='exp',    timeframes_ranges=[1,150],N=4) # timeframes 
        for tf in timeframes:
            out_df[f'sma-{tf}']=self.fun_d['sma'](out_df,tf,'close')
            out_df[f'std-{tf}']=self.fun_d['std'](out_df,tf,'close') 
        out_df.dropna(how='any',inplace=True)
        out_df['original_index']=out_df.index
        out_df.reset_index(inplace=True)
        return out_df
        
    def workflow2(self): # rolling z scorizes stuff 
        self.df['candle']=self.df['close']-self.df['open'] # normalized candle
        timeframes=mf.get_timeframes(dist_fun='exp', timeframes_ranges=[1,150],N=4) # timeframes 
        for tf in timeframes:
            self.df[f'sma-{tf}']=self.fun_d['sma'](self.df,tf,'close')
            self.df[f'std-{tf}']=self.fun_d['std'](self.df,tf,'close') 
        
        # z scorize some of the columns: 
        not_zscore_columns=['ts-5','ts-15','index','original_index']
        zscore_df=pd.DataFrame({})
        for tf in timeframes: # this should be optimized for regex changes in ts column 
            for col in [c for c in self.df.columns if c not in not_zscore_columns]:
                key=f'zsc-{tf}-{col}'
                zscore_df[key]=self.fun_d['zscore'](self.df,tf,col)
            
        zscore_df['index']=zscore_df.index
        zscore_df['close']=self.df['close']
        return zscore_df


    def get_labels(self,df): # uses lookahead_entries script and adds signal columns to self.df 
        lookahead_df, p_longs,p_shorts,percentile_longs,percentile_shorts,lowest_longs,highest_shorts = example_workflow(df=self.df)

        cols=['LONG','SHORT']
        df=df.merge(lookahead_df[cols].astype(int),left_index=True,right_index=True)
        return df , p_longs,p_shorts,percentile_longs,percentile_shorts,lowest_longs,highest_shorts

     
        
# torch dataset class 
class PriceAction(Dataset):
    def __init__(self,df=pd.DataFrame({}) ,transform=None, ycol='LONG') -> None:
        super().__init__()

        self.df=df
        self.transform=transform 
        
        self.ycol=ycol # output column 
        self.xcols=[ c for c in self.df.columns if c !=ycol]

        self.x=self.df[self.xcols]#.to_numpy()
        self.y=self.df[self.ycol]#.to_numpy()
        self.n_samples=self.x.shape[0]

    def __getitem__(self, index):
        xcols=list(self.xcols)
        ycols=[self.ycol]
        
        sample =  self.df[xcols].iloc[index],self.df[ycols].iloc[index]  
        if self.transform: 
            sample = self.transform(sample)
        return sample 
    def __len__(self): 
        return self.n_samples 
    
# transformation that pops unnecessary columns from tensor 
class popColumns():
    def __init__(self,cols=[]) -> None:
        self.dropped_cols=['index','candle','original_index','close','SHORT'] # columns to be excluded from input tensor 
        [self.dropped_cols.append(col) for col in cols]
        
    def __call__(self,sample): #
        inputs, targets = sample
        [inputs.pop(key) for key in self.dropped_cols if key in list(inputs.keys()) ] # pop unnecessary columns from dataframe  if they are present there 
        return inputs,targets

# transformation which  casts dataframes to tensors 
class dfToTensor():
    def __call__(self,sample): 
        inputs, targets = sample
        inputs=inputs.to_numpy()
        targets=targets.to_numpy()
        return torch.from_numpy(inputs),torch.from_numpy(targets)


if __name__=='__main__':
    # prep data 
    t=prep_data()
    zscore_df=t.workflow2()
    df, p_longs,p_shorts,percentile_longs,percentile_shorts,lowest_longs,highest_shorts =t.get_labels(df=zscore_df)
    df2=df.dropna(how='any')
    composed_transform=torchvision.transforms.Compose([popColumns(),dfToTensor()])
    dataset=PriceAction(df=df2,transform=composed_transform)
    dataloader=DataLoader(dataset=dataset,batch_size=1000,shuffle=True,num_workers=1)
    dataiter=iter(dataloader)
    features,labels=dataiter.next()
    print(features)
    print(labels)
#    dataset=PriceAction(df=df,ycl='LONG')
#    print(dataset.x)