import torch 
import pandas as pd 
import torch.nn as nn 
from indicators import indicators
import numpy as np 
import matplotlib.pyplot as plt 
import datetime 
import help_funs as hf
import signal
import time
import readchar


def handler(signum, frame):
    msg = "Ctrl-c was pressed. Do you really want to exit? y/n "
    print(msg, end="", flush=True)
    res = readchar.readchar()
    if res == 'y':
        print("")
        exit(1) 
    else:
        print("", end="\r", flush=True)
        print(" " * len(msg), end="", flush=True) # clear the printed line
        print("    ", end="\r", flush=True)
 
 
signal.signal(signal.SIGINT, handler)

class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear=nn.Linear(n_input_features,1)
        
    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x) )
        return y_predicted
    
    
class FitModel(torch.nn.Module):
    def __init__(self, num_units=-1,n_features=-1):
        super(FitModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_features, num_units)
            ,torch.nn.ReLU()
            ,torch.nn.Linear(num_units,1)
        )
    def forward(self, x):
        return self.model(x)
    
    
        
if __name__=='__main__':
    filename='BTC-USD2022-01-01_2022-03-30'
    filename='dump_BTC-USD2022-01-01_2022-01-30_'
#    filename='dump_BTC-USD2022-01-10_2022-01-18_'
#    filename='dump_BTC-USD2022-01-01_2022-03-30_'
    df=pd.read_csv(f'./data/{filename}.csv')
#    df=df[300:].copy(deep=True).reset_index()


    cols=[c for c in list(df.columns) if c not in ('close','open') ]
    i=indicators(df=df)    
    
    roi=i.calculate_roi(inplace=False)    
    i.df['roi']=roi
    percentile=np.percentile(roi[roi>1.01],50)

    i.df=i.df.iloc[300:].copy(deep=True).reset_index()
    roi=i.df['roi']
    
    percentile=1
    roi_labels=roi.apply(lambda x: int(x>percentile)).to_numpy()
    roi_labels_torch=torch.from_numpy(roi_labels.astype(np.float32)).view(roi_labels.shape[0],1)
    roi_torch=torch.from_numpy(roi.to_numpy().astype(np.float32)).view(roi_labels.shape[0],1)
    weightsz=torch.from_numpy(roi.to_numpy().astype(np.float32)).view(roi_labels.shape[0],1)

    X=i.df[cols].to_numpy(dtype=int)
    Xt=torch.from_numpy(X.astype(np.float32) ) # labels tensor 
    n_samples, n_features = Xt.shape
        
#    criterion = nn.CrossEntropyLoss() # binary cross entropy 
#    model=FitModel(num_units=len(cols))
#    model=LogisticRegression(n_input_features=len(cols))
#    optimizer = torch.optim.SGD(model.parameters(),lr=0.00025) # optimizer 

    y=roi_labels_torch
    
    if 0:    
        N=100
        x=np.linspace(0,10,N)
        y=np.array([i**2 for i in x])
        Xt=torch.from_numpy(x.astype(np.float32)).view(len(x),1)
        y=torch.from_numpy(y.astype(np.float32)).view(len(y),1)
    
    model=FitModel(num_units=Xt.shape[0],n_features=Xt.shape[1])
    model.load_state_dict(torch.load('./models/model_chad.pt'))
    optimizer = torch.optim.SGD(model.parameters(),lr=0.00025) # optimizer 
    criterion=nn.MSELoss()
    
    if 0: # cool plot 
        plt.imshow(Xt,cmap='gray')
        plt.show()

    k=1000
    N=500
    n=-10
    while n<N*k and n>0:
        if 1:
            if n/1000==n//1000:
                print(n)
                print(l.item())
                print('no of entries: ',len(torch.where(y_pred.round()==1)[0]) )
                torch.save(model.state_dict(), f'./models/model.pt')
                    
            n+=1
            y_pred=model(Xt)
            l=criterion(y_pred,y)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
    y_pred=model(Xt)          
    now=datetime.datetime.now().isoformat()[:16].replace('-','').replace(':','').replace('T','')
    torch.save(model.state_dict(), f'./models/model_{now}.pt')
    for k,v in i.metrics_d.items():
        print(f'{k}: {v(y_pred.round(),y)}')
    
    if 1:

        y_pred_entry_ser=pd.Series(y_pred.detach().numpy().round()[:,0])
        i.df['entry']=y_pred_entry_ser
        df=i.df
        msk=i.df['entry']==1
        print('no of entries:',len(i.df[msk]))
        i.calculate_roi()
        i.calculate_exits()
        fig,ax=plt.subplots(2,1,sharex=True)
        roi_msk=df['roi']>1
        ax[0].plot(df.index,df['close'])
        ax[0].plot(df[roi_msk].index,df[roi_msk]['close'],'or')
        ax[0].grid()
        ax[1].grid()
        entry_msk=i.df['entry']==1
        exit_msk=i.df['exit']==1
        #ax[1].plot(df[entry_msk].index,df[entry_msk]['close'],'^g')
        #ax[1].plot(df[exit_msk].index,df[exit_msk]['close'],'vr')
        ax[1].plot(df.index,df['roi'])
        ax[0].plot(df[entry_msk].index,df[entry_msk]['close'],'xg')
        plt.show()
        
        #i.plot_stuff(extra_col='roi',plot_flag=True)
        #i.df['roi_labels'] = i.df['roi'].apply(lambda x: int(x>percentile))
    
        
    

    