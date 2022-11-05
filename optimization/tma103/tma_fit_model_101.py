# this script uses feautres_labels.csv from tma_get_signals and tries to match a pytorch model to it 
import pandas as pd 
import torch 
from sklearn.model_selection import train_test_split 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
sys.path.append("../..")
sys.path.append("..")
from myfuns import myfuns as mf


class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear=nn.Linear(n_input_features,1)
        
    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x) )
        return y_predicted




def torch_loop(model, loss_fn,optimizer,X_train,Y_train,N=100):
    
    for epoch in range(N):
        y=model(X_train) # calling the module class invokes forward function fren ! loss calculation
        
        l=loss_fn(y,Y_train) # order here is important ! 
        # graduebt 
        print(l)
        l.backward()    # using backward method to calcylate dl/dw gradient 
        # update weights 
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 100 ==0:
            pass
#            print(epoch)
    
def tensor_to_ser(t):
    return pd.Series(t[:,0].numpy())


def acc(model,X_test,Y_test):
    with torch.no_grad():
        y_predicted=model(X_test).round()
        #print(y_predicted[:,0].numpy())
        #print(y_predicted.shape)

    return y_predicted 


if __name__=='__main__':
    csv_prefix='BTC-USD2022-01-10_2022-01-18'
    labels_df=pd.read_csv(f'{csv_prefix}_labels_df.csv')
    features_df=pd.read_csv(f'{csv_prefix}_features_df.csv')
    plot_df=pd.read_csv(f'{csv_prefix}_plot_df.csv')
    
    X=torch.from_numpy(features_df.to_numpy().astype(np.float32))
    Y=torch.from_numpy(labels_df['LONGS_SIGNAL'].to_numpy().astype(np.float32))
    Y=Y.view(len(Y),1)

    X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=1234)
    
    n_samples, n_features = X_train.shape 
    model=LogisticRegression(n_features)                # model 
    loss_fn = nn.BCELoss() # binary cross entropy     # loss function 
    loss_fn=nn.MSELoss()
    loss_fn=nn.BCEWithLogitsLoss()
    loss_fn=nn.L1Loss()
    #loss_fn= nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01) # optimizer 
    
    torch_loop(model=model,loss_fn=loss_fn,optimizer=optimizer,X_train=X_train,Y_train=Y_train,N=1000)
    y_predicted=acc(model=model,X_test=X_test,Y_test=Y_test)
    
    y_test_ser=tensor_to_ser(Y_test)
    msk=y_test_ser>0
    y_pred_ser=tensor_to_ser(y_predicted)

    
    dic={'y_test':y_test_ser,'y_pred':y_pred_ser}
    df=pd.DataFrame(dic)
    

    plt.show()
    print(df[msk])
    
    exit(1)

    mf.plot_candlestick(df=plot_df,longs_ser=ser*plot_df['low'])
    plt.show()
    
    
    