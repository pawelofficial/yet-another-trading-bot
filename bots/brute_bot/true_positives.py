import numpy as np 
import matplotlib.pyplot as plt 
import random 
from scipy.stats import norm
import torch 

N=7
torch.manual_seed(0)
t1=torch.rand(N,1).round()
torch.manual_seed(2)
t2=torch.rand(N,1).round()

def myloop(t1,t2):
    tp=[]
    tn=[]
    for i in range(t1.size(0)):
        t1_item=t1[i,:].item()
        t2_item=t2[i,:].item()
        print(t1_item,t2_item)
        if t1_item==t2_item==1:
            tp.append(1)
        if t1_item==t2_item==0:
            tn.append(1)
    return sum(tp)/sum(t1).item()  ,sum(tn) / len( torch.where(t1==0)[0])
            




# tp: 1 -> 1

tp = lambda t1,t2 : torch.mul(t1,t2).sum()/t1.sum() # true positives 
# tn: 0 -> 0 
tn = lambda t1,t2: torch.mul( torch.where(t1==0,1,0), torch.where(t2==0,1,0)).sum() / len ( torch.where(t1==0)[0])
# fp -> 1,0 
fp = lambda t1,t2 : torch.mul(torch.where(t1==1,1,0)  , torch.where(t2==0,1,0)).sum() / len(torch.where(t1==1)[0]) 
# fn -> 0,1 
fn=lambda t1,t2 : torch.mul(torch.where(t1==0,1,0)  , torch.where(t2==1,1,0)).sum() / len(torch.where(t1==0)[0]) 



tw=torch.where(t1==0,1,0)
#print(tw)

l1=[0,0,0,1,1,1]
l2=[1,0,0,1,1,1]

#l1=[0,0,0]
#l2=[0,0,0]

t1=torch.tensor(l1)#.view(len(l1),1)
t2=torch.tensor(l2)#.view(len(l2),1)

print(torch.where(t1==1))


print(f'tp: {tp(t1,t2)}' )
print(f'tn: {tn(t1,t2)}' )
print(f'fp: {fp(t1,t2)}' )
print(f'fn: {fn(t1,t2)}' )



y=torch.rand(10,1)
print(y)
