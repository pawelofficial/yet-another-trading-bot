# sofotmax is a cool function 
import numpy as np 
import torch 

# numpy softmax implementation 
def softmax(x : np.array):
    l=np.exp(x)
    return l / np.sum( l,axis=0  )

ar=np.array([1,2,3,4])
s=softmax(ar)
print(s)

# now using pytorch 
x=torch.tensor(ar,dtype=torch.float32)
output=torch.softmax(x,dim = 0 )
print(output)