import sys
sys.path.append("../..")
import myfuns.myfuns as mf  
import torch 
import os 
import numpy as np 


clearo = lambda : os.system('CLS')
# read csv 
#df=mf.read_csv(filename='../../data/BTC-USD2021-06-05_2022-06-05.csv') 

# 1. empty 
x=torch.empty(1) # empty tensor 
x=torch.empty(2,3) # 2 rows, 3 cols ! 
x=torch.empty(2,3,2)
# 2. random values 
x=torch.rand(2,2)
# 3. zeros 
x=torch.zeros(2,2)
x=torch.ones(2,2)

#!! look at dtypes 
x=torch.ones(2,2,dtype=torch.int)
#print(x.dtype)
#print(x.size())

# tensor from data 
l=[[1,2,3],[1,2,3]]
x=torch.tensor(l)
#print(x)

# !! basic operations 
x=torch.ones(2,2)
y=torch.ones(2,2)
# elementwise operations
z=x+y 
z=torch.add(x,y)
#inplace addition 
y.add_(x) # all _ are inplace 
x.sub_(x)
print(z)
print(y)
print(x)
z.mul_(z)
x=x/x
print(x)
print(z)

# !! slicing 
x=torch.rand(5,3)
print(x)
print(x[:,0]) # all rows column zero 
print(x[1,:]) # row no 1 all columns 
print(x[0,2].item()) # for 1:1 tensors you can get the value like that 


# !! reshaping tensors  - views 
os.system('CLS')
x=torch.rand(4,4)
y=x.view(16) # one row 
print(x)
print(y)
y=x.view(-1,8) # pytorch will figure out that shape you mean is 2,8 
print(y)
y=x.view(16,1) # 16 rows 
print(y)

# numpy to torch 
os.system('CLS')
a=torch.ones(5)
print(a)
b=a.numpy() # this  points to the same memory as a, hence when you inplace modify a it will get modified as well !! 
print(b)
a.sub_(a) # zeros 
print(b) # zeros as well 
a=np.ones(5)
print(a)
b=torch.from_numpy(a)
print(b)
b.sub_(b) # be careful ! 
print(a) 
clearo()

print(torch.cuda.is_available()) # :(   
# using cuda !! 
if torch.cuda.is_available(): # using gpu 
    device=torch.device("cuda")
    x=torch.ones(5,device=device) # creating tensor in gpu 
    # or create it and set it later 
    y=torch.ones(5)
    y=y.to(device)
    z=x+y # performed on gpu ! gpu stands for grand performance user 
    # gpu cant mcconvert tensors to numpy though !!  so you have to move it back 
    z=z.to("cpu")
    z.numpy()
    
# !! requires grad - necessary for back propagation, 
# whenever you have variable you want to optimize you need a gradient 
clearo()
x=torch.ones(5,requires_grad=True )
print(x)

        
    
    
    
    






