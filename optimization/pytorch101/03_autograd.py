from re import L
from urllib import request
import torch as t 
import os 

cls = lambda : os.system('CLS')

x=t.rand(3,requires_grad=True) # this will make pytorch make computation graph 
print(x)

y=x+2
print(y)
y=x/11 
print(y)

z=y*y*2
print(z)
z=z.mean() # mean backward 
print(z)
z.backward() # calculates gradient - dz/dx !!  can be implicit
print(x.grad)
# if your tensor s not a scalar you need to give backward a vector 

v=t.tensor([0.1,1.0,0.01])
zz=t.rand(3,requires_grad=True)
zz.backward(v)
print(zz.grad)

# prevent pytorch from tracking history !!
# 3 options:
# x.reauires_grad(False)
# x.detach() -> creates new tensor 
# with t.no_grad():
#    do your operations 
cls()

x=t.rand(3,requires_grad=True)
print(x) 
x.detach_()
print(x) # no gradient 


cls() 
# 6. during training steps you have to clear weights grad !!  
weights=t.ones(4,requires_grad=True)
# training loop 
# dummy loop 
for epoch in range(2):
    model_output = (weights*3).sum()
    # calculate gradient 
    model_output.backward()
    print(weights.grad)
    # first output is 3s 
    # second is 6s 
    # third is 9s 
    # we have to empty the gradient which acumulates 
    weights.grad.zero_()
    
cls()
# same concept - zeroing weights with torch optimizer ! 
optimizer=t.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()
    
    
    
    