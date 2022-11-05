import torch as t 

# example of backpropagation 

x=t.tensor(1.0) # x= 1 
y=t.tensor(2.0) # y = 2 
w=t.tensor(1.0,requires_grad=True) # initial w = 0 


# forward pass and compute the loss 
y_hat = w * x  # our fun -> 1 
loss = (y_hat - y )**2  # this we want to minimize 

print(loss)

# backward pass - torch computes local gradiants and backward pass for us 

loss.backward() 
print(w.grad) # gradient after first backward pass ! 

# update our weights 
# next forward and backward pass 
# hopefully loss -> 0 


