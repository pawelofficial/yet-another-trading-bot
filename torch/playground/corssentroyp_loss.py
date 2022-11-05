# for cross entropy you need 
# one hot encoded input 
# computed probabilities of your inputs for example using softmax 

# cross entropy is a loss function 

# sofotmax is a cool function 
import numpy as np 
import torch 
import torch.nn as nn 

def cross_entropy(actual,predicted):
    loss= -np.sum(actual * np.log(predicted))
    return loss 


# labels 
Y=np.array([1,0,0])                     # one hot coded output 
Y_pred_good=np.array([0.7,0.2,0.1])     # good predictions of probabilities of  output 
Y_pred_bad=np.array([0.1, 0.2, 0.6])    # bad prediction 

# calculate cross entropy 
l1=cross_entropy(Y,Y_pred_good)
l2=cross_entropy(Y,Y_pred_bad)
# print cross entropies 
print(f'good prediction cross entropy : {l1:.4f}')
print(f'bad prediction cross entropy: {l2:.4f}')


# now in pytorch - note that nn.CrossEntropy() implemntation in pytorch: 
# Y actual must not be hot one encoded 
# Y_pred has row scores (logits) - no softmax 
# already applies logsoftmax - we dont have to / mustnt do softmax layer on our own 

# let's create labels \
Y=torch.tensor([0]) # size: n_samples x n_classes  --> 1 sample 3 classes 
Y_pred_good=torch.tensor( [[2.0,1.0,0.1] ]  )
Y_pred_bad=torch.tensor([[0.1,0.3,0.6]])

loss_fn=nn.CrossEntropyLoss()
l1=loss_fn(Y_pred_good,Y)
l2=loss_fn(Y_pred_bad,Y)

print(f'good prediction cross entropy : {l1.item():.4f}')
print(f'bad prediction cross entropy: {l2.item():.4f}')

# getting predictions 
_,prediction1=torch.max(Y_pred_good,1) # will return highest probability index  from y_pred_good
_,prediction2=torch.max(Y_pred_bad,1) # will return highest probability index from y_pred_bad 

print(prediction1)
print(prediction2)

# now for 3 samples 
Y=torch.tensor([2,0,1] )
Y_pred_good=torch.tensor([[0.1,0.1,2.0],[2.0,0.1,0.1],[0.1,2.0,0.1]]   )
Y_pred_bad=torch.tensor( [[2.0,0.1,0.1],[0.1,0.1,1.0],[2.0,0.1,0.1 ]]  )

print(loss_fn(Y_pred_good,Y))
print(loss_fn(Y_pred_bad,Y))
