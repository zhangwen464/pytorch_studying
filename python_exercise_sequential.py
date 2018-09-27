# author :momo
#time :2018.09.20
# net1 & net2 comparision


import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
from torch.autograd import Variable


class Net(torch.nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net,self).__init__()
		self.hidden=torch.nn.Linear(n_feature, n_hidden)
		self.perdict=torch.nn.Linear(n_hidden,n_output)

	def forward(self,x):
		x=F.relu(self.hidden(x))
		x=self.perdict(x)
		return x

net1=Net(1,10,1)

#我们用class继承了一个torch中的神经网络结构，但是用另外一种方式更快

net2=torch.nn.Sequential(
	torch.nn.Linear(1,10),
	torch.nn.ReLU(),
	torch.nn.Linear(10,1)
	)


print (net1)

print (net2)

#我们发现net2多显示了一些内容，因为net2中把激励函数也纳入了网络中。
#但是在net1中，激励函数实际上是在forward功能中才被调用的
#相比net2,net1的好处是，你可以根据个人需要添加个性化你自己的前向传播过程，比如rnn。
#如果你不需要这些个性化过程，net2更加简洁合适。






