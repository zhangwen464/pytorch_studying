# author :momo
#time :2018.09.20
# save model

import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
from torch.autograd import Variable

x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size())
x,y=Variable(x,requires_grad=False),Variable(y,requires_grad=False)

plt.scatter(x.data.numpy(),y.data.numpy())
plt.ylim((-0.2,1.4))
plt.xlim((-1.5,1.5))

def save():
	#构建网络
	net1 = torch.nn.Sequential(
		torch.nn.Linear(1,10),
		torch.nn.ReLU(),
		torch.nn.Linear(10,1)
		)

	optimizer=torch.optim.SGD(net1.parameters(),lr=0.5)
	loss_func=torch.nn.MSELoss()

	#训练
	for t in range(100):
		prediction=net1(x)
		loss=loss_func(prediction,y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	plt.figure(1,figsize=(10,3))
	plt.subplot(131)
	plt.title('net1')
	plt.scatter(x.data.numpy(),y.data.numpy())
	plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)


#保存网络
	torch.save(net1,'net.pkl')       #保存整个网络
	torch.save(net1.state_dict(),'net_params.pkl')        #只保存网络中的参数（速度快，占内存少）



def restore_net():
	net2=torch.load('net.pkl')
	prediction=net2(x)

	plt.subplot(132)
	plt.title('net2')
	plt.scatter(x.data.numpy(),y.data.numpy())
	plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)



def restore_params():
	net3=torch.nn.Sequential(
		torch.nn.Linear(1,10),
		torch.nn.ReLU(),
		torch.nn.Linear(10,1))

	net3.load_state_dict(torch.load('net_params.pkl'))
	prediction=net3(x)

	plt.subplot(133)
	plt.title('net3')
	plt.scatter(x.data.numpy(),y.data.numpy())
	plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)


# 运行测试
save()
restore_net()
restore_params()











