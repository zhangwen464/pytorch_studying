# author :momo
#time :2018.09.20
# regression network

#prepare training data


import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt 
import torch.nn.functional as F

x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size())

#用Variable来修饰这些数据tensor
x,y =torch.autograd.Variable(x),Variable(y)

#画图
plt.scatter(x.data.numpy(),y.data.numpy())
plt.ylim((-0.2,1.4))
plt.xlim((-1.5,1.5))

# plt.show()


class Net(torch.nn.Module): #继承torch的模型
	def __init__(self,n_feature,n_hidden,n_output):
		super(Net,self).__init__() #继承__init__
		#定义每层用什么样的形式
		self.hidden = torch.nn.Linear(n_feature, n_hidden) #隐层线性输出
		self.predict = torch.nn.Linear(n_hidden, n_output) #输出层线性输出


	def forward(self,x): #同时也是Module中forward功能
		#正向传播输入值，神经网络分析输出值
		x=F.relu(self.hidden(x))  #激励函数，（隐藏层的线性值）
		x=self.predict(x)
		return x

net=Net(n_feature=1,n_hidden=10,n_output=1)

# print (net)



optimizer=torch.optim.SGD(net.parameters(),lr=0.5) #传入net的所有参数，学习率是0.5
loss_func=torch.nn.MSELoss() #预测值和真实值的误差计算公式


plt.ion() # 打开交互模式画图


for t in range(200):
	prediction=net(x)

	loss=loss_func(prediction,y)

	optimizer.zero_grad()  #清空上一步的参与更新参考值
	loss.backward()        #误差反向传播，计算参数更新
	optimizer.step()       #将参数更新值施加到net的parameters上

	if t % 5==0:
		plt.cla()
		plt.scatter(x.data.numpy(),y.data.numpy())
		plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
		plt.text(0.5,0,'Loss=%.4f' % loss.data[0],fontdict={'size':20,'color':'red'})
		plt.show()
		plt.pause(0.5)





