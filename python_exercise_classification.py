# author :momo
#time :2018.09.20
# classification network

#prepare training data
import torch
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from torch.autograd import Variable


n_data=torch.ones(100,2)                   #数据基本形态
x0=torch.normal(2*n_data,1)                #类型0 x data(tensor),shape=(100,2)
y0=torch.zeros(100)                        #类型0 y data(tensor),shape=(100,1)
x1=torch.normal(-2*n_data,1)               #类型1 x data(tensor),shape=(100,2)
y1=torch.ones(100)                         #类型1 y data(tensor),shape=(100,1)


#注意x,y数据的数据形式是一定要像下面一样
x=torch.cat((x0,x1),0).type(torch.FloatTensor)
y=torch.cat((y0,y1),).type(torch.LongTensor)

#torch只能在variable上训练
x,y=Variable(x),Variable(y)

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()
#plt.scatter(x.data.numpy(),y.data.numpy())
#plt.show()

class Net(torch.nn.Module):
	def __init__(self,n_feature,n_hidden,n_output):
		super(Net,self).__init__()
		self.hidden=torch.nn.Linear(n_feature,n_hidden)
		self.out = torch.nn.Linear(n_hidden, n_output)

	def forward(self,x):
		#正向传播输入址，网络分析输出值
		x = F.relu(self.hidden(x))         #激励函数（隐藏层的线性值）
		x = self.out(x)                    # 输出值，但是这个不是预测值，预测值还需要再另外计算
		return x

net=Net(n_feature=2,n_hidden=10,n_output=2)           #n_feature对应输入数据x的维数，输出值对应种类数

# print (net)

optimizer=torch.optim.SGD(net.parameters(),lr=0.02) #传入net的所有参数，学习率设置

#算误差的时候，注意真实值不是one-hot形式的，而是1D tensor,(batch,1)
#但是预测值是2D tensor(batch, n_classes)

loss_func=torch.nn.CrossEntropyLoss()

plt.ion()   # 画图
plt.show()

for t in range(100):
	out=net(x)               #喂给net训练数据x,输出分析值
	loss=loss_func(out,y)

	optimizer.zero_grad()   # 清空上一步的残余更新参数值
	loss.backward()         # 误差反向传播, 计算参数更新值
	optimizer.step()        # 将参数更新值施加到 net 的 parameters 上

	if t%2==0:
		plt.cla()
		#再加上softmax的激励函数后的最大概率才是预测值
		prediction = torch.max(F.softmax(out),1)[1]     #取第一维度最大值，并返回索引值 ,0或者1
		pred_y = prediction.data.numpy().squeeze()
		target_y = y.data.numpy()            #参考标签值 y
		plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y,s=100,lw=0,cmap='RdYlGn')
		accuracy=sum(pred_y==target_y)/200
		plt.text(1.5,-4,'Accuracy=%.2f' % accuracy,fontdict={'size':20,'color':'red'})
		plt.pause(0.2)


plt.ioff()
plt.show()













