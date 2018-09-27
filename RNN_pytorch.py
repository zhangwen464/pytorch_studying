# author :momo
#time :2018.09.20
# RNN

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.datasets as dsets    #数据库模块
import torchvision.transforms as transforms
import torch.nn.functional as F 
import matplotlib.pyplot as plt 


torch.manual_seed(1)

#Hyper parameters
EPOCH=1             #整批训练一次
BATCH_SIZE=50       
TIME_STEP=28        # rnn 时间步／图片高度
INPUT_SIZE=28       # rnn 每步输入值/图片每行像素
LR=0.001            # learning rate
DOWNLOAD_MNIST=False


#Mnist手写数字
train_data=torchvision.datasets.MNIST(
	root='/Users/momo/Desktop/pytorch_exercise/mnist/',        #数据存储的位置
	train=True,                                               #this is the training data
	transform=torchvision.transforms.ToTensor(),              #转换PIL.Image or numpy.ndaaray成torch.FloatTensor(C*H*W)，训练的时候normalize成[0.0,1.0]区间
	download=DOWNLOAD_MNIST,                                  #如果已经下载了就不需要下载数据了
	)

print (train_data.train_data.size())        # (6000,28,28)
print (train_data.train_labels.size())      # (6000)


plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
plt.title('%i' %train_data.train_labels[0])
plt.show()

train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

test_data=torchvision.datasets.MNIST (
	root='/Users/momo/Desktop/pytorch_exercise/mnist/',train=False,)

#为了节约时间，我们只测试前2000个
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)

test_x = Variable(torch.unsqueeze(test_data.test_data,dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255
test_y = test_data.test_labels[:2000]


class RNN(nn.Module):
	def __init__(self):
		super(RNN,self).__init__()

		self.rnn=nn.LSTM(         # LSTM效果要比nn.RNN要好
			input_size=28,        # 图片每行的数据像素点
			hidden_size=64,       # rnn hidden umit
			num_layers=1,         # 有几层RNN layers
			batch_first=True,     # input & output 会是以batch size为第一维度的特征集，e.g (batch,time_step,input_size)
		)

		self.out=nn.Linear(64,10) #输出层

	def forward(self,x):
		# x shape(batch,time_step,input_size)
		# r_out shape (batch, time_step,output_size)
		# h_n shape(n_layers, batch,hidden_size) LSTM有两个hidden states,h_n是分线，h_c是主线
		# h_c shape(n_layers,batch,hidden_size)
		r_out,(h_n,h_c)=self.rnn(x,None)    #None表示hidden state会用全0的state

		#选取最后一个时间点的r_out输出，r_out[:,-1,:]的值也是h_n的值
		out=self.out(r_out[:,-1,:])      
		return out                          # size 为（64，10）


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
	for step,(x,y) in enumerate(train_loader):
		b_x=Variable(x.view(-1,28,28))     #reshape x to (batch,time_step,input_size)
		b_y=Variable(y)

		output=rnn(b_x)
		loss=loss_function(output,b_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if step % 100 == 0:
			test_output = rnn(test_x.view(-1,28,28))
			pred_y = torch.max(test_output,1)[1].data.squeeze()       #torch.max返回两个结果，第一个是最大值，第二个是对应的索引值；第二个参数为1代表按行取最大值并返回对应的索引值
			accuracy = sum(pred_y == test_y)/test_y.size(0)
			print('Epoch:', epoch, '|Step:', step,'|train loss:%.4f'%loss.data[0], '|test accuracy:%.4f'%accuracy)


test_output=rnn(test_x[:10].view(-1,28,28))
pred_y=torch.max(test_output,1)[1].data.squeeze()
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')











