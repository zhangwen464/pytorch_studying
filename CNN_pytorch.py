# author :momo
#time :2018.09.20
# CNN

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision     #数据库模块
import torch.nn.functional as F 
import matplotlib.pyplot as plt 


torch.manual_seed(1)

EPOCH=1
BATCHZ_SIZE=50
LR=0.001
DOWNLOAD_MNIST=False      #如果你已经下载好了mnist数据，则值设置为False

#Mnist手写数字
train_data=torchvision.datasets.MNIST(
	root='/Users/momo/Desktop/pytorch_exercise/mnist/',        #数据存储的位置
	train=True,                                               #this is the training data
	transform=torchvision.transforms.ToTensor(),              #转换PIL.Image or numpy.ndaaray成torch.FloatTensor(C*H*W)，训练的时候normalize成[0.0,1.0]区间
	download=DOWNLOAD_MNIST,                                  #如果已经下载了就不需要下载数据了
	)
print (train_data.train_data.size())      # (6000,28,28) 
print (train_data.train_labels.size())    # (6000)


plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
plt.title('%i' %train_data.train_labels[0])
plt.show()

#批训练 50 samples,1 channel, 28*28 (50,1,28,28)
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCHZ_SIZE,shuffle=True)

test_data=torchvision.datasets.MNIST (
	root='/Users/momo/Desktop/pytorch_exercise/mnist/',train=False,)

#为了节约时间，我们只测试前2000个
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_x = Variable(torch.unsqueeze(test_data.test_data,dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255
test_y = test_data.test_labels[:2000]

#CNN模型

class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		self.conv1 = nn.Sequential(        # input shape(1,28,28)
			nn.Conv2d(
				in_channels=1,             # input height 
				out_channels=16,           # n_filters
				kernel_size=5,             # filter size
				stride=1,                  # filter movement/step
				padding=2,                 # 想要conv2d 输出的图片长宽不变，padding=(kernel_size-1)/2,当sttide=1
			),                             # output shape (16,28,28)
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)    # output shape (16,14,14)
		)
		self.conv2 = nn.Sequential(        # input shape(16,14,14)
			nn.Conv2d(
				in_channels=16,             # input height 
				out_channels=32,           # n_filters
				kernel_size=5,             # filter size
				stride=1,                  # filter movement/step
				padding=2,                 # 
			),                             # output shape (32,14,14)
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)    # output shape (32,7,7)
		)
		self.out=nn.Linear(32*7*7,10)


	def forward(self,x):
		x=self.conv1(x)
		x=self.conv2(x)
		x=x.view(x.size(0),-1)            #  input经过卷积层之后，输出是包括batchsize维度为4的tensor,即(batchsize,channels,height,width)，
			                              #  x.size(0)即为batchsize。展平多维的卷积图成 (batch_size, 32*7*7)
			                              #  view()函数的功能与reshape类似，用来转换size大小
			                              #  x = x.view(batchsize, -1)中batchsize指转换后有几行，而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数
		output=self.out(x)                # 然后输入线性层，输出维度为10
		return output

cnn=CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
	for step,(x,y) in enumerate(train_loader):
		b_x=Variable(x)
		b_y=Variable(y)

		output=cnn(b_x)
		loss=loss_function(output,b_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


		if step % 100 == 0:
			test_output = cnn(test_x)
			pred_y = torch.max(test_output,1)[1].data.squeeze()       #torch.max返回两个结果，第一个是最大值，第二个是对应的索引值；第二个参数为1代表按行取最大值并返回对应的索引值
			accuracy = sum(pred_y == test_y)/test_y.size(0)
			print('Epoch:', epoch, '|Step:', step,'|train loss:%.4f'%loss.data[0], '|test accuracy:%.4f'%accuracy)

test_output=cnn(test_x[:10])
pred_y=torch.max(test_output,1)[1].data.squeeze()
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')



 



















