# author :momo
#time :2018.09.20
# Auto-Encoder 
#神经网络也可以进行非监督学习，只需要训练数据，不需要标签数据，自编码就是这样一种形式，自编码能够自动分类数据，而且也能潜逃在半监督学习上。
#用少量的有标签样本和大量的无标签样本学习。

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.datasets as dsets    #数据库模块
import torchvision.transforms as transforms
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
import numpy as np 


torch.manual_seed(1)

EPOCH=10
BATCH_SIZE=64
LR=0.005
DOWNLOAD_MNIST=False
N_TEST_IMG=5   # 显示5张图片看效果


# mnist digits datasets
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

# AutoEncoder 包括encoder和decoder，压缩和解压，压缩后得到提取的特征值，再从压缩的特征值解压成原图片
class AutoEncoder(nn.Module):
	def __init__(self):
		super(AutoEncoder,self).__init__()

		#压缩
		self.encoder=nn.Sequential(
			nn.Linear(28*28,128),
			nn.Tanh(),
			nn.Linear(128,64),
			nn.Tanh(),
			nn.Linear(64,12),
			nn.Tanh(),
			nn.Linear(12,3),      # 压缩成三个特征，进行3d图像可视化
		)

		#解压
		self.decoder=nn.Sequential(
			nn.Linear(3,12),
			nn.Tanh(),
			nn.Linear(12,64),
			nn.Tanh(),
			nn.Linear(64,128),
			nn.Tanh(),
			nn.Linear(128,28*28),
			nn.Sigmoid(),         #激励函数让输出值在(0,1)
		)

	def forward(self,x):
		encoder=self.encoder(x)
		decoder=self.decoder(encoder)
		return encoder,decoder

autoencoder=AutoEncoder()
print (autoencoder)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_function = nn.MSELoss()


#initialize figure
f,a=plt.subplots(2,N_TEST_IMG,figsize=(5,2))
plt.ion()

view_data=train_data.train_data[:N_TEST_IMG].view(-1,28*28).type(torch.FloatTensor)/255.
for i in range(N_TEST_IMG):
	a[0][i].imshow(np.reshape(Variable(view_data).data.numpy()[i],(28,28)),cmap='gray');
	a[0][i].set_xticks(());
	a[0][i].set_yticks(())

for epoch in range(EPOCH):
	for step,(x,y) in enumerate(train_loader):
		b_x=Variable(x.view(-1,28*28))     #reshape x to (batch,time_step,input_size) (batch, 28*28)
		b_y=Variable(x.view(-1,28*28))     #(batch,time_step,input_size)   (batch,28*28)
		b_lable=Variable(y)

		encoder,decoder=autoencoder(b_x)

		loss=loss_function(decoder,b_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if step % 100 == 0:
			print('Epoch:',epoch,'|train loss: %.4f' % loss.data.numpy())

			# plotting decoded image(second row)
			_, decoded_data=autoencoder(Variable(view_data))
			for i in range(N_TEST_IMG):
				a[1][i].clear()
				a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i],(28,28)),cmap='gray')
				a[1][i].set_xticks(());
				a[1][i].set_yticks(())
			plt.draw();
			plt.pause(0.1)
plt.ioff()
plt.show()



# visualize in 3D plot
view_data = Variable(train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.)
encoded_data, _ = autoencoder(view_data)    # 提取压缩的特征值
fig = plt.figure(2)
ax = Axes3D(fig)    # 3D 图
# x, y, z 的数据值
X = encoded_data.data[:, 0].numpy()
Y = encoded_data.data[:, 1].numpy()
Z = encoded_data.data[:, 2].numpy()

values = train_data.train_labels[:200].numpy()  # 标签值

for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9))    # 上色

    ax.text(x, y, z, s, backgroundcolor=c)  # 标位子
    
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()




		

