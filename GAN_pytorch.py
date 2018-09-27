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
np.random.seed(1)

#hyper parameters
BATCH_SIZE=64
LR_G=0.0001
LR_D=0.0001
N_IDEAS=5             #think of this as number of ideas for generating an art work(Generator)
ART_COMPONENTS=15      # it could be total point G can draw in the canvas
PAINT_POINTS=np.vstack([np.linspace(-1,1,ART_COMPONENTS) for _ in range(BATCH_SIZE)]) #纵轴连接，(64,15)


def artist_works():      #painting form the famous artist (real target)
	a=np.random.uniform(1,2,size=BATCH_SIZE)[:,np.newaxis]     #a为系数，有64个取值
	paintings=a*np.power(PAINT_POINTS,2)+(a-1)       #(64,15)
	paintings=torch.from_numpy(paintings).float()
	return Variable(paintings)

plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3,label='upper bound')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3,label='lower bound')
plt.legend(loc='upper right')
plt.show()

G=nn.Sequential(
	nn.Linear(N_IDEAS,128),
	nn.ReLU(),
	nn.Linear(128,ART_COMPONENTS),
)

D=nn.Sequential(
	nn.Linear(ART_COMPONENTS,128),
	nn.ReLU(),
	nn.Linear(128,1),
	nn.Sigmoid(),
)

opt_D = torch.optim.Adam(D.parameters(),lr=LR_D)
opt_G = torch.optim.Adam(D.parameters(),lr=LR_G)

plt.ion()


for step in range (1000):
	artist_paintings=artist_works()                        #(64,15)
	G_ideas=Variable(torch.randn(BATCH_SIZE,N_IDEAS))      #(64,5)
	G_paintings=G(G_ideas)                                 #(64,15)

	prob_artist0=D(artist_paintings)
	prob_artist1=D(G_paintings)

	D_loss=-torch.mean(torch.log(prob_artist0)+torch.log(1-prob_artist1))
	G_loss=torch.mean(torch.log(1-prob_artist1))

	opt_D.zero_grad()
	D_loss.backward(retain_variables=True)         # retain_variables 这个参数是为了再次使用计算图纸
	opt_D.step()

	opt_G.zero_grad()
	G_loss.backward(retain_variables=True)
	opt_G.step()

	if step % 50 == 0:
		plt.cla()
		plt.plot(PAINT_POINTS[0],G_paintings.data.numpy()[0],c='#4ad631',lw=3,label='Generated painting',)
		plt.plot(PAINT_POINTS[0],2 * np.power(PAINT_POINTS[0], 2) + 1,c='#74BCFF',lw=3,label='upper bound',)
		plt.plot(PAINT_POINTS[0],1 * np.power(PAINT_POINTS[0], 2) + 0,c='#FF9359',lw=3,label='lower bound',)
		plt.text(-.5,2.3,'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size':15})
		plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 15})
		plt.ylim((0,3))
		plt.legend(loc='upper right', fontsize=12)
		plt.draw()
		plt.pause(0.01)

plt.ioff()
plt.show()

 




















