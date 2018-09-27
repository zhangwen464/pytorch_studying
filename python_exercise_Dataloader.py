# author :momo
#time :2018.09.20
# dataloader


import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
from torch.autograd import Variable
import torch.utils.data as Data 

torch.manual_seed(1)

BATCH_SIZE=5

x=torch.linspace(1,10,10)
y=torch.linspace(10,1,10)


#转换成torch可以识别的Dataset
torch_dataset=Data.TensorDataset(data_tensor=x,target_tensor=y)

#把dataset放入Dataloader
loader=Data.DataLoader(
	dataset=torch_dataset,        #torch TensorDataset format
	batch_size=BATCH_SIZE,        # mini batch size
	shuffle=True,                 # 打乱数据
	num_workers=2,)                # 多线程来读数据

for epoch in range(3):
	for step, (batch_x,batch_y) in enumerate(loader):

		print ('Epoch:',epoch,'|Step:',step,'|batch x:',batch_x.numpy(),'|batch y',batch_y.numpy())







