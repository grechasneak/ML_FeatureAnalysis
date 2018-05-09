from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import pickle
from sklearn.cross_validation import train_test_split 
import pandas as pd
import numpy as np
import torch.utils.data as data_utils
import gc

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 28 * 2, stride = 2)
		self.conv2 = nn.Conv2d(6, 6, 14 * 2)
		self.conv3 = nn.Conv2d(6, 16, 10 * 2)
		self.conv4 = nn.Conv2d(16, 16, 5 * 2)
		self.fc1 = nn.Linear(11648, 5000)
		self.bat_norm = nn.BatchNorm1d(5000)
		self.fc2 = nn.Linear(5000, 1000)
		self.fc3 = nn.Linear(1000, 1)


	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = x.view(-1, self.num_flat_features(x))	
		x = F.relu(self.fc1(x))
		x = self.bat_norm(x)
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]	 # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

def eval_net(dataloader):
	correct = 0
	total = 0
	total_loss = 0
	net.eval() 
	criterion = nn.MSELoss()
	for data in dataloader:
		images, bias = data
		images, bias = Variable(images).cuda(), Variable(bias).cuda()
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += bias.size(0)
		correct += (predicted == bias.data).sum()
		loss = criterion(outputs, bias)
		total_loss += loss.data[0]
	net.train() # Why would I do this?
	return total_loss / total, correct / total

    
#Object saver utility function
def save_object(python_object, filename):
	with open(filename+".pickle","wb") as f:
		pickle.dump(python_object,f)


class SenData():
	def __init__(self, object, transform = None):
		self.df = object
		self.transform = transform
		
	def __len__(self):
		return len(self.df.index)
		
	def __getitem__(self, idx):
		sen = self.df.s.iloc[idx]
		sen = sen.reshape(-1, 12 * 44)
		sen = torch.from_numpy(sen).type(dtype)
		sen = sen.view((1,-1, 12 * 44))

		bias = self.df.bias.iloc[idx]
		sample = {'sen': sen, 'bias': bias}
		
		if self.transform:
			sample = self.transform(sample)
		return sample
		
	
if __name__ == "__main__":
	torch.cuda.set_device(1)
	BATCH_SIZE = 64 #mini_batch size
	MAX_EPOCH = 10 #maximum epoch to train
	
	dtype = torch.FloatTensor 
		 
	 
	data_frame = pickle.load( open( "filename.pickle", "rb" ) )
	train_set, test_set = train_test_split(data_frame, test_size=0.2, random_state=42)
	
	train_data = SenData(train_set)
	test_data = SenData(test_set)
	
	
	
#	dataset = data_utils.TensorDataset(train_set['sen'], train_set['bias'])
											
	trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,
											    shuffle=True, num_workers=2)
								   
	testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,
											  shuffle=False, num_workers=2)


	print('Building model...')
	net = Net().cuda()
	net.train() 

	
	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters(), lr = 0.001)
	
	train_losses = []
	train_accuracies = []
	test_losses = []
	test_accuracies = []	
	
	print('Start training...')
	for epoch in range(MAX_EPOCH):	# loop over the dataset multiple times

		running_loss = 0.0
		for i, case in enumerate(trainloader):
			# get the inputs

			sen, bias = case['sen'], case['bias']
			 
			# wrap them in Variable
			inputs  = Variable(sen).cuda()
			bias = Variable(bias).type(dtype).cuda()
			
			
			# zero the parameter gradients
			optimizer.zero_grad()
			# forward + backward + optimize
			outputs = net(inputs)
			
			loss = criterion(outputs, bias)
			loss.backward()
			optimizer.step()
			# print statistics
			print(loss.data[0])
			

		print('	   Finish training this EPOCH, start evaluating...')
		train_loss, train_acc = eval_net(trainloader)
		test_loss, test_acc = eval_net(testloader)
		
		train_losses.append(train_loss)
		train_accuracies.append(train_acc)
		test_losses.append(test_loss)
		test_accuracies.append(test_acc)
		
		# print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
			  # (epoch+1, train_loss, train_acc, test_loss, test_acc))
			  
	# save_object(train_accuracies, 'train_acc')
	# save_object(test_accuracies, 'test_acc')
	# save_object(train_losses, 'train_loss')
	# save_object(test_losses, 'test_loss')
		 
	# print('Finished Training')
	# print('Saving model...')
	# #torch.save(net.state_dict(), 'mytraining_5E.pth')
	
	# # #load in pretrained weights
	# # pretrained_dict = torch.load('mytraining_5E.pth')
	
	# # #get weights of current model
	# # model_dict = net.state_dict()
	
	# # # filter out unnecessary keys
	# # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	# # # overwrite entries in the existing network
	# # model_dict.update(pretrained_dict) 
	# # # load the new network state
	# # net.load_state_dict(model_dict)
	