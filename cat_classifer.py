import numpy as np 
import pandas as pd 
import torch
import glob
import matplotlib.pyplot as plt
import torchvision
import cv2
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchsummary import summary
from random import shuffle
import os


class Simple_Net(nn.Module):
    def __init__(self):
        super(Simple_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(50, 100, 7)
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(100 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #batch size is hardcoded here
		#x = x.view(16, 100 * 12 * 12)

        x = x.reshape(16, 100 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

class Simpler_Net(nn.Module):
    def __init__(self):
        super(Simpler_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 25, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(25, 50, 7)
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(50 * 12 * 12, 30)
        self.fc2 = nn.Linear(30, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #batch size is hardcoded here
        #x = x.view(16, 100 * 12 * 12)

        x = x.reshape(16, 50 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class custom_dataset(data.Dataset):
	def __init__(self, img_addrs, labels):
		super(custom_dataset, self).__init__()
		self.img_addrs = img_addrs
		self.labels  = labels

	def __len__(self):
		return len(self.img_addrs)

	def __getitem__(self, index):
		img_addr = self.img_addrs[index]
		img_label = self.labels[index]

		img = cv2.imread(img_addr)
		img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
		img_t = torch.Tensor(img)
		label_t = torch.Tensor(img_label)

		return img_t, label_t

def train(train_img_path, test_img_path, pths_path, batch_size, lr, epoch_iter, save_interval, eval_interval):

	num_workers = 1
	shuffle_data = True  
	addrs = glob.glob(train_img_path)
	labels = [ [1, 0] if 'cat' in addr else [0, 1] for addr in addrs]  # 1 = Cat, 0 = Dog

    # to shuffle data
	if shuffle_data:
		c = list(zip(addrs, labels))
		shuffle(c)
		addrs, labels = zip(*c)

	train_end = 0.2
	test_end = 0.3
        
    # Divide the hata into 80% train, and 20% test
	train_addrs = addrs[0:int(train_end*len(addrs))]
	train_labels = labels[0:int(train_end*len(labels))]
    
    #val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
    #val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
    
	test_addrs = addrs[int(train_end*len(addrs)):int(test_end*len(addrs))]
	test_labels = labels[int(train_end*len(labels)):int(test_end*len(addrs))]

	train_data = custom_dataset(train_addrs, train_labels)
    #val_data = custom_dataset(val_addrs, val_labels)
	test_data = custom_dataset(test_addrs, test_labels)

	train_loader = data.DataLoader(train_data, batch_size=batch_size, \
                                   shuffle=shuffle_data, num_workers=num_workers, drop_last=True)

	test_loader = data.DataLoader(test_data, batch_size=batch_size, \
 	                              	   shuffle=shuffle_data, num_workers=num_workers, drop_last=True)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print ('Selected Device')
	print (device)

	#model = Simple_Net()
	model = Simpler_Net()
	model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25, 100], gamma=.1)
	epochs = []
	accuracies = []
	train_losses = []

	for epoch in range(epoch_iter):
		print ('Beginning Epoch:')
		print (epoch)

		#Training code
		model.train()
		epoch_loss = 0
		for i, (img, label) in enumerate(train_loader):
			# print ('Batch')
			img, label = img.to(device), label.to(device)
			img_p = img.permute(0,3,1,2)
			output = model(img_p)
			# print ('Output')
			# print (output)
			# print ('Label')
			# print (label)

			label_0 = label[:, 0]
			label_0 = label_0.type(torch.LongTensor)
			label_0 = label_0.to(device)

			loss = criterion(output, label_0)
			optimizer.zero_grad()
			loss.backward()

			#print ('Printing grads')
			total_grad = 0
			for j, p in enumerate(model.parameters()):
				#print (p.grad.norm())
				total_grad += p.grad.norm()

			if (total_grad == 0):
				print ('GRAD IS ZERO!')

			optimizer.step()

			epoch_loss += loss.item()

		scheduler.step()

		print ('TRAIN LOSS:')
		print (epoch_loss)
		train_losses.append(epoch_loss)

		if ((epoch + 1) % eval_interval) == 0:
			#eval code
			print ('Starting Eval')
			model.eval()
			total_correct = 0
			total_samples = 0
			for i, (img, label) in enumerate(test_loader):
				# print ('Eval Batch')
				# print (i)
				with torch.no_grad():
					img, label = img.to(device), label.to(device)
					img_p = img.permute(0,3,1,2)
					output = model(img_p)
					output_b = (output[:, 1] > 0.5).float()

					label_0 = label[:, 0].float()

					total_correct += (batch_size-torch.sum(torch.abs(output_b - label_0)))
					total_samples += batch_size

			test_accuracy = (total_correct/total_samples)

			print ('TEST ACCURACY:')
			print (test_accuracy)

			epochs.append(epoch+1)
			accuracies.append(test_accuracy)

		if ((epoch + 1) % save_interval) == 0:
			state_dict = model.state_dict()
			torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))

if __name__ == '__main__':
	train_img_path = './data/train1/*.jpg'
	test_img_path = './data/train1/*.jpg'
	#train_img_path = '../data/c_d/*.jpg'
	#test_img_path = '../data/c_d/*.jpg'
	pths_path = './pths'
	batch_size = 16
	lr = 1e-5
	epoch_iter = 300
	save_interval = 5
	eval_interval = 5

	train(train_img_path, test_img_path, pths_path, batch_size, lr, epoch_iter, save_interval, eval_interval)







