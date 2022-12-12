import torch
import torch.nn as nn
import torch.nn.functional as F

class   esc50CNN(nn.Module):
	"""
	Network / Convolutional newral network for esc50 task
	"""
	def __init__(self, n_out):
		"""
		Construct CNN

		Parameters
		----------
		n_out	: int
			dimension of output
		"""
		super(esc50CNN, self).__init__()

		self.conv1 = nn.Conv2d(3, 8, 5, padding=2, stride=1)	# 3 * 64 * 64 -> 8 * 64 * 64
		self.pool1 = nn.MaxPool2d(2, 2)	# 8 * 64 * 64 -> 8 * 32 * 32
		self.dropout_1 = nn.Dropout(0.2)
		self.conv2 = nn.Conv2d(8, 16, 5, padding=2, stride=1)	# 8 * 32 * 32 -> 16 * 32 * 32
		self.pool2 = nn.MaxPool2d(2, 2)	# 16 * 32 * 32 -> 16 * 16 * 16
		self.dropout_2 = nn.Dropout(0.2)
		self.conv3 = nn.Conv2d(16, 100, 5, padding=2, stride=1)	# 16 * 16 * 16 -> 100 * 16 * 16
		self.pool3 = nn.MaxPool2d(2, 2)	# 100 * 16 * 16 -> 100 * 8 * 8    
		self.dropout_3 = nn.Dropout(0.2)    
		self.conv4 = nn.Conv2d(100, 200, 5, padding=2, stride=1)	# 100 * 8 * 8 -> 200 * 8 * 8
		self.pool4 = nn.MaxPool2d(2, 2)	# 200 * 8 * 8 -> 200 * 4 * 4    
		self.dropout_4 = nn.Dropout(0.2)    
		self.fc1 = nn.Linear(200 * 4 * 4, 17424)	# 3200(=200*4*4) -> 17424
		self.fc2 = nn.Linear(17424, 1024)
		self.fc3 = nn.Linear(1024, 500)
		self.fc4 = nn.Linear(500, n_out)


	def forward(self, x):
		"""
		Calculate forward propagation

		Parameters
		----------
		x	: torch.Tensor
			input to the network
			batchsize * channel * height * width

		Returns
		-------
		y	: torch.Tensor
			output from the network
			batchsize * outout_dimension
		"""
		h = self.pool1(F.relu(self.conv1(x)))
		h = self.dropout_1(h)
		h = self.pool2(F.relu(self.conv2(h)))
		h = self.dropout_2(h)
		h = self.pool3(F.relu(self.conv3(h)))
		h = self.dropout_3(h)
		h = self.pool4(F.relu(self.conv4(h)))
		h = self.dropout_4(h)
		h = h.view(-1, 200 * 4 * 4)
		h = self.fc1(h)
		h = self.fc2(h)
		h = self.fc3(h)
		y = self.fc4(h)
		return y
