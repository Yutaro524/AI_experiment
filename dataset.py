import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
class MyESCDataset(Dataset):
	def __init__(self, root=None, train=True, transform=None):
		self.root = root
		self.transform = transform
		mode = 'train' if train else 'test'
		# TODO 画像パスとそのラベルのセットをself.all_dataに入れる
		self.all_data = []
		for cls in range(50):
			cls_dir = os.path.join(self.root, str(cls))

			img_path_list = os.listdir(cls_dir) #glob,os,pathlibなどのモジュールを使うことが考えられる。その際、cls_dirを用いるとよい。
			cls_data = [[os.path.join(cls_dir, img_path), cls] for img_path in img_path_list]

			self.all_data.extend(cls_data)

	def __len__(self):
		#データセットの数を返す関数
		return len(self.all_data)
		
	def __getitem__(self, idx):
		# TODO 画像とラベルの読み取り
		#self.all_dataを用いる
		img = Image.open(self.all_data[idx][0]).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		label = self.all_data[idx][1]
		return [img, label]

class OGVCDataset(Dataset):
	def __init__(self, root=None, train=True, transform=None):
		self.root = root
		self.transform = transform
		mode = 'train' if train else 'test'

		classes = {
			"ACC":0, "ANG":1, "ANT":2, "DIS":3, "FEA":4, "JOY":5, "SAD":6, "SUR":7
			#データセットを参照して適切な辞書を作成する
		}
		# TODO 画像パスとそのラベルのセットをself.all_dataに入れる
		self.all_data = []
		for cls in classes.keys():
			filename = self.root + "/*/*/" + cls + "/*.png"
			cls_label = classes[cls]
			img_path_list = glob.glob(filename)

			cls_data = [[image, cls_label] for image in img_path_list]

			self.all_data.extend(cls_data)

	def __len__(self):
		#データセットの数を返す関数
		return len(self.all_data)
		
	def __getitem__(self, idx):
		# TODO 画像とラベルの読み取り
		#self.all_dataを用いる
		img = Image.open(self.all_data[idx][0]).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		label = self.all_data[idx][1]
		return [img, label]