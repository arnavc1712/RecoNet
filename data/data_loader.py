import os
import json
import torch
import random
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class RecDataset(Dataset):

	def get_ix_to_item(self):
		return self.ix_to_item

	def get_item_to_ix(self):
		return self.item_to_ix

	def get_num_items(self):
		return max(list(self.ix_to_item.keys()))+1

	def get_num_users(self):
		return self.num_users

	def __init__(self,mode,opt):
		super(RecDataset,self).__init__()

		self.mode = mode
		self.splits = {}
		self.splits['train'] = []
		self.splits['test'] = []

		# self.ix_to_item={}
		# self.item_to_ix = {}
		self.ix_to_item = pickle.load(open("ix_to_item.pkl","rb"))
		self.item_to_ix = pickle.load(open("item_to_ix.pkl","rb"))
		
		# with open(os.path.join(os.getcwd(),opt["item_data_path"]),encoding="ISO-8859-1") as f:
		# 	items = f.readlines()
		# for it in items:
		# 	split = str(it).split("::")[0:2]
		# 	# temp = [int(split[0]),split[1]]
		# 	self.ix_to_item[int(split[0])] = split[1]
		# 	self.item_to_ix[split[1]] = int(split[0])

		# print(ix_to_item)
		# item_df = pd.DataFrame(data=items_arr,columns=["id","item"])

		# self.sos_id = max(list(self.ix_to_item.keys())) + 1
		# self.ix_to_item[self.sos_id] = "<sos>"
		self.ix_to_item[0] = "<pad>"
		# self.item_to_ix["<sos>"] = self.sos_id
		self.item_to_ix["<pad>"] = 0

		path_to_data = os.path.join(os.getcwd(),opt["seq_data_path"])

		
		seq_data = np.array(pickle.load(open(path_to_data,"rb")))
		data = seq_data[:,1]

		self.user_ids = np.array(list(map(lambda x:int(x),seq_data[:,0].tolist()))).reshape(-1,1)
		unique_user_ids = np.unique(self.user_ids)
		print(f"Number of users: {max(unique_user_ids)}")
		print(f"Max number of items: {max(list(self.ix_to_item.keys()))+1}")
		self.num_users = max(unique_user_ids) + 1

		self.max_len = max([len(x) for x in data])

		self.sequences_data = []

		for i,seq in enumerate(data):
			# seq = np.array([self.item_to_ix["<sos>"]]+list(map(lambda x:int(x),seq)))
			seq = np.array(list(map(lambda x:int(x),seq)))
			self.sequences_data.append(seq)
			

		self.sequences_data = np.array(self.sequences_data)


		print(f"Total number of examples is {len(self.sequences_data)}")
		self.splits['train_seq'],self.splits['test_seq'],self.splits["train_user"],self.splits["test_user"] = train_test_split(self.sequences_data,self.user_ids,test_size=0.1,random_state=RANDOM_SEED)
		print(f"Total number of Training examples is {len(self.splits['train_seq'])}")
		print(f"Total number of Test examples is {len(self.splits['test_seq'])}")


	def __getitem__(self,ix):
		
		data = {}
		sequence = self.splits[f"{self.mode}_seq"][ix]
		# print(sequence.shape)
		data["inputs"] = torch.from_numpy(sequence[:-1]).type(torch.LongTensor)
		data["targets"] = torch.from_numpy(sequence[1:]).type(torch.LongTensor)
		data["user_ids"] = torch.from_numpy(self.splits[f"{self.mode}_user"][ix]).type(torch.LongTensor)
		return data



	def __len__(self):
		return len(self.splits[f"{self.mode}_seq"])


def rec_collate_fn(batch_lst):
	batch_lens = [_['inputs'].shape[0] for _ in batch_lst]
	max_seq_len = max(batch_lens) ## Finding maximum length in batch for videos
	input_ids = torch.zeros(len(batch_lst),max_seq_len)
	target_ids = torch.zeros(len(batch_lst),max_seq_len)
	user_ids = []
	for batch_id,batch_data in enumerate(batch_lst):
		input_ids[batch_id][(max_seq_len-batch_data["inputs"].shape[0]):] = batch_data["inputs"] ## PAdding zeros on the left
		target_ids[batch_id][(max_seq_len-batch_data["targets"].shape[0]):] = batch_data["targets"]
		user_ids.append(batch_data["user_ids"])

	return input_ids.type(torch.LongTensor),target_ids.type(torch.LongTensor),torch.tensor(user_ids).type(torch.LongTensor)





