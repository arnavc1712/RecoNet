import torch
import torch.optim as optim
import opts
from torch.utils.data import DataLoader
from utils.utils import *
from data.data_loader import RecDataset
import torch.optim as optim
from losses import hinge_loss, adaptive_hinge_loss, binary_cross_entropy
from model.transformer.Optim import ScheduledOptim
import os
import pickle
from torch.utils.tensorboard import SummaryWriter

from model.RNNRec.model import Model

writer = SummaryWriter('runs/plot_item_embeddings')
with open("./data/movie_map.pkl","rb") as f:
	movie_map = pickle.load(f)

movie_map[0]="<PAD>"
opt = opts.parse_opt()
opt = vars(opt)

dataset = RecDataset('train',opt,model="rnn")

model = Model(num_users=dataset.get_num_users(),
				  num_items=dataset.get_num_items(),
				  opt=opt)

model.load_state_dict(torch.load("./save/model_rnn_40.pth"))

print(model.item_emb.weight.shape)

writer.add_embedding(model.item_emb.weight,metadata=list(movie_map.values()))