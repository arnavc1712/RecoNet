import torch
import torch.optim as optim
import opts
from torch.utils.data import DataLoader
from utils.utils import *
from data.data_loader import RecDataset
# from model.transformer.recModel import Encoder
import torch.optim as optim
from losses import hinge_loss, adaptive_hinge_loss, binary_cross_entropy
from model.transformer.Optim import ScheduledOptim
import os

from model.RNNRec.model import Model



def train(loader,optimizer,model,opt,dataset):
	# model.train()
	# epoch_loss = 0.0
	# ix_to_item = loader.dataset.get_ix_to_item()
	# item_to_ix = loader.dataset.get_item_to_ix()
	model.train()
	for epoch in range(opt['epochs']):
		for i,(user, seq, pos, neg,seq_len) in enumerate(loader):

			torch.cuda.synchronize()		
			optimizer.zero_grad()
			user = user.cuda()
			seq = seq.cuda()
			pos = pos.cuda()
			neg = neg.cuda()
			user_rep = model.get_user_rep(user,seq,seq_len).contiguous()
			# print(user_rep.shape)
			pos = pos.view(seq.shape[0]*opt['max_seq_len'])
			neg = neg.view(seq.shape[0]*opt['max_seq_len'])
			pos_logits = model(user_rep,pos)
			neg_logits = model(user_rep,neg)

			istarget = pos.ne(0).type(torch.float).view(seq.shape[0]*opt['max_seq_len'])
			# print(pos)
			# print(istarget.shape)
			print(torch.sum((-torch.log(torch.sigmoid(pos_logits) + 1e-24)*istarget)))
			print(torch.sum((-torch.log(torch.sigmoid(neg_logits) + 1e-24)*istarget)))
			loss = torch.sum((-torch.log(torch.sigmoid(pos_logits) + 1e-24)*istarget) - (torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)*istarget))
			loss = loss/torch.sum(istarget)

			# print(sum(istarget))

			
			print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
			torch.cuda.synchronize()

			loss.backward()
			torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1)
			optimizer.step()


		if epoch%20==0:
			t_test = evaluateRNN(model.eval(), (dataset.user_train, dataset.user_valid, dataset.user_test, dataset.num_users, dataset.num_items), opt)
			
			print(f"NCDG : {t_test[0]}\t HIT@10 : {t_test[1]}")

			# t_valid = evaluate_valid(model, dataset, args, sess)





def main(opt):
	dataset = RecDataset('train',opt,model="rnn")
	dataloader = DataLoader(dataset,batch_size=opt['batch_size'],shuffle=True)

	# model = Encoder(seq_len=opt['max_seq_len'],
 #            dim_item=opt["dim_item"],
 #            dim_user=opt["dim_item"],
 #            n_users=dataset.get_num_users(),
 #            n_items=dataset.get_num_items(),
 #            n_layers=opt["num_layer"],
 #            n_head=opt["num_head"],
 #            d_k=opt["dim_model"]//opt["num_head"],
 #            d_v=opt["dim_model"]//opt["num_head"],
 #            d_model=opt["dim_model"],
 #            d_inner=opt["dim_inner"],
 #            input_dropout_p=opt["input_dropout_p"],
 #            dropout=opt["dropout"])

	# model = model.cuda()

	model = Model(num_users=dataset.get_num_users(),
				  num_items=dataset.get_num_items(),
				  opt=opt)

	optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                          betas=(0.9, 0.98), eps=1e-09,weight_decay=0.001)
	# optimizer = ScheduledOptim(optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
	# betas=(0.9, 0.98), eps=1e-09), opt["dim_model"], opt["warm_up_steps"])
	train(dataloader,optimizer,model,opt,dataset)


if __name__ == "__main__":
	print("Running")
	opt = opts.parse_opt()
	opt = vars(opt)
	main(opt)