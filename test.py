import torch
import torch.optim as optim
import opts
from torch.utils.data import DataLoader
from utils.utils import *
from data.data_loader import RecDataset,rec_collate_fn
from model.transformer.recModel import Encoder
import torch.optim as optim
from losses import hinge_loss, adaptive_hinge_loss,binary_cross_entropy
from model.transformer.Optim import ScheduledOptim
from sampling import sample_items
import os

def test(loader,model,opt):
	model.eval()
	ix_to_item = loader.dataset.get_ix_to_item()
	item_to_ix = loader.dataset.get_item_to_ix()
	num_items = len(item_to_ix)
	total = len(loader)
	print("total = ", total)
	true_pred = 0
	total = 0
	for i,(input_ids,target_ids,user_ids) in enumerate(loader):
		neg_items = sample_items(num_items=num_items,shape=(100))
		neg_items = torch.tensor(neg_items).unsqueeze(0).repeat(input_ids.shape[0],1)
		print(neg_items.shape)
		total += input_ids.shape[0]
		src_pos = pos_generate(input_ids)
		input_ids = input_ids.cuda()
		target_ids = target_ids.cuda()
		src_pos = src_pos.cuda()
		user_ids = user_ids.cuda()
		user_rep,attns = model.user_representation(input_ids,src_pos,user_ids,return_attns=True,include_user=opt['include_user'])

		tgt_indices_bf_padding = []
		inpt_indices_bf_padding = []
		for i in range(len(target_ids)):
			# print(input_ids[i])
			if len((input_ids[i]==0).nonzero().flatten()):
				inpt_indices_bf_padding.append((target_ids[i]==0).nonzero().flatten()[0].item()-1)
				tgt_indices_bf_padding.append((target_ids[i]==0).nonzero().flatten()[0].item()-1)
				# print(input_ids[i])
				# print(inpt_indices_bf_padding[i])
			else:
				inpt_indices_bf_padding.append(len(target_ids[i])-1)
				tgt_indices_bf_padding.append(len(target_ids[i])-1)
			# print((input_ids[i]==0).nonzero().flatten())

		# print(input_ids.shape)
		# print(input_ids[0])
		# print(target_ids[0])
		# print(inpt_indices_bf_padding[0])
		# print(tgt_indices_bf_padding[0])
		# break
		inpt_indices_bf_padding = np.array(inpt_indices_bf_padding)
		tgt_indices_bf_padding = np.array(tgt_indices_bf_padding)
		# print(user_rep.shape)
		user_rep_temp = np.expand_dims(user_rep.detach().cpu().numpy()[np.array(list(range(len(input_ids)))),inpt_indices_bf_padding,:],axis=1)
		user_rep_temp = torch.from_numpy(user_rep_temp).type(torch.FloatTensor)
		
		targets = target_ids.detach().cpu().numpy()[np.array(list(range(len(target_ids)))),tgt_indices_bf_padding]
		targets = torch.from_numpy(targets).type(torch.LongTensor)

		print("TARGETS")
		print(targets.shape)
		btch_sz,length,dim = user_rep_temp.size()
		item_ids = np.array(list(ix_to_item.keys())).reshape(-1,1)
		item_ids = torch.from_numpy(item_ids).type(torch.LongTensor).unsqueeze(0).repeat(btch_sz,1,1).cuda()
		print("Item Ids")
		print(item_ids.shape)
		# size = (len(item_ids),) + user_rep_temp.size()
		user_rep_temp = user_rep_temp.unsqueeze(1).repeat(1,item_ids.shape[1],1,1).cuda()
		# print(user_rep_temp.size())
		out = model(user_rep_temp,item_ids)
		# print(out.shape)
		preds = []
		for scores in out:
			preds.append(scores.detach().cpu().numpy().flatten())
		# print(preds[0:10])

		for i,predictions in enumerate(preds):
			most_probable_10 = predictions.argsort()[-opt['num_recs']:][::-1]
			most_prob_10_items = list(map(lambda x:ix_to_item[x],most_probable_10))

			g_t = targets[i].detach().cpu().numpy().flatten()[0]

			if g_t in most_probable_10:
				print(most_prob_10_items)
				print(ix_to_item[g_t])
				true_pred+=1


	print(f"Total: {total} ")
	print(f"RIght predictions: {true_pred}")
	print("Model Accuracy = ", true_pred/total)
			




def main(opt):
	dataset = RecDataset('test',opt)
	dataloader = DataLoader(dataset,batch_size=128,shuffle=True,collate_fn=rec_collate_fn)
	print(f"Number of Items: {dataset.get_num_items()}")
	model = Encoder(seq_len=opt['max_seq_len'],
			n_users=dataset.get_num_users(),
			dim_user=opt["dim_item"],
            dim_item=opt["dim_item"],
            n_items=dataset.get_num_items(),
            n_layers=opt["num_layer"],
            n_head=opt["num_head"],
            d_k=opt["dim_model"]//opt["num_head"],
            d_v=opt["dim_model"]//opt["num_head"],
            d_model=opt["dim_model"],
            d_inner=opt["dim_inner"],
            input_dropout_p=opt["input_dropout_p"],
            dropout=opt["dropout"])
	model.load_state_dict(torch.load(opt['load_checkpoint']))
	model.cuda()
	model.eval()
	test(dataloader,model,opt)


if __name__ == "__main__":
	print("Running")
	opt = opts.parse_opt()
	opt = vars(opt)
	main(opt)