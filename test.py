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
import os

def test(loader,model,opt):
	model.eval()
	ix_to_item = loader.dataset.get_ix_to_item()
	item_to_ix = loader.dataset.get_item_to_ix()
	total = len(loader)
	print("total = ", total)
	true_pred = 0
	for i,(input_ids,target_ids,user_ids) in enumerate(loader):

			src_pos = pos_generate(input_ids)
			user_rep,attns = model.user_representation(input_ids,src_pos,user_ids,return_attns=True,include_user=opt['include_user'])


			#pred, true = show_predictions(input_ids,target_ids,user_rep[:,-1:,:],model,ix_to_item,attns,opt)
			show_predictions(input_ids,target_ids,user_rep[:,-1:,:],model,ix_to_item,attns,opt)

			user_rep_temp = user_rep[:,-1,:]
			random_id = random.randint(0, len(input_ids)-1)

    		#print("Sequence")
			target_ids=target_ids.cpu()
			input_ids = input_ids.cpu()
			list(map(lambda x:ix_to_item[x],input_ids[random_id].numpy().flatten()))
    		# print("\n")

			target = target_ids[random_id][-1:]
			user_rep_temp = user_rep_temp[random_id]
			item_ids = np.array(list(ix_to_item.keys())).reshape(-1,1)
			item_ids = torch.from_numpy(item_ids).type(torch.LongTensor)
			size = (len(item_ids),) + user_rep_temp.size()
			out = model(user_rep_temp.expand(*size),item_ids)
			preds = out.detach().cpu().numpy().flatten()
			most_probable_10 = preds.argsort()[-opt["num_recs"]:][::-1]
			#most_prob_10_items = list(map(lambda x:ix_to_item[x],most_probable_10))
			g_t = target.detach().numpy().flatten()[0]
			
			pred = most_probable_10
			true = g_t

			print("prediction and true label match", pred , true)
			if true in pred:
				true_pred +=1
	print("Model Accuracy = ", true_pred/total)
			




def main(opt):
	dataset = RecDataset('test',opt)
	dataloader = DataLoader(dataset,batch_size=128,shuffle=True,collate_fn=rec_collate_fn)
	print(dataset.get_num_items())
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
	model.load_state_dict(torch.load('save/recnet_0.pth'))
	model.eval()
	test(dataloader,model,opt)


if __name__ == "__main__":
	print("Running")
	opt = opts.parse_opt()
	opt = vars(opt)
	main(opt)