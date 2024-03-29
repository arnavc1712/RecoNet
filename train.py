import torch
import torch.optim as optim
import opts
from torch.utils.data import DataLoader
from utils.utils import *
from data.data_loader import RecDataset,rec_collate_fn
from model.transformer.recModel import Encoder
import torch.optim as optim
from losses import hinge_loss, adaptive_hinge_loss,binary_cross_entropy
import os





def train(loader,model,optimizer,opt):
	model.train()
	epoch_loss = 0.0
	ix_to_item = loader.dataset.get_ix_to_item()
	item_to_ix = loader.dataset.get_item_to_ix()
	for epoch in range(opt['epochs']):
		for i,data in enumerate(loader):

			# print(data.shape)
			# print(data)
			# break


			src_pos = pos_generate(data)
			
			
			user_rep,_,attns= model.user_representation(data,src_pos,return_attns=True)

			positive_prediction = model(user_rep,data[:,1:])
			# print(positive_prediction)

			# negative_var = model._get_negative_prediction(data[:,1:].size(),user_rep)
			if opt["loss"]=="adaptive_hinge_loss":
				negative_prediction = model._get_multiple_negative_predictions(
	                        data[:,1:].size(),
	                        user_rep,
	                        n=opt["num_neg_sml"])

			else:
				negative_prediction = model._get_negative_prediction(data[:,1:].size(),user_rep)
			negative_prediction = model(user_rep, negative_prediction)

			show_predictions(data,_,model,ix_to_item,attns)
	
			optimizer.zero_grad()

			loss = binary_cross_entropy(positive_prediction,
											negative_prediction,
											mask=(data[:,1:] != 0))

			# loss = adaptive_hinge_loss(positive_prediction,
			# 				  negative_prediction,
			# 				  mask=(data[:,1:] != 0))
			epoch_loss += loss.item()

			print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
			# print(loss.item())
			loss.backward()

			optimizer.step()

		# if epoch % opt['save_checkpoint_every'] == 0:
		# 	model_path = os.path.join(opt['checkpoint_path'], 'recnet_%d.pth' % (epoch))
		# 	model_info_path = os.path.join(opt['checkpoint_path'], 'model_score.txt')
		# 	torch.save(model.state_dict(), model_path)
			
		# 	print('model saved to %s' % (model_path))
		# 	with open(model_info_path, 'a') as f:
		# 		f.write('recnet_%d, loss: %.6f\n' % (epoch, loss.item()))

	





def main(opt):
	dataset = RecDataset('train',opt)
	dataloader = DataLoader(dataset,batch_size=128,shuffle=True,collate_fn=rec_collate_fn)
	print(dataset.get_num_items())
	model = Encoder(seq_len=opt['max_seq_len'],
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

	optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                          betas=(0.9, 0.98), eps=1e-09)
	train(dataloader,model,optimizer,opt)


if __name__ == "__main__":
	print("Running")
	opt = opts.parse_opt()
	opt = vars(opt)
	main(opt)