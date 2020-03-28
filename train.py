import torch
import torch.optim as optim
import opts
from torch.utils.data import DataLoader
from utils.utils import *
from data.data_loader import RecDataset,rec_collate_fn
from model.transformer.recModel import Encoder
import torch.optim as optim
from losses import hinge_loss, adaptive_hinge_loss
from model.transformer.Optim import ScheduledOptim
import os




def train(loader,model,optimizer,opt):
	model.train()
	epoch_loss = 0.0
	ix_to_item = loader.dataset.get_ix_to_item()
	item_to_ix = loader.dataset.get_item_to_ix()
	for epoch in range(opt['epochs']):
		for i,(input_ids,target_ids) in enumerate(loader):
			torch.cuda.synchronize()

			# print(data.shape)
			# print(data)
			# break


			src_pos = pos_generate(input_ids)
			input_ids = input_ids.cuda()
			target_ids = target_ids.cuda()
			src_pos = src_pos.cuda()
			
			
			user_rep,attns= model.user_representation(input_ids,src_pos,return_attns=True)


			positive_prediction = model(user_rep,target_ids)

			# negative_var = model._get_negative_prediction(data[:,1:].size(),user_rep)
			if opt["loss"]=="adaptive_hinge_loss":
				negative_prediction = model._get_multiple_negative_predictions(
	                        input_ids.size(),
	                        user_rep,
	                        n=opt["num_neg_sml"])
			else:
				negative_prediction = model._get_negative_prediction(input_ids.size(),user_rep)
			negative_prediction = model(user_rep, negative_prediction)

			# print(negative_prediction)
			# print(negative_prediction.shape)
			# print(positive_prediction)
			show_predictions(input_ids,target_ids,user_rep[:,-1:,:],model,ix_to_item,attns)
			# print(_.size())

			# print(user_rep)
			# print(user_rep.shape)

			optimizer.zero_grad()

			loss = adaptive_hinge_loss(positive_prediction,
							  negative_prediction,
							  mask=(target_ids != 0))
			epoch_loss += loss.item()
			torch.cuda.synchronize()
			
			print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}, Current lr: {optimizer._optimizer.param_groups[0]['lr']}")
			# print(loss.item())
			loss.backward()

			optimizer.step_and_update_lr()

		if epoch % opt['save_checkpoint_every'] == 0:
			model_path = os.path.join(opt['checkpoint_path'], 'recnet_%d.pth' % (epoch))
			model_info_path = os.path.join(opt['checkpoint_path'], 'model_score.txt')
			torch.save(model.state_dict(), model_path)
			
			print('model saved to %s' % (model_path))
			with open(model_info_path, 'a') as f:
				f.write('recnet_%d, loss: %.6f\n' % (epoch, loss.item()))

	





def main(opt):
	dataset = RecDataset('train',opt)
	dataloader = DataLoader(dataset,batch_size=128,shuffle=True,collate_fn=rec_collate_fn)

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

	model = model.cuda()


	optimizer = ScheduledOptim(optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                          betas=(0.9, 0.98), eps=1e-09),opt["dim_model"],opt["warm_up_steps"])
	train(dataloader,model,optimizer,opt)


if __name__ == "__main__":
	print("Running")
	opt = opts.parse_opt()
	opt = vars(opt)
	main(opt)