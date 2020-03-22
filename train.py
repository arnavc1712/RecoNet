import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.utils import *
from data.data_loader import RecDataset,rec_collate_fn
from model.transformer.recModel import Encoder
import torch.optim as optim
from losses import hinge_loss, adaptive_hinge_loss





def train(loader,model,optimizer):
	model.train()
	epoch_loss = 0.0
	ix_to_item = loader.dataset.get_ix_to_item()
	item_to_ix = loader.dataset.get_item_to_ix()
	for epoch in range(100):
		for i,data in enumerate(loader):

			# print(data.shape)
			# print(data)
			# break


			src_pos = pos_generate(data)
			
			
			user_rep,_,attns= model.user_representation(data,src_pos,return_attns=True)

			positive_prediction = model(user_rep,data[:,1:])

			# negative_var = model._get_negative_prediction(data[:,1:].size(),user_rep)
			negative_prediction = model._get_multiple_negative_predictions(
                        data[:,1:].size(),
                        user_rep,
                        n=10)
			negative_prediction = model(user_rep, negative_prediction)

			# print(negative_prediction)
			# print(negative_prediction.shape)
			# print(positive_prediction)
			show_predictions(data,_,model,ix_to_item,attns)
			# print(_.size())

			# print(user_rep)
			# print(user_rep.shape)

			optimizer.zero_grad()

			loss = adaptive_hinge_loss(positive_prediction,
							  negative_prediction,
							  mask=(data[:,1:] != 0))
			epoch_loss += loss.item()

			print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
			# print(loss.item())
			loss.backward()

			optimizer.step()


	





def main():
	dataset = RecDataset('train')
	dataloader = DataLoader(dataset,batch_size=128,shuffle=True,collate_fn=rec_collate_fn)
	model = Encoder(seq_len=10,
            dim_item=50,
            n_items=1683,
            n_layers=2, n_head=2, d_k=50, d_v=50,
            d_model=50, d_inner=50, input_dropout_p=0.2,dropout=0.1)

	optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                          betas=(0.9, 0.98), eps=1e-09)
	train(dataloader,model,optimizer)


if __name__ == "__main__":
	print("Running")
	main()