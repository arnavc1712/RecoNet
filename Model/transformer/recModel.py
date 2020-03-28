import torch
import numpy as np
import torch.nn as nn
import model.transformer.Constants as Constants
from model.transformer.Layers import EncoderLayer
from model.transformer.embed_layers import ScaledEmbedding
import torch.nn.functional as F
from sampling import sample_items


__author__ = 'Yu-Hsiang Huang'
__RevisedBy__ = 'Arnav Chakravarthy'


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])


    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table).cuda()


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1).cuda()


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask.cuda()


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask.cuda()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            seq_len,
            dim_item,
            n_items,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, input_dropout_p=0.2,dropout=0.1):

        super().__init__()

        n_position = seq_len+2  
        self.num_items=n_items

        self.d_model = d_model
        self.item_embedding = ScaledEmbedding(n_items,dim_item,padding_idx=0)
        self.input_dropout = nn.Dropout(input_dropout_p)

        self.item2hid = nn.Linear(dim_item,d_model)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position,dim_item,padding_idx=0), freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def user_representation(self,src_emb,src_pos,return_attns=False):

        ''' 
            src_emb: Sequences of items (batch_size,max_batch_len)
            src_emb: Position of items in sequence (batch_size,max_batch_len+1)

        '''
        enc_slf_attn_list = []

        ## We do not need padding masks as the video clips are 40 frames each

        # -- Prepare masks
        slf_attn_mask_subseq = get_subsequent_mask(src_emb) ## Get subsequent mask
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=src_emb, seq_q=src_emb) ## Get padding mask for avoiding attention
        non_pad_mask = get_non_pad_mask(src_emb)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0) ## To avoid subsequent items being involved in the self attention
 
        src_item_emb = self.item_embedding(src_emb) ## (Batch_size, max_seq_len, dim_item)
        # -- Forward
       
        enc_output = src_item_emb + self.position_enc(src_pos) ## (Batch_size, max_batch_len, dim_item)

        ## Converting dimensions of video + postional embeddings into d_model
        batch_size, seq_len, dim_item = enc_output.size()

        enc_output = enc_output.view(-1, dim_item)

        enc_output = self.item2hid(enc_output)

        enc_output = self.input_dropout(enc_output)
        enc_output = enc_output.view(batch_size,seq_len,self.d_model)


        for i,enc_layer in enumerate(self.layer_stack):
            enc_output, enc_slf_attn = enc_layer(enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += enc_slf_attn


        if return_attns:
            return enc_output[:,:-1,:],enc_output[:,-1:,:], enc_slf_attn_list
        
        return enc_output[:,:-1,:],enc_output[:,-1:,:],


    def _get_negative_prediction(self, shape, user_representation):

        negative_items = sample_items(
            self.num_items,
            shape)
            # random_state=self._random_state)
        negative_var = torch.from_numpy(negative_items)

        return negative_var

    def _get_multiple_negative_predictions(self, shape, user_representation,
                                           n=5):
        batch_size, sliding_window = shape
        size = (n,) + (1,) * (user_representation.dim() - 1)
        negative_prediction = self._get_negative_prediction(
            (n * batch_size, sliding_window),
            user_representation.repeat(*size))

        return negative_prediction.view(n, batch_size, sliding_window)

    def forward(self, user_rep,item_seq):
        target_embed = self.item_embedding(item_seq)
        dot = (user_rep*target_embed).sum(-1).squeeze()

        return dot

        

       
