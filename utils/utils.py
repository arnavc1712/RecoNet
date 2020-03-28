import torch
import torch.nn as nn
import torch.nn.functional as F
import model.transformer.Constants as Constants
import random
import numpy as np


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    seq = seq.cpu()
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j].item()
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        # self.loss_fn = nn.NLLLoss(reduce=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        batch_size = target.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    return loss, n_correct


def show_predictions(input_ids,target_ids,user_rep,model,ix_to_item,attns):
    random_id = random.randint(0, len(seq)-1)

    print("Sequence")
    target_ids=target_ids.cpu()
    input_ids = input_ids.cpu()
    # user_rep = user_rep.cpu()
    # model = model.cpu()
    print(list(map(lambda x:ix_to_item[x],input_ids[random_id].numpy().flatten())))
    print("\n")
    target = target_ids[random_id][-1:]
    user_rep = user_rep[random_id]

    item_ids = np.array(list(ix_to_item.keys())).reshape(-1,1)
    item_ids = torch.from_numpy(item_ids).type(torch.LongTensor).cuda()
    size = (len(item_ids),) + user_rep.size()
    out = model(user_rep.expand(*size),item_ids)
    preds = out.detach().cpu().numpy().flatten()

    most_probable_10 = preds.argsort()[-10:][::-1]
    most_prob_10_items = list(map(lambda x:ix_to_item[x],most_probable_10))
    g_t = ix_to_item[seq.detach().numpy().flatten()[0]]

    print("Most probable")
    print(most_prob_10_items)
    print("\n")

    print("Attnentions")
    print(attns[random_id][-1])

    print("True Label")
    print(g_t)
    print("\n")




def pos_generate(item_seq):

    seq = list(range(1, item_seq.shape[1] + 1))
    src_pos = torch.tensor([seq] * item_seq.shape[0])

    return src_pos


def pos_emb_generation(visual_feats, word_labels):
    '''
        Generate the position embedding input for Transformers.
    '''
    seq = list(range(1, visual_feats.shape[1] + 1))
    src_pos = torch.tensor([seq] * visual_feats.shape[0]).cuda()

    seq = list(range(1, word_labels.shape[1] + 1))
    tgt_pos = torch.tensor([seq] * word_labels.shape[0]).cuda()
    binary_mask = (word_labels != 0).long()

    return src_pos, tgt_pos*binary_mask


# def show_prediction(seq_probs, labels, vocab):
#     '''
#         :return: predicted words and GT words.
#     '''
#     # Print out the predicted sentences and GT
#     _ = seq_probs.view(labels.shape[0], labels[:, :-1].shape[1], -1)[0]
#     pred_idx = torch.argmax(_, 1)
#     print(' \n')
#     print([vocab[str(widx.cpu().numpy())] for widx in pred_idx if widx != 0])
#     print([vocab[str(word.cpu().numpy())] for word in labels[0] if word != 0])

