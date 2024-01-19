import torch
import torch.nn.functional as F
import torch.nn as nn


def cal_nll_loss(logit, idx, mask, weights=None):
    eps = 0.1
    acc = (logit.max(dim=-1)[1]==idx).float()
    mean_acc = (acc * mask).sum() / mask.sum()
    
    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -logit.sum(dim=-1)
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        nll_loss = (nll_loss * weights).sum(dim=-1)

    return nll_loss.contiguous(), mean_acc


# main losses (reconstruction loss)
def main_loss_fn(words_logits, words_id, words_mask, num_props, mask_list, hneg_words_logits=None, **kwargs):
    bsz = words_logits[mask_list[0]].size(0) // num_props

    nll_losses = []
    nll_losses_mask = dict()
    min_nll_loss_mask = dict()
    idx_mask = dict()

    final_loss = 0
    for mask in mask_list:
        num_props_i = words_logits[mask].size(0) // bsz

        words_mask_i = words_mask.unsqueeze(1) \
            .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
        words_id_i = words_id.unsqueeze(1) \
            .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
        nll_loss, acc = cal_nll_loss(words_logits[mask], words_id_i, words_mask_i)

        nll_losses.append(nll_loss.view(bsz, num_props_i))
        nll_losses_mask[mask] = nll_loss.view(bsz, num_props_i)

    min_nll_loss, idx = torch.cat(nll_losses, 1).sort(dim=-1)
    min_nll_loss = min_nll_loss[:, 0:kwargs['num_train_prop']]
    idx = idx[:, 0:kwargs['num_train_prop']]
    final_loss += min_nll_loss.mean()

    if hneg_words_logits:
        hneg_nll_losses = []
        idx_hneg = torch.zeros_like(idx).cuda()
        idx_hneg.requires_grad = False
        for mask in mask_list:
            hneg_nll_loss, hneg_acc = cal_nll_loss(hneg_words_logits[mask], words_id, words_mask)
            hneg_nll_losses.append(hneg_nll_loss.view(bsz, -1))

        min_hneg_nll_loss = torch.cat(hneg_nll_losses, 1).gather(index=idx_hneg, dim=-1)

        final_loss += min_hneg_nll_loss.mean()

        final_loss = final_loss / 2
    
    loss_dict = {
        'final_loss': final_loss.item(),
        'nll_loss': min_nll_loss.mean().item(),
    }
    if hneg_words_logits:
        loss_dict.update({
            'hneg_nll_loss': min_hneg_nll_loss.mean().item(),
            })

    return final_loss, loss_dict


# sub losses (pushing loss, pulling loss, intra-video contrastive loss)
def sub_loss_fn(words_logits, words_id, words_mask, num_props, mask_list, neg_words_logits=None, hneg_words_logits=None, **kwargs):
    bsz = words_logits[mask_list[0]].size(0) // num_props

    nll_losses = []
    nll_losses_mask = dict()
    min_nll_loss_mask = dict()
    idx_mask = dict()
    for mask in mask_list:
        num_props_i = words_logits[mask].size(0)//bsz

        words_mask_i = words_mask.unsqueeze(1) \
            .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
        words_id_i = words_id.unsqueeze(1) \
            .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
        nll_loss, acc = cal_nll_loss(words_logits[mask], words_id_i, words_mask_i)

        nll_losses.append(nll_loss.view(bsz, num_props_i))
        nll_losses_mask[mask] = nll_loss.view(bsz, num_props_i)

    min_nll_loss, idx = torch.cat(nll_losses, 1).sort(dim=-1)

    min_nll_loss = min_nll_loss[:, 0:kwargs['num_train_prop']]
    idx = idx[:, 0:kwargs['num_train_prop']]

    rank_loss = 0
    if hneg_words_logits:
        hneg_nll_losses = []
        idx_hneg = torch.zeros_like(idx).cuda()
        idx_hneg.requires_grad = False
        for mask in mask_list:
            hneg_nll_loss, hneg_acc = cal_nll_loss(hneg_words_logits[mask], words_id, words_mask)
            hneg_nll_losses.append(hneg_nll_loss.view(bsz, -1))
            
        min_hneg_nll_loss = torch.cat(hneg_nll_losses, 1).gather(index=idx_hneg, dim=-1)

        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        hneg_loss = torch.max(min_nll_loss - min_hneg_nll_loss + kwargs["margin_hneg"], tmp_0)
        rank_loss += kwargs['alpha_ivc'] * hneg_loss.mean()

    if neg_words_logits:
        neg_nll_losses = []
        for mask in mask_list:
            num_props_i = words_logits[mask].size(0)//bsz

            words_mask_i = words_mask.unsqueeze(1) \
                .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
            words_id_i = words_id.unsqueeze(1) \
                .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
            neg_nll_loss, neg_acc = cal_nll_loss(neg_words_logits[mask], words_id_i, words_mask_i)

            neg_nll_losses.append(neg_nll_loss.view(bsz, num_props_i))
            
        min_neg_nll_losses = torch.cat(neg_nll_losses, 1).gather(index=idx, dim=-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        neg_loss = torch.max(min_nll_loss - min_neg_nll_losses + kwargs["margin_eneg"], tmp_0)
        rank_loss += kwargs['alpha_ivc'] * neg_loss.mean()

    loss = rank_loss

    push_losses = 0

    for mask in mask_list:
        num_props_i = words_logits[mask].size(0)//bsz

        prop_weight = kwargs['prop_weights'][mask].view(bsz, num_props_i, -1)

        prop_weight = prop_weight / prop_weight.sum(dim=-1, keepdim=True)
        target1 = torch.eye(num_props_i).unsqueeze(0).cuda() * kwargs["lambda_inter_push"]
        source1 = torch.bmm(prop_weight, prop_weight.transpose(1, 2))
        push_loss1 = torch.norm(target1 - source1, p="fro", dim=[1, 2]) ** 2

        push_losses += push_loss1
        if torch.is_tensor(push_loss1):
            loss += kwargs['alpha_inter_push'] * push_loss1.mean()

        push_loss2 = 0
        each_mask_weights = kwargs['each_mask_weights'][mask]
        for each_mask_weight in each_mask_weights:
            _, num_props_i, props_len = each_mask_weight.size()
            if num_props_i == 1:
                continue
            each_mask_weight_ = each_mask_weight / each_mask_weight.sum(dim=-1, keepdim=True)
            target2 = torch.eye(num_props_i).unsqueeze(0).cuda() * kwargs["lambda_intra_push"]
            source2 = torch.bmm(each_mask_weight_, each_mask_weight_.transpose(1, 2))
            push_loss2 += (torch.norm(target2 - source2, p="fro", dim=[1, 2]) ** 2)

        push_losses += push_loss2
        if torch.is_tensor(push_loss2):
            loss += kwargs['alpha_intra_push'] * push_loss2.mean()

    pull_losses = 0
    loss_fn = nn.MSELoss()

    for mask in mask_list:
        mask_centers = kwargs['mask_centers'][mask]
        for i in range(num_props):
            num_props_i = i+1
            mask_center = mask_centers[i].view(bsz, num_props_i)
            pull_loss = 0
            
            # two props pulling
            if i > 0:
                mask_center_sorted = mask_center.sort(dim=-1)[0]
                pull_loss += loss_fn(mask_center_sorted[:,0], mask_center_sorted[:,-1])

            pull_losses += pull_loss

            
    if torch.is_tensor(pull_losses):	
        loss += kwargs['alpha_pull'] * pull_losses.mean()

    return loss, {
            'ivc_loss': rank_loss.item() if rank_loss != 0 else 0.0,
            'neg_loss': neg_loss.mean().item() if neg_words_logits else 0.0,
            'hneg_loss': hneg_loss.mean().item() if hneg_words_logits else 0.0,
            'push_loss': push_losses.mean().item() if torch.is_tensor(push_losses) else 0.0,
            'pull_loss': pull_losses.mean().item() if torch.is_tensor(pull_losses) else 0.0,
        }
