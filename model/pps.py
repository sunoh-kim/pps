import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module import AttentivePooling, SinusoidalPositionalEmbedding, TrainablePositionalEmbedding, DualTransformer 

import math

class PPS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_epoch = config['max_epoch']
        self.dropout = config['dropout']
        self.vocab_size = config['vocab_size']

        # gaussian mixture proposal
        self.mask_type = config['mask_type']
        self.num_props = config['num_props']

        self.sigma_gauss = config["sigma_gauss"]
        self.sigma_laplace = config["sigma_laplace"]

        self.use_attn = config['use_attn']
        if self.use_attn:
            self.attn_func = AttentivePooling(config['hidden_dim'])

        # negative proposal
        self.neg_type = config['neg_type']
        self.num_neg_mask = config['num_neg_mask']
        self.sigma_neg_inv_gauss = config["sigma_neg_inv_gauss"]
        self.sigma_neg_gauss = config["sigma_neg_gauss"]

        # encoder
        self.frame_fc = nn.Linear(config['frame_feat_dim'], config['hidden_dim'])
        self.word_fc = nn.Linear(config['word_feat_dim'], config['hidden_dim'])
        self.mask_vec = nn.Parameter(torch.zeros(config['word_feat_dim']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['word_feat_dim']).float(), requires_grad=True)
        self.pred_vec_v = nn.Parameter(torch.zeros(config['frame_feat_dim']).float(), requires_grad=True)
        self.pred_vec_q = nn.Parameter(torch.zeros(config['hidden_dim']).float(), requires_grad=True)

        # transformer
        self.pos_mask_list = []
        self.neg_mask_list = []

        self.trans_mask_generator = nn.ModuleDict()
        self.fc_mask = nn.ModuleDict()
        if self.mask_type == 'gaussian':
            self.pos_mask_list.append('gauss')
            self.trans_mask_generator['gauss'] = DualTransformer(**config['DualTransformer'])

            self.fc_mask['gauss'] = nn.ModuleList([nn.Linear(config['hidden_dim'], 2*(i+1)) for i in range(self.num_props)])

        elif self.mask_type == 'laplace':
            self.pos_mask_list.append('laplace')
            self.trans_mask_generator['laplace'] = DualTransformer(**config['DualTransformer'])

            self.fc_mask['laplace'] = nn.ModuleList([nn.Linear(config['hidden_dim'], 2*(i+1)) for i in range(self.num_props)])

        if self.neg_type == 'learnable_inverse_gaussian':
            self.neg_mask_list.append('l_inv_gauss')
            self.trans_mask_generator['l_inv_gauss'] = DualTransformer(**config['DualTransformer'])
            self.fc_mask['l_inv_gauss'] = nn.ModuleDict()
            if self.mask_type == 'gaussian':
                self.fc_mask['l_inv_gauss']['gauss'] = nn.Linear(config['hidden_dim'], self.num_props*self.num_neg_mask*2)
            if self.mask_type == 'laplace':
                self.fc_mask['l_inv_gauss']['laplace'] = nn.Linear(config['hidden_dim'], self.num_props*self.num_neg_mask*2)
        elif self.neg_type == 'learnable_gaussian':
            self.neg_mask_list.append('l_gauss')
            self.trans_mask_generator['l_gauss'] = DualTransformer(**config['DualTransformer'])
            self.fc_mask['l_gauss'] = nn.ModuleDict()
            if self.mask_type == 'gaussian':
                self.fc_mask['l_gauss']['gauss'] = nn.Linear(config['hidden_dim'], self.num_props*self.num_neg_mask*2)
            if self.mask_type == 'laplace':
                self.fc_mask['l_gauss']['laplace'] = nn.Linear(config['hidden_dim'], self.num_props*self.num_neg_mask*2)
        elif self.neg_type == 'inverse_gaussian':
            self.neg_mask_list.append('inv_gauss')
        elif self.neg_type == 'gaussian':
            self.neg_mask_list.append('gauss')
        elif self.neg_type == 'saturated_gaussian':
            self.neg_mask_list.append('sat_gauss')
        else:
            raise NotImplementedError

        self.trans_reconstructor = DualTransformer(**config['DualTransformer'])
        self.fc_comp = nn.Linear(config['hidden_dim'], self.vocab_size)

        # positional encoder
        self.word_pos_encoder = SinusoidalPositionalEmbedding(config['hidden_dim'], 0, config['max_num_words']+1)

        self.downsample_ratio = 4
        self.masked_words_ratio = config['masked_words_ratio']
        self.pred_gamma = config['pred_gamma']

    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights, **kwargs):
        bsz, n_frames, _ = frames_feat.shape

        # encoding
        pred_vec_v = self.pred_vec_v.view(1, 1, -1).expand(bsz, 1, -1)
        frames_feat = torch.cat([frames_feat, pred_vec_v], dim=1)
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)
        frames_mask_ori = _generate_mask(frames_feat, frames_len)

        words_feat[:, 0] = self.start_vec.cuda()
        words_feat_ori = words_feat

        words_pos = self.word_pos_encoder(words_feat)

        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)

        ## Generation
        # generate centers & widths for masks
        h = dict()
        enc_out = dict()
        mask_centers = dict()
        mask_widths = dict()
        for mask in self.pos_mask_list:
            enc_out[mask], h[mask] = self.trans_mask_generator[mask](frames_feat, frames_mask_ori, words_feat + words_pos, words_mask, decoding=1)

            mask_centers[mask] = []
            mask_widths[mask] = []

            for i in range(self.num_props):
                num_props_i = i+1
                mask_param = torch.sigmoid(self.fc_mask[mask][i](h[mask][:, -1])).view(bsz*num_props_i, 2)
                mask_center = mask_param[:, 0]
                mask_width = mask_param[:, 1]
                # use one width
                mask_width = mask_width[:bsz].unsqueeze(1).expand(bsz,num_props_i).contiguous().view(bsz*num_props_i)
                mask_centers[mask].append(mask_center)
                mask_widths[mask].append(mask_width)

        for neg_mask in self.neg_mask_list:
            if neg_mask == 'l_inv_gauss' or neg_mask == 'l_gauss':
                enc_out[neg_mask], h[neg_mask] = self.trans_mask_generator[neg_mask](frames_feat, frames_mask_ori, words_feat + words_pos, words_mask, decoding=1)

                mask_centers[neg_mask] = dict()
                mask_widths[neg_mask] = dict()

                for pos_mask in self.pos_mask_list:
                    mask_param = torch.sigmoid(self.fc_mask[neg_mask][pos_mask](h[mask][:, -1])).view(bsz*self.num_props*self.num_neg_mask, 2)
                    mask_center = mask_param[:, 0]
                    mask_width = mask_param[:, 1]
                    mask_centers[neg_mask][pos_mask] = mask_center
                    mask_widths[neg_mask][pos_mask] = mask_width

        # generate masks for proposal
        props_len = n_frames//self.downsample_ratio
        keep_idx = torch.linspace(0, n_frames-1, steps=props_len).long()

        prop_weights = dict()
        prop_lefts = dict()
        prop_rights = dict()
        each_mask_weights = dict()

        prop_lefts_neg = dict()
        prop_rights_neg = dict()

        for mask in self.pos_mask_list:
            prop_weights[mask] = []
            prop_lefts[mask] = []
            prop_rights[mask] = []
            each_mask_weights[mask] = []
            prop_lefts_neg[mask] = []
            prop_rights_neg[mask] = []

            for i in range(self.num_props):
                if mask == 'laplace':
                    weight = self.generate_laplace_mask\
                        (props_len, mask_centers[mask][i], mask_widths[mask][i], num_mask=i+1)
                elif mask == 'gauss':
                    weight = self.generate_gauss_mask\
                        (props_len, mask_centers[mask][i], mask_widths[mask][i], num_mask=i+1)
                else:
                    raise NotImplementedError

                prop_weight, prop_left, prop_right, each_mask_weight = self.generate_prop\
                    (weight, props_len, mask_centers[mask][i], mask_widths[mask][i], num_mask=i+1)

                prop_left_neg = torch.clamp(prop_left.min(dim=1, keepdim=True)[0], min=0)
                prop_right_neg = torch.clamp(prop_right.max(dim=1, keepdim=True)[0], max=1)

                prop_lefts_neg[mask].append(prop_left_neg)
                prop_rights_neg[mask].append(prop_right_neg)

                prop_weights[mask].append(prop_weight)
                prop_lefts[mask].append(prop_left)
                prop_rights[mask].append(prop_right)
                each_mask_weights[mask].append(each_mask_weight)

            prop_weights[mask] = torch.cat(prop_weights[mask], 1).view(bsz*self.num_props, props_len)

        ## Reconstruction
        # masking
        frames_feat = frames_feat[:, keep_idx]
        frames_mask = frames_mask_ori[:, keep_idx]

        words_feat = torch.cat([words_feat_ori, torch.zeros(bsz, 1, words_feat_ori.shape[2]).float().cuda()], dim=1)
        for b in range(bsz):
            words_feat[b, words_len[b]+1] = self.pred_vec_q[b].cuda()

        words_pos = self.word_pos_encoder(words_feat)

        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)

        words_feat, masked_words = self._mask_words(words_feat, words_len, weights=weights)
        words_feat = words_feat + words_pos

        words_logits = dict()

        # make mask attention
        mask_attns_tensor = None
        if self.use_attn:
            for mask in self.pos_mask_list:
                prop_attn_weights = []
                mask_attns_tensor = torch.zeros(bsz, self.num_props, self.num_props).float().cuda()
                mask_attns_tensor[:,0,:1] = 1
                for i in range(self.num_props):
                    each_mask_weight = each_mask_weights[mask][i]
                    if i == 0:
                        prop_attn_weights.append(each_mask_weight)
                        continue
                    num_props_i = i + 1

                    props_feat_i = frames_feat.unsqueeze(1) \
                        .expand(bsz, num_props_i, -1, -1).contiguous().view(bsz*num_props_i, props_len, -1)
                    props_mask_i = frames_mask.unsqueeze(1) \
                        .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
                    words_mask_i = words_mask.unsqueeze(1) \
                        .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
                    words_id_i = words_id.unsqueeze(1) \
                        .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
                    words_feat_i = words_feat.unsqueeze(1) \
                        .expand(bsz, num_props_i, -1, -1).contiguous().view(bsz*num_props_i, words_mask_i.size(1), -1)

                    each_mask_weight = each_mask_weight.view(bsz*num_props_i, props_len)


                    _, h_each_mask = self.trans_reconstructor(props_feat_i, props_mask_i, words_feat_i, words_mask_i, decoding=2, gauss_weight=each_mask_weight)

                    h_cls = h_each_mask.gather(index=words_mask_i.sum(dim=1,keepdims=True).unsqueeze(1).expand(-1,1,h_each_mask.shape[2]), dim=1)

                    h_cls = h_cls.view(bsz, num_props_i, -1)

                    _, mask_attn = self.attn_func(h_cls)
                    each_attn_mask_weight = mask_attn.unsqueeze(2) * each_mask_weight.view(bsz, num_props_i, props_len)
                    each_attn_mask_weight = each_attn_mask_weight.sum(dim=1)
                    
                    each_attn_mask_weight = each_attn_mask_weight.unsqueeze(1)
                    prop_attn_weights.append(each_attn_mask_weight)

                    mask_attns_tensor[:,i,:num_props_i] = mask_attn

                    prop_lefts[mask][i] = (mask_attn * prop_lefts[mask][i]).sum(dim=1, keepdims=True)
                    prop_rights[mask][i] = (mask_attn * prop_rights[mask][i]).sum(dim=1, keepdims=True)

                prop_weights[mask] = torch.cat(prop_attn_weights, 1).view(bsz*self.num_props, props_len)


        # make proposal's left and right points
        for mask in self.pos_mask_list:
            prop_lefts[mask] = torch.cat(prop_lefts[mask], 1)
            prop_rights[mask] = torch.cat(prop_rights[mask], 1)
            prop_width = prop_rights[mask] - prop_lefts[mask]
            prop_lefts[mask] = torch.clamp(prop_lefts[mask]+self.pred_gamma*prop_width/2, min=0)
            prop_rights[mask] = torch.clamp(prop_rights[mask]-self.pred_gamma*prop_width/2, max=1)

        # make positive proposal
        for mask in self.pos_mask_list:
            words_logits[mask] = []

            num_props_i = self.num_props

            props_feat_i = frames_feat.unsqueeze(1) \
                .expand(bsz, num_props_i, -1, -1).contiguous().view(bsz*num_props_i, props_len, -1)
            props_mask_i = frames_mask.unsqueeze(1) \
                .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
            words_mask_i = words_mask.unsqueeze(1) \
                .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
            words_id_i = words_id.unsqueeze(1) \
                .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
            words_feat_i = words_feat.unsqueeze(1) \
                .expand(bsz, num_props_i, -1, -1).contiguous().view(bsz*num_props_i, words_mask_i.size(1), -1)

            _, h2 = self.trans_reconstructor(props_feat_i, props_mask_i, words_feat_i, words_mask_i, decoding=2, gauss_weight=prop_weights[mask])
            words_logit = self.fc_comp(h2)
            words_logits[mask] = words_logit[:,:-2]

        # make negative proposal
        neg_words_logits = dict()
        hneg_words_logits = dict()
        neg_prop_weights = dict()

        for mask in self.pos_mask_list:
            neg_prop_weights[mask] = dict()

            prop_lefts_neg[mask] = torch.cat(prop_lefts_neg[mask], 1)
            prop_rights_neg[mask] = torch.cat(prop_rights_neg[mask], 1)

            num_props_i = self.num_props

            props_feat_i = frames_feat.unsqueeze(1) \
                .expand(bsz, num_props_i, -1, -1).contiguous().view(bsz*num_props_i, props_len, -1)
            props_mask_i = frames_mask.unsqueeze(1) \
                .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
            words_mask_i = words_mask.unsqueeze(1) \
                .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
            words_id_i = words_id.unsqueeze(1) \
                .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
            words_feat_i = words_feat.unsqueeze(1) \
                .expand(bsz, num_props_i, -1, -1).contiguous().view(bsz*num_props_i, words_mask_i.size(1), -1)

            for neg_mask in self.neg_mask_list:
                if neg_mask == 'l_inv_gauss':
                    neg_weight = self.generate_negative_inverse_gaussian_mask\
                        (props_len, mask_centers[neg_mask][mask], mask_widths[neg_mask][mask])
                    neg_weight = neg_weight.view(bsz*num_props_i, self.num_neg_mask, -1)
                    neg_weight = neg_weight.sum(1)
                    neg_weight = neg_weight/neg_weight.max(dim=-1, keepdim=True)[0]
                elif neg_mask == 'l_gauss':
                    neg_weight = self.generate_negative_gauss_mask\
                        (props_len, mask_centers[neg_mask][mask], mask_widths[neg_mask][mask])
                    neg_weight = neg_weight.view(bsz*num_props_i, self.num_neg_mask, -1)
                    neg_weight = neg_weight.sum(1)
                    neg_weight = neg_weight/neg_weight.max(dim=-1, keepdim=True)[0]
                else:
                    neg_weight = self.generate_not_learnable_negative_mask(props_len, prop_lefts_neg[mask], prop_rights_neg[mask], kwargs['epoch'])

                _, neg_h = self.trans_reconstructor(props_feat_i, props_mask_i, words_feat_i, words_mask_i, decoding=2, gauss_weight=neg_weight)

                neg_words_logit = self.fc_comp(neg_h)
                neg_words_logits[mask] = neg_words_logit[:,:-2]
                neg_prop_weights[mask][neg_mask] = neg_weight

            _, hneg_h = self.trans_reconstructor(frames_feat, frames_mask, words_feat, words_mask, decoding=2)

            hneg_words_logit = self.fc_comp(hneg_h)
            hneg_words_logits[mask] = hneg_words_logit[:,:-2]

        words_mask = words_mask[:,:-2]

        return {
            'neg_words_logits': neg_words_logits,
            'hneg_words_logits': hneg_words_logits,
            'words_logits': words_logits,
            'words_id': words_id,
            'words_mask': words_mask,
            'prop_weights': prop_weights,
            'prop_lefts': prop_lefts,
            'prop_rights': prop_rights,
            'each_mask_weights': each_mask_weights,
            'frames_mask': frames_mask_ori,
            'neg_prop_weights': neg_prop_weights,
            'mask_centers': mask_centers,
            'mask_attns': mask_attns_tensor
        }

    def generate_prop(self, weight, props_len, center, width, num_mask=1):
        bsz = center.size(0) // num_mask

        left = center-width/2
        right = center+width/2

        left = left.view(bsz, num_mask)
        right = right.view(bsz, num_mask)

        if num_mask == 1:
            prop_weight = weight
            prop_weight = prop_weight.unsqueeze(1)
            each_mask_weight = prop_weight

            return prop_weight, left, right, each_mask_weight
        else:
            each_mask_weight = weight
            each_mask_weight = each_mask_weight.view(bsz, num_mask, props_len)
            prop_weight = weight.view(bsz, num_mask, props_len).sum(1)
            prop_weight = prop_weight/prop_weight.max(dim=-1, keepdim=True)[0]
            prop_weight = prop_weight.unsqueeze(1)

            prop_left = left
            prop_right = right

            return prop_weight, prop_left, prop_right, each_mask_weight

    def generate_gauss_mask(self, props_len, center, width, num_mask=1):
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        mask_center = center.unsqueeze(-1)
        mask_width = width.unsqueeze(-1).clamp(1e-2) / self.sigma_gauss

        w = 0.3989422804014327  # inverse square root 2 pi
        weight = w/mask_width*torch.exp(-(weight-mask_center)**2/(2*mask_width**2))
        weight = weight/weight.max(dim=-1, keepdim=True)[0]

        return weight

    def generate_laplace_mask(self, props_len, center, width, num_mask=1):
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        mask_center = center.unsqueeze(-1)
        mask_width = width.unsqueeze(-1).clamp(1e-2) / self.sigma_laplace

        weight = 1/(2*mask_width)*torch.exp(-torch.abs(weight-mask_center)/(mask_width))
        weight = weight/weight.max(dim=-1, keepdim=True)[0]

        return weight
    
    def generate_negative_gauss_mask(self, props_len, center, width, num_mask=1):
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        mask_center = center.unsqueeze(-1)
        mask_width = width.unsqueeze(-1).clamp(1e-2) / self.sigma_neg_gauss

        w = 0.3989422804014327  # inverse square root 2 pi
        weight = w/mask_width*torch.exp(-(weight-mask_center)**2/(2*mask_width**2))
        weight = weight/weight.max(dim=-1, keepdim=True)[0]

        return weight

    def generate_negative_inverse_gaussian_mask(self, props_len, center, width):
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        mask_center = center.unsqueeze(-1)
        mask_width = width.unsqueeze(-1).clamp(1e-2) / self.sigma_neg_inv_gauss

        w = 0.3989422804014327  # inverse square root 2 pi
        weight = w/mask_width*torch.exp(-(weight-mask_center)**2/(2*mask_width**2))
        weight = 1-weight/weight.max(dim=-1, keepdim=True)[0]

        return weight

    def generate_not_learnable_negative_mask(self, props_len, left, right, epoch):
        def Gauss(pos, w1, c):
            w1 = w1.unsqueeze(-1).clamp(1e-2) / (self.sigma_neg_gauss)
            c = c.unsqueeze(-1)
            w = 0.3989422804014327  # inverse square root 2 pi
            y1 = w/w1*torch.exp(-(pos-c)**2/(2*w1**2))
            return y1/y1.max(dim=-1, keepdim=True)[0]

        left = left.view(-1)
        right = right.view(-1)

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(left.size(0), -1).to(left.device)

        if self.neg_type == 'inverse_gaussian':
            center = (right + left)/2
            width = right - left

            mask_center = center.unsqueeze(-1)
            mask_width = width.unsqueeze(-1).clamp(1e-2) / self.sigma_neg_inv_gauss

            w = 0.3989422804014327  # inverse square root 2 pi
            neg_weight = w/mask_width*torch.exp(-(weight-mask_center)**2/(2*mask_width**2))
            neg_weight = 1-neg_weight/neg_weight.max(dim=-1, keepdim=True)[0]
            return neg_weight

        left_width = left
        left_center = left_width * 0.5
        right_width = 1 - right
        right_center = 1 - right_width * 0.5

        if self.neg_type == 'gaussian':
            left_neg_weight = Gauss(weight, left_center, left_center)
            right_neg_weight = Gauss(weight, 1-right_center, right_center)
        elif self.neg_type == 'saturated_gaussian':
            left_neg_weight = Gauss(weight, left_center, left_center)
            right_neg_weight = Gauss(weight, 1-right_center, right_center)
            for b in range(left.size(0)):
                left_center_b = max(int(left_center[b] * props_len)+1, 0)
                right_center_b = min(int(right_center[b] * props_len), props_len-1)
                left_neg_weight[b][:left_center_b] = 1
                right_neg_weight[b][right_center_b:] = 1
        else:
            raise NotImplementedError

        left_neg_weight = left_neg_weight/left_neg_weight.max(dim=-1, keepdim=True)[0]
        right_neg_weight = right_neg_weight/right_neg_weight.max(dim=-1, keepdim=True)[0]

        return left_neg_weight+right_neg_weight

    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)

            num_masked_words = max(l // self.masked_words_ratio, 1)

            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())
            if l < 1:
                continue
            p = weights[i, :l].cpu().numpy() if weights is not None else None

            if num_masked_words <= sum(p > 0):
                choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            else:
                choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=True, p=p)

            masked_words[-1][choices] = 1
        
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words

    def froze_mask_generator(self):
        for name, param in self.named_parameters():
            if 'fc_mask' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def froze_reconstructor(self):
        for name, param in self.named_parameters():
            if 'fc_mask' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfroze(self):
        for name, param in self.named_parameters():
            param.requires_grad = True


def _generate_mask(x, x_len):
    mask = []
    for l in x_len:
        mask.append(torch.zeros([x.size(1)]).byte().cuda())
        mask[-1][:l] = 1
    mask = torch.stack(mask, 0)
    return mask

