import collections
import logging
import os

import numpy as np
import torch

from model.loss import cal_nll_loss, main_loss_fn, sub_loss_fn
from util.utils import TimeMeter, AverageMeter, Accumulator

import random
import wandb


def info(msg):
    """ 
    Add log and print it
    """
    print(msg)
    logging.info(msg)


class Runner:
    """ 
    Base runner for training and evaluating
    """
    def __init__(self, args):
        self.args = args
        self._build_dataset()

        self.args['model']['vocab_size'] = self.train_set.vocab_size
        self.args['model']['max_epoch'] = self.args['train']['num_epochs']

        # build model
        self._build_model()
        if 'train' in args:
            self._build_optimizer()
            self.num_updates = 0

        self.save_model = self.args['train']['save_model'] if 'save_model' in args['train'] else False
        self.use_wandb = self.args['train']['use_wandb'] if 'use_wandb' in args['train'] else False
        self.loss_meter = None
        self.evaluator = None
        self.counters = None

        self.use_early_end = self.args['train']['use_early_end'] if 'use_early_end' in self.args['train'] else False

        if self.args['dataset']['name'] == 'ActivityNet':
            self.key_to_vis = ['final_loss', 'ivc_loss', 'push_loss', 'pull_loss', 'Rank1,IoU@0.1', 'Rank1,IoU@0.3', 'Rank1,IoU@0.5', 'Rank1,mIoU', 'Rank5,IoU@0.1', 'Rank5,IoU@0.3', 'Rank5,IoU@0.5', 'Rank5,mIoU']
        if self.args['dataset']['name'] == 'CharadesSTA':
            self.key_to_vis = ['final_loss', 'ivc_loss', 'push_loss', 'pull_loss', 'Rank1,IoU@0.3', 'Rank1,IoU@0.5', 'Rank1,IoU@0.7', 'Rank1,mIoU', 'Rank5,IoU@0.3', 'Rank5,IoU@0.5', 'Rank5,IoU@0.7', 'Rank5,mIoU']

    def train(self):
        # make folder for save file
        if self.save_model:
            self.save_path = self.args['train']['save_path']
            os.makedirs(self.save_path, mode=0o755, exist_ok=True)
            os.system('cp %s %s'%(self.args['config_path'], os.path.join(self.save_path, 'config.json')))

        # make wandb to visualize learning curve
        if self.use_wandb:
            wandb.login()
            wandb.init(
                project="%s_%s" % (self.args['model']['name'], self.args['dataset']['name']),
                name=self.args['exp_name'],
                config=self.args,
                dir=self.args['train']['wandb_path'])

        # start training
        for epoch in range(1, self.args['train']['num_epochs']+1):
            info('Start Epoch {}'.format(epoch))

            # start one epoch
            self._train_one_epoch(epoch)

            # make save file
            if self.save_model:
                save_path = os.path.join(self.save_path, 'model-{}.pt'.format(epoch))
                self._save_model(save_path)

            results = self.eval(epoch=epoch)

            # update wandb
            if self.use_wandb:
                if self.counters is None:
                    self._create_counters()
                self.update_counters()
                wandb_dict = {v.get_name(): v.get_average() for k, v in self.counters.items() if k in self.key_to_vis}
                wandb.log(wandb_dict)
                self.reset_counters()
            
            info('=' * 60)

        if self.use_wandb:
            wandb.finish()

    def _train_one_epoch(self, epoch, **kwargs):
        self.model.train()

        # log function
        def print_log():
            msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, bid, curr_lr)
            for k, v in loss_meter.items():
                msg += '{} = {:.4f}, '.format(k, v.avg)
                v.reset()
            msg += '{:.3f} seconds/batch'.format(1.0 / time_meter.avg)
            info(msg)

        display_n_batches, bid = 50, 0
        time_meter = TimeMeter()
        loss_meter = collections.defaultdict(lambda: AverageMeter())

        # start one epoch
        for bid, batch in enumerate(self.train_loader, 1):
            # forward
            self.optimizer.zero_grad()
            net_input = move_to_cuda(batch['net_input'])
            output = self.model(epoch=epoch, **net_input)

            # compute loss
            # main losses (reconstruction loss)
            loss, loss_dict = main_loss_fn(**output, num_props=self.model.num_props, mask_list=self.model.pos_mask_list, **self.args['loss'])

            # sub losses (pushing loss, pulling loss, intra-video contrastive loss)
            sub_loss, sub_loss_dict = sub_loss_fn(**output, num_props=self.model.num_props, mask_list=self.model.pos_mask_list, **self.args['loss'])
            loss_dict.update(sub_loss_dict)
            loss = loss + sub_loss

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)

            # log
            time_meter.update()
            for k, v in loss_dict.items():
                loss_meter[k].update(v)

            if bid % display_n_batches == 0:
                print_log()

        if bid % display_n_batches != 0:
            print_log()

        self.loss_meter = loss_meter

    def eval(self, save=None, epoch=0):
        # evaluate
        self.model.eval()
        with torch.no_grad():
            metrics_logger = collections.defaultdict(lambda: AverageMeter())

            with torch.no_grad():
                # start iterations
                for bid, batch in enumerate(self.test_loader, 1):
                    durations = np.asarray([i[1] for i in batch['raw']])
                    gt = np.asarray([i[2] for i in batch['raw']])

                    # forward
                    net_input = move_to_cuda(batch['net_input'])
                    output = self.model(epoch=epoch, **net_input)
                    bsz = len(durations)

                    mask_list = self.model.pos_mask_list
                    neg_mask_list = self.model.neg_mask_list

                    # compute loss
                    nll_losses = []
                    for mask in mask_list:
                        words_logits = output['words_logits'][mask]
                        num_props_i = words_logits.size(0)//bsz

                        words_mask = output['words_mask'].unsqueeze(1) \
                            .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
                        words_id = output['words_id'].unsqueeze(1) \
                            .expand(bsz, num_props_i, -1).contiguous().view(bsz*num_props_i, -1)
                        nll_loss, acc = cal_nll_loss(words_logits, words_id, words_mask)

                        nll_losses.append(nll_loss.view(bsz, num_props_i))

                    nll_losses_sort, nll_loss_idx = torch.cat(nll_losses, 1).sort(dim=-1)
                    nll_losses_sort = nll_losses_sort.detach().cpu().numpy()

                    # predict temporal location
                    left = torch.cat(list(output['prop_lefts'].values()), 1).gather(index=nll_loss_idx, dim=-1)
                    right = torch.cat(list(output['prop_rights'].values()), 1).gather(index=nll_loss_idx, dim=-1)

                    gt = gt / durations[:, np.newaxis]

                    selected_props = torch.stack([left, right], dim=-1)
                    selected_props = selected_props.cpu().numpy()

                    num_all_props = selected_props.shape[1]
                    k = min(num_all_props, 5)

                    # top-1 selection strategy
                    if self.args['dataset']['name'] == 'ActivityNet':
                        c = np.ones((bsz, num_all_props))
                        votes = np.zeros((bsz, num_all_props))
                        for i in range(num_all_props):
                            for j in range(num_all_props):
                                iou = calculate_IoU((selected_props[:, i, 0], selected_props[:, i, 1]), (selected_props[:, j, 0], selected_props[:, j, 1]))
                                iou = iou * c[:, j]
                                votes[:, i] = votes[:, i] + iou
                        idx = np.argmax(votes, axis=1)
                        res = top_1_metric(selected_props[np.arange(bsz), idx], gt)
        
                    elif self.args['dataset']['name'] == 'CharadesSTA':
                        # Following CPL, on charades, the IoU of many proposals is small, and it doesn't make sense to get these proposals to vote.
                        # So we weight the voting results of each proposal according to it's IoU.
                        c = 1 - np.divide(nll_losses_sort, np.max(nll_losses_sort, axis=1, keepdims=True))
                        votes = np.zeros((bsz, num_all_props))
                        for i in range(num_all_props):
                            for j in range(num_all_props):
                                iou = calculate_IoU((selected_props[:, i, 0], selected_props[:, i, 1]), (selected_props[:, j, 0], selected_props[:, j, 1]))
                                iou = iou * c[:, j]
                                votes[:, i] = votes[:, i] + iou
                        idx = np.argmax(votes, axis=1)
                        res = top_1_metric(selected_props[np.arange(bsz), idx], gt)
                    
                    # compute result of top-n
                    for key, v in res.items():
                        metrics_logger['Rank1,'+key].update(v, bsz)
                    res = top_n_metric(selected_props[:, :k].transpose(1, 0, 2), gt)
                    for key, v in res.items():
                        metrics_logger['Rank%d,'%(k)+key].update(v, bsz)

            msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in metrics_logger.items() if k in self.key_to_vis])
            info('|'+msg+'|')

            self.evaluator = metrics_logger

            return metrics_logger

    def _build_dataset(self):
        import dataset as da
        import pickle
        from torch.utils.data import DataLoader

        args = self.args['dataset']
        cls = getattr(da, args['name'], None)
        with open(args['vocab_path'], 'rb') as fp:
            vocab = pickle.load(fp)
        self.train_set = cls(data_path=args['train_data'], vocab=vocab, args=args, is_training=True, split='train')
        self.test_set = cls(data_path=args['test_data'], vocab=vocab, args=args, split='test')
        # self.val_set = cls(data_path=args['val_data'], vocab=vocab, args=args, split='val') if args['val_data'] else None
        info('train: {} samples, test: {} samples'.format(len(self.train_set), len(self.test_set)))
        batch_size = self.args['train']['batch_size']

        def worker_init_fn(worker_id):
            def set_seed(seed):

                random.seed(seed)
                np.random.seed(seed + 1)
                torch.manual_seed(seed + 3)
                torch.cuda.manual_seed(seed + 4)
                torch.cuda.manual_seed_all(seed + 4)

            set_seed(8 + worker_id)

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                                       collate_fn=self.train_set.collate_data, num_workers=2,
                                       worker_init_fn=worker_init_fn)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False,
                                      collate_fn=self.test_set.collate_data,
                                      num_workers=0)

    def _build_model(self):
        import model

        model_config = self.args['model']
        self.model = getattr(model, model_config['name'], None)(model_config)
        self.model = self.model.cuda()
        print(self.model)
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total:', total_num, 'Trainable:', trainable_num)

    def _build_optimizer(self):
        from model.optimizer import AdamOptimizer
        from model.optimizer.lr_scheduler import InverseSquareRootSchedule

        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        args = self.args['train']["optimizer"]
        self.optimizer = AdamOptimizer(args, parameters)
        self.lr_scheduler = InverseSquareRootSchedule(args, self.optimizer)

    def _save_model(self, path):
        state_dict = {
            'num_updates': self.num_updates,
            'config': self.args,
            'model_parameters': self.model.state_dict(),
        }
        torch.save(state_dict, path)
        info('save model to {}, num_updates {}.'.format(path, self.num_updates))

    def _load_model(self, path):
        state_dict = torch.load(path)
        self.num_updates = state_dict['num_updates']
        self.lr_scheduler.step_update(self.num_updates)
        parameters = state_dict['model_parameters']
        self.model.load_state_dict(parameters)
        info('load model from {}, num_updates {}.'.format(path, self.num_updates))

    def _create_counters(self):
        self.counters = dict()
        if self.loss_meter is not None:
            for k, _ in self.loss_meter.items():
                self.counters[k] = Accumulator(k)
        if self.evaluator is not None:
            for k, _ in self.evaluator.items():
                self.counters[k] = Accumulator(k)
    
    def update_counters(self):
        if self.loss_meter is not None:
            for k, v in self.loss_meter.items():
                self.counters[k].add(v.sum_all, v.count_all)
        if self.evaluator is not None:
            for k, v in self.evaluator.items():
                self.counters[k].add(v.sum_all, v.count_all)
    
    def reset_counters(self):
        for k, v in self.counters.items():
            v.reset()


def calculate_IoU(i0, i1):
    """ 
    compute IoU
    """
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def top_n_metric(preds, label):
    """ 
    compute result of top-n
    """
    result = {}
    bsz = preds[0].shape[0]
    top_iou = []
    for pred in preds:
        iou = calculate_IoU((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
        top_iou.append(iou)
    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def top_1_metric(pred, label):
    """ 
    compute result of top-1
    """
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def apply_to_sample(f, sample):
    """ 
    apply to sample (dict(), list(), etc...)
    """
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    """ 
    move to cuda
    """
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)
