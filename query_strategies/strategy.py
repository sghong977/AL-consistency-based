import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CIFAR10 import helpers as helper

import torch.optim.lr_scheduler as lr_scheduler

import sys


class Strategy:
    def __init__(self, X, Y, idxs_lb, net, handler, args, cutsize=16, alpha=1., al_apply=False, cm_train_apply=False):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.clf = net().to(self.device)
        # for ALT
        self.cutout = helper.Cutout(1, cutsize)  # cutout function added

        self.alpha = alpha
        self.criterion = torch.nn.CrossEntropyLoss()
        self.al_apply = al_apply
        self.cm_train_apply = cm_train_apply

    # area = 0.25 (length=16)
    def getCutMix(self, src, tar, length=16):
        h = tar.size(2)
        w = tar.size(3)

        y = np.random.randint(h)   # clip set the lower and upper bound.
        x = np.random.randint(w)

        y1 = int(np.clip(y - length / 2, 0, h))
        y2 = int(np.clip(y + length / 2, 0, h))
        x1 = int(np.clip(x - length / 2, 0, w))
        x2 = int(np.clip(x + length / 2, 0, w))

        tar[:, :, y1:y2, x1:x2] = src[0, :, y1:y2, x1:x2]
        return tar

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    # DO NOT USE ALT_TRAIN AND CUTMIX REGULARIZER TOGETHER (hard to balance)
    def _train(self, epoch, loader_tr, optimizer, AL_apply=False, k_cut=2, cutmix_reg=False, s_times=1):
        self.clf.train()
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            
            # CO and CM loss
            if (AL_apply == True) and (cutmix_reg==True):
                out = self.clf(x)[0]  # batch size
                loss = F.cross_entropy(out, y)
                # CUTOUT LOSS
                _cutouts = []
                for i in range(k_cut):
                    cutout1 = self.cutout.apply(x)    # apply cutout! (randomly make a hole)
                    output1 = self.clf(cutout1)[0]
                    _cutouts.append(output1)
                target_var = y
                for i in range(k_cut):
                    loss += self.criterion(_cutouts[i], target_var)  / float(k_cut)
                    for j in range(i+1):
                        loss += self.alpha*(F.mse_loss(_cutouts[i], _cutouts[j]))  / (float(k_cut)*float(k_cut-1)/2.)
                # CUTMIX LOSS
                _cutmixes = []
                s_idx = np.random.choice(len(self.X), 1)
                _src = iter(DataLoader(self.handler(self.X[[s_idx]], self.Y[[s_idx]], transform=self.args['train_transform']), shuffle=False, **self.args['loader_tr_args']))
                for _x, _, _ in _src:
                    source = _x.to(self.device)       # torch tensor [1 3 32 32]
                for k in range(k_cut): 
                    cutmix1 = self.getCutMix(source.clone(), x.clone())          # apply mix
                    output1 = self.clf(cutmix1)[0]
                    _cutmixes.append(output1)
                target_var = y
                for i in range(k_cut):
                    loss += self.criterion(_cutmixes[i], target_var)  / float(k_cut)
                    for j in range(i+1):
                        loss += self.alpha*(F.mse_loss(_cutmixes[i], _cutmixes[j]))  / (float(k_cut)*float(k_cut-1)/2.)

            # only CO loss
            elif (AL_apply == True) and (cutmix_reg==False):
                out = self.clf(x)[0]  # batch size
                loss = F.cross_entropy(out, y)
                _cutouts = []
                for i in range(k_cut):
                    cutout1 = self.cutout.apply(x)    # apply cutout! (randomly make a hole)
                    output1 = self.clf(cutout1)[0]
                    _cutouts.append(output1)
                target_var = y
                for i in range(k_cut):
                    loss += self.criterion(_cutouts[i], target_var)  / float(k_cut)
                    for j in range(i+1):
                        loss += self.alpha*(F.mse_loss(_cutouts[i], _cutouts[j]))  / (float(k_cut)*float(k_cut-1)/2.)

            # only CM loss
            elif cutmix_reg == True:
                out = self.clf(x)[0]  # batch size
                loss = F.cross_entropy(out, y)
                _cutmixes = []

                # random source images from TRAIN SET
                s_idx = np.random.choice(len(self.X), 1)
                _src = iter(DataLoader(self.handler(self.X[[s_idx]], self.Y[[s_idx]], transform=self.args['train_transform']), shuffle=False, **self.args['loader_tr_args']))
                for _x, _, _ in _src:
                    source = _x.to(self.device)       # torch tensor [1 3 32 32]
                # calculate entropy 'k_cut' times
                for k in range(k_cut): 
                    cutmix1 = self.getCutMix(source.clone(), x.clone())          # apply mix
                    output1 = self.clf(cutmix1)[0]
                    _cutmixes.append(output1)
                target_var = y
                for i in range(k_cut):
                    loss += self.criterion(_cutmixes[i], target_var)  / float(k_cut)
                    for j in range(i+1):
                        loss += self.alpha*(F.mse_loss(_cutmixes[i], _cutmixes[j]))  / (float(k_cut)*float(k_cut-1)/2.)
            else:
                out = self.clf(x)[0]
                loss = F.cross_entropy(out, y)
            
                # backprop
            loss.backward()
            optimizer.step()

    def train(self):
        n_epoch = self.args['n_epoch']
        self.clf = self.net().to(self.device)  # if pretrained, self.net.to(self.device)
        optimizer = optim.SGD(self.clf.parameters(), **self.args['optimizer_args'])
        if self.args['dataname'] == 'CIFAR100':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args['drop'], gamma=0.1)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        
        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args['train_transform']),
                            shuffle=True, **self.args['loader_tr_args'])

        for epoch in range(1, n_epoch+1):
            self._train(epoch, loader_tr, optimizer, AL_apply=self.al_apply, cutmix_reg=self.cm_train_apply)
            scheduler.step()   # lr drop

    def predict(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                            shuffle=False, **self.args['loader_te_args'])
        
        self.clf.eval()
        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)[0]

                pred = out.max(1)[1]
                P[idxs] = pred.cpu()

        return P

# for top5 error (logging)
    def predict_(self, X, Y, name=''):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                            shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        n_class = 10
        if self.args['dataname'] == 'CIFAR100':
            n_class = 100
        P = torch.zeros((len(Y), n_class), dtype=torch.float32)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)[0]
                P[idxs] = out.cpu()

        return P

    ## Implementing CutMix Entropy
    def predict_Mix(self, X, Y, src, n_src=10, k_c=5, ul_src=False):
        labled_flag = 0
        if ul_src is False:
            labled_flag = 1
            
        is_src_rand = False
        if src is None:
            # optino 2 (random source, 10 times * 2 cutout)
            s_times = n_src
            k_cut = k_c
            is_src_rand = True
        elif len(src) == 1 :
            # option 0 (single source)
            s_times = 1
        elif len(src) > 1:
            # option 1 (5 sources)
            s_times = n_src
            k_cut = k_c

        # to generate scores of unlabeled_data
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                            shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()

        P = torch.zeros(len(Y), dtype=torch.float32)    # k_cut?
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                # """               
                tmp = []
                # change source image 's_times'
                for s_t in range(s_times):
                    if is_src_rand is True:
                        # option 2 : random source images
                        if labled_flag is 0:
                            s_idx = np.random.choice(len(X), 1)
                            _src = iter(DataLoader(self.handler(X[[s_idx]], Y[[s_idx]], transform=self.args['test_transform']), shuffle=False, **self.args['loader_te_args']))
                        else:
                            s_idx = np.random.choice(np.where(self.idxs_lb==labled_flag)[0], 1)   # pick one from labeled data
                            _src = iter(DataLoader(self.handler(self.X[[s_idx]], self.Y[[s_idx]], transform=self.args['train_transform']), shuffle=False, **self.args['loader_tr_args']))
                        for _x, _, _ in _src:
                            source = _x       # torch tensor [1 3 32 32]
                    elif len(src) > 1:
                        source = src[s_t]
                    else:
                        source = src
                    # calculate entropy 'k_cut' times
                    for i in range(k_cut):
                        cutmix1 = self.getCutMix(source, x)    # apply mix
                        output1 = self.clf(cutmix1)[0]
                        prob = F.softmax(output1, dim=1) # apply softmax
                        log_prob = torch.log(prob)
                        ent = (prob*log_prob)
                        tmp.append(ent.cpu())    # 1000 * 10
                # """

                # calculate scores
                P[idxs] += torch.FloatTensor(sum(tmp)).sum(1) / (k_cut * s_times)     

        return P

    ## TODO
    def predict_ALT(self, X, Y, k_cut=5):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        P = torch.zeros(len(Y), dtype=torch.float32)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                # 5 cutout
                cutouts = []
                for i in range(k_cut):
                    cutout1 = self.cutout.apply(x)    # apply cutout! (randomly make a hole)
                    output1 = self.clf(cutout1)[0]

                    cutouts.append(output1)
                loss = 0
                for i in range(k_cut):
                    for j in range(i+1):
                        loss += self.alpha*(F.mse_loss(cutouts[i], cutouts[j]))  / (float(k_cut)*float(k_cut-1)/2.)
                P[idxs] = loss.cpu().float()

        return P


    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['test_transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)[0]  # delete e1
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        
        return probs
