# my implementation
# YEAH!
import numpy as np
from .strategy import Strategy
import torch

from torch.utils.data import DataLoader
from CIFAR10 import helpers as helper


class CutMixEntropySampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, al_apply, cm_train_apply, alpha=1.):
		super(CutMixEntropySampling, self).__init__(X, Y, idxs_lb, net, handler, args, cm_train_apply=cm_train_apply, al_apply=al_apply)

	def query(self, pick=10, options=2, n_src=10, k_cut=2, ul_src=False):
		print("CutMixEntropySampling query!\n")
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

		# 1. pick source
		source = None
		# AL_OPTION 0 - single source image (deterministic)
		if (options == 0):
			s_idx = np.random.choice(np.where(self.idxs_lb==1)[0], 1)   # pick one from labeled data
			source = iter(DataLoader(self.handler(self.X[[s_idx]], self.Y[[s_idx]], transform=self.args['train_transform']), shuffle=False, **self.args['loader_tr_args']))
			for x, y, idxs in source:
				source = x       # torch tensor [1 3 32 32]
		# options 1 - N source images (deterministic)
		elif (options == 1):
			s_idx = np.random.choice(np.where(self.idxs_lb==1)[0], 5)   # pick one from labeled data
			src = iter(DataLoader(self.handler(self.X[s_idx], self.Y[s_idx], transform=self.args['train_transform']), shuffle=False, **self.args['loader_tr_args']))
			source = []
			for x, y, idxs in src:
				source.append(x)       # torch tensor [1 3 32 32]		
		# option 2 - totally random source images
		elif (options == 2):
			source = None
			
		# 2. calculate cutmix softmax, entropy, then sort them
		# probs [data_idx][times][probs]
		probs = self.predict_Mix(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], src=source, n_src=n_src, k_c=k_cut, ul_src=ul_src)

		return idxs_unlabeled[probs.sort()[1][:pick]]
