# ALT : k cutout

import numpy as np
from .strategy import Strategy
import torch


class CutoutSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, alpha=1., al_apply=False, cm_train_apply=False):
		super(CutoutSampling, self).__init__(X, Y, idxs_lb, net, handler, args, al_apply=al_apply, alpha=alpha, cm_train_apply=cm_train_apply)

	def query(self, pick):
		print("CutoutSampling query!\n")
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_ALT(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		return idxs_unlabeled[probs.sort(descending=True)[1][:pick]]
    