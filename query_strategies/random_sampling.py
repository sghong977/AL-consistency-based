import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, al_apply=False, cm_train_apply=False):
		super(RandomSampling, self).__init__(X, Y, idxs_lb, net, handler, args, al_apply=al_apply, cm_train_apply=cm_train_apply)

	def query(self, n):
		return np.random.choice(np.where(self.idxs_lb==0)[0], n)
