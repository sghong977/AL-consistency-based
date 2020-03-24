import numpy as np
from dataset import get_dataset, get_handler
from model import get_net
from torchvision import transforms
import torch
import time
import sys
from query_strategies import    RandomSampling, MarginSampling, EntropySampling, \
                                CutoutSampling, CutMixEntropySampling

import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#------------ commandline parameters ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--pick', type=int, required=True)
parser.add_argument('--round', type=int, required=True)
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--initnum', type=int, required=True)

parser.add_argument('--train_al', type=bool, default=False)   # if true, training loss contains uncertainty value.
parser.add_argument('--train_cm', type=bool, default=False)   # if true, training loss contains uncertainty value.

parser.add_argument('--SEED', type=int, default=1)   # model SEED
parser.add_argument('--drop', type=int, default=160) # starting epoch of lr drop
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--augdisabled', type=bool, default=False)
parser.add_argument('--pretrained_path', type=str)

parser.add_argument('--ALtype', type=str, default='Random')  # choose Active learning type
parser.add_argument('--alpha', type=float, default=1.)   # alpha. cutout scaling

# CutoutSampling, CutMixEntropy
parser.add_argument('--k_cut', type=int)   # # of cutout num
parser.add_argument('--n_src', type=int)   # # of source images

args = parser.parse_args()

# --------------------- parameters -----------------------
NUM_INIT_LB = args.initnum    # # of labeled pool
NUM_ROUND = args.round      # 0801 500
NUM_QUERY = args.pick     # Actual # of query data. must be smaller than NUM_QUERY

AL_type = args.ALtype

# CutoutSampling or CutMix
K_cutout = args.k_cut
N_src = args.n_src
unlabeled_src = True    ############

# train settings
n_epoch = args.epoch
al_train_apply = args.train_al
cm_train_apply = args.train_cm  # Cutmix regularizer
SEED = args.SEED
drop = args.drop
DATA_NAME = args.dataset
num_classes = 10
selected_model = args.model
is_augmentation = not(args.augdisabled) # default : includes data augmentation
pretrained_path = args.pretrained_path
alpha=args.alpha

#-------------------- argpool -----------
args_pool = {'FashionMNIST':
                {'n_epoch': n_epoch, 'train_transform': transforms.Compose([
                    transforms.RandomCrop(32, padding=4), #
                    transforms.RandomHorizontalFlip(),      #
                    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                'test_transform': transforms.Compose([
                    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 128, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.05, 'momentum': 0.5, 'weight_decay':5e-4},
                 'drop':[drop],  'dataname':'FashionMNIST'},
            'CIFAR10':
                {'n_epoch': n_epoch, 'train_transform': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                'test_transform': transforms.Compose([
                    transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 128, 'num_workers': 2},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 2},
                 'optimizer_args':{'lr': 0.1, 'momentum': 0.9,  'weight_decay':5e-4},
                 'drop':[drop],  'dataname':'CIFAR10'},
            }
# Learning loss args_pool
args_pool[DATA_NAME]['method'] = AL_type

#before augmentation
if not is_augmentation:
    args_pool[DATA_NAME]['train_transform'] = args_pool[DATA_NAME]['test_transform']

args = args_pool[DATA_NAME]


# -------------------  set seed -------------------------
#np.random.seed(SEED)   # 
torch.manual_seed(SEED)
torch.backends.cudnn.enabled = True   # random seed    
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# load dataset
X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_lb[0:NUM_INIT_LB] = True

# load network
net = get_net(DATA_NAME, model=selected_model)
handler = get_handler(DATA_NAME)

print(net)

#-- Strategy --
if AL_type == 'Random':
    strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args, al_apply=al_train_apply, cm_train_apply=cm_train_apply)
elif AL_type == 'CutoutSampling':
    strategy = CutoutSampling(X_tr, Y_tr, idxs_lb, net, handler, args, al_apply=al_train_apply, alpha=alpha, cm_train_apply=cm_train_apply)  # ALT
elif AL_type == 'CutMixEntropy':
    strategy = CutMixEntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args, al_apply=al_train_apply, cm_train_apply=cm_train_apply) #
elif AL_type == 'MarginSampling':
    strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args,  al_apply=al_train_apply, cm_train_apply=cm_train_apply)
elif AL_type == 'EntropySampling':
    strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args,  al_apply=al_train_apply, cm_train_apply=cm_train_apply)
else:
    print('choose another sampling strategy')
    sys.exit()

########################## LOG : PRINT SETTINGS #########################
print("\n\n###----------- FINAL SETTING -----------###")
print("Dataset : ", DATA_NAME)
print("Epoch per round : ", n_epoch)
print("Drop : ", drop)
print('# of initial pool: {}'.format(NUM_INIT_LB))
print('# of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
print('# of testing pool: {}'.format(n_test))
print("Cutout train : ", al_train_apply)
print("CutMix train : ", cm_train_apply)
print('SEED {}'.format(SEED))
print('Active Learning Strategy : ', AL_type)
print("Num query : ", NUM_QUERY)
print("Num Round : ", NUM_ROUND)
print("Is Augmentation : ", is_augmentation)
print("Model:", selected_model)
if AL_type == 'CutMixEntropy':
    print("K_cutout and N_source images : ", K_cutout, ', ' , N_src)
    print("\nUnlabeled source image : ",unlabeled_src,"\n")

# top K accuracy
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).numpy()[0])
        return res

strategy.clf.load_state_dict(torch.load(pretrained_path))

P = strategy.predict_(X_te, Y_te, name=AL_type)            # predict
acc = np.zeros((NUM_ROUND+1, 2))
acc[0] = accuracy(P, Y_te, topk=(1, 5))
print('Round 0\ntesting accuracy {}'.format(acc[0], ".5f"))

##----------------- ROUND N ----------------------- ##
for rd in range(1, NUM_ROUND+1):
    print('Round {}'.format(rd))

    # query
    if AL_type == 'CutMixEntropy':
        q_idxs = strategy.query(pick=NUM_QUERY, k_cut=K_cutout, n_src=N_src, ul_src=unlabeled_src)
    else:
        q_idxs = strategy.query(NUM_QUERY)

    idxs_lb[q_idxs] = True

    # update
    strategy.update(idxs_lb)
    strategy.train()

    # round accuracy
    P = strategy.predict_(X_te, Y_te, name=AL_type)
    acc[rd] = accuracy(P, Y_te, topk=(1,5)) #for top 5 acc, set topk=(1,5)
    print('testing accuracy {}'.format(acc[rd], ".5f"))

# print results
print('SEED {}'.format(SEED))
print(type(strategy).__name__)
print(acc)

# save as file
timestr = "./results/" + DATA_NAME + time.strftime("%Y%m%d-%H%M%S")
with open(timestr+AL_type+'_acc.txt', 'w') as f:
    for a in acc:
        f.write("%s\n" % round(a[0],5))                                                                              
with open(timestr+AL_type+'_acc_top5.txt', 'w') as f:
    for a in acc:
        f.write("%s\n" % round(a[1],5))

with open(timestr+'info.txt', 'w') as f:
    f.write("Dataset:" +'\t\t\t\t\t'+ DATA_NAME + '\n')
    f.write("Num Round : "+'\t\t\t\t'+ str(NUM_ROUND) + '\n')
    f.write("Init data : " +'\t\t\t\t'+ str(NUM_INIT_LB) + '\n')
    f.write("n_epoch (for each round) : " +'\t'+ str(n_epoch) + '\n')
    f.write("DROP : " +'\t\t\t\t\t\t'+ str(drop) + '\n')
    f.write("SEED : "+'\t\t\t\t\t\t'+ str(SEED) + '\n')
    f.write("AL train : "+'\t\t\t\t\t'+ str(al_train_apply) + '\n')
    f.write("Cutmix train : "+'\t\t\t\t\t'+ str(cm_train_apply) + '\n')
    f.write("Is Augmentation : " +'\t\t\t'+ str(is_augmentation)+ '\n')
    f.write("Model : " +'\t\t\t\t\t'+ selected_model + '\n')
    f.write('Active Learning Strategy :' +'\t'+ AL_type+'\n')
    f.write("NUM_QUERY : "+'\t\t\t'+ str(NUM_QUERY) + '\n')
    print(acc[:,0])
    for a in acc:
        f.write("%s\n" % round(a[0],5)) 
        