import copy, argparse
import numpy as np
import math
from collections import Counter
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from . import medmnist_class
import os 

def make_data_loader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, pin_memory=True)


class SplitDataset(Dataset):
    def __init__(self, dataset, idx):
        super().__init__()
        self.dataset = dataset
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        input = self.dataset[self.idx[index]]
        return input

def fetch_dataset(data_name):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if not os.path.isdir(root):
        os.makedirs(root)
    if data_name in medmnist_class.medmnist_classes :
        dataset['train'] = medmnist_class.medmnist_classes[data_name](root=root, split='train', download=True, transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])]))
        dataset['train'].target = dataset['train'].labels.squeeze().tolist()

        dataset['val'] = medmnist_class.medmnist_classes[data_name](root=root, split='train', download=True, transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])]))
        dataset['val'].target = dataset['val'].labels.squeeze().tolist()

        dataset['test'] = medmnist_class.medmnist_classes[data_name](root=root, split='test', download=True, transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])]))
        dataset['test'].target = dataset['test'].labels.squeeze().tolist()

    else:
        raise ValueError('Not valid dataset name')
    return dataset

def split_dataset(dataset, num_users, data_split_mode, args):
    if data_split_mode == 'iid':
        data_splits = iid(dataset['train'], num_users)
    elif 'non-iid' in data_split_mode:
        data_splits = non_iid(dataset['train'], num_users, data_split_mode)
    else:
        raise ValueError('Not valid data split mode')
    dataloaders = [make_data_loader(SplitDataset(dataset['train'], data_split), args.batch) for data_split in data_splits]
    return dataloaders, make_data_loader(dataset['val'], args.test_batch), make_data_loader(dataset['test'], args.test_batch)


def iid(dataset, num_users):
    label = torch.tensor(dataset.target)

    d_idxs = np.random.permutation(len(dataset))
    local_datas = np.array_split(d_idxs, num_users)
    data_split = []

    for i in range(num_users):
        data_split.append(local_datas[i])

    return data_split


def non_iid(dataset, num_users, data_split_mode):
    label = np.array(dataset.target)
    skew = float(data_split_mode.split('-')[-1])

    K = len(set(label))
    dpairs = [[did, dataset[did][1]] for did in range(len(dataset))]

    MIN_ALPHA = 0.01
    alpha = (-4*np.log(skew + 10e-8))**4
    alpha = max(alpha, MIN_ALPHA)
    labels = [pair[-1] for pair in dpairs]
    lb_counter = Counter(labels)
    p = np.array([1.0*v/len(dpairs) for v in lb_counter.values()])
    lb_dict = {}
    labels = np.array(labels)
    for lb in range(len(lb_counter.keys())):
        lb_dict[lb] = np.where(labels==lb)[0]
    proportions = [np.random.dirichlet(alpha*p) for _ in range(num_users)]
    while np.any(np.isnan(proportions)):
        proportions = [np.random.dirichlet(alpha * p) for _ in range(num_users)]
    while True:
        # generate dirichlet distribution till ||E(proportion) - P(D)||<=1e-5*self.num_classes
        mean_prop = np.mean(proportions, axis=0)
        error_norm = ((mean_prop-p)**2).sum()
        print("Error: {:.8f}".format(error_norm))
        if error_norm<=1e-2/K:
            break
        exclude_norms = []
        for cid in range(num_users):
            mean_excid = (mean_prop*num_users-proportions[cid])/(num_users-1)
            error_excid = ((mean_excid-p)**2).sum()
            exclude_norms.append(error_excid)
        excid = np.argmin(exclude_norms)
        sup_prop = [np.random.dirichlet(alpha*p) for _ in range(num_users)]
        alter_norms = []
        for cid in range(num_users):
            if np.any(np.isnan(sup_prop[cid])):
                continue
            mean_alter_cid = mean_prop - proportions[excid]/num_users + sup_prop[cid]/num_users
            error_alter = ((mean_alter_cid-p)**2).sum()
            alter_norms.append(error_alter)
        if len(alter_norms)>0:
            alcid = np.argmin(alter_norms)
            proportions[excid] = sup_prop[alcid]
    local_datas = [[] for _ in range(num_users)]
    # self.dirichlet_dist = [] # for efficiently visualizing
    for lb in lb_counter.keys():
        lb_idxs = lb_dict[lb]
        lb_proportion = np.array([pi[lb] for pi in proportions])
        lb_proportion = lb_proportion/lb_proportion.sum()
        lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
        lb_datas = np.split(lb_idxs, lb_proportion)
        # self.dirichlet_dist.append([len(lb_data) for lb_data in lb_datas])
        local_datas = [local_data+lb_data.tolist() for local_data,lb_data in zip(local_datas, lb_datas)]
    # self.dirichlet_dist = np.array(self.dirichlet_dist).T
    for i in range(num_users):
        np.random.shuffle(local_datas[i])        

    data_split = []

    for i in range(num_users):
        data_split.append(local_datas[i])

    return data_split



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed=None):
    import random
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, val):
        self.values.append(val)
        self.counter += 1

    def extend(self, items):
        self.values.extend(items)
        self.counter += len(items)

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        values = self.values
        if len(values) > 0:
            return ','.join([f" {metric}: {eval(f'np.{metric}')(values)}"
                             for metric in ['mean', 'std', 'min', 'max']])
        else:
            return 'empy meter'

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


class LocalMaskCrossEntropyLoss(nn.CrossEntropyLoss):
    """Should be used for class-wise non-iid.
    Refer to HeteroFL (https://openreview.net/forum?id=TNkPBBYFkXg)
    """
    def __init__(self, num_classes, **kwargs):
        super(LocalMaskCrossEntropyLoss, self).__init__(**kwargs)
        self.num_classes = num_classes
        
    def forward(self, input, target):
        classes = torch.unique(target)
        mask = torch.zeros_like(input)
        for c in range(self.num_classes):
            if c in classes:
                mask[:, c] = 1  # select included classes
        return F.cross_entropy(input*mask, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


# ///////////// samplers /////////////
class _Sampler(object):
    def __init__(self, arr):
        self.arr = copy.deepcopy(arr)

    def next(self):
        raise NotImplementedError()


class shuffle_sampler(_Sampler):
    def __init__(self, arr, rng=None):
        super().__init__(arr)
        if rng is None:
            rng = np.random
        rng.shuffle(self.arr)
        self._idx = 0
        self._max_idx = len(self.arr)

    def next(self):
        if self._idx >= self._max_idx:
            np.random.shuffle(self.arr)
            self._idx = 0
        v = self.arr[self._idx]
        self._idx += 1
        return v


class random_sampler(_Sampler):
    def next(self):
        # np.random.randint(0, int(1 / slim_ratios[0]))
        v = np.random.choice(self.arr)  # single value. If multiple value, note the replace param.
        return v


class constant_sampler(_Sampler):
    def __init__(self, value):
        super().__init__([])
        self.value = value

    def next(self):
        return self.value


# ///////////// lr schedulers /////////////
class CosineAnnealingLR(object):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, T_max, eta_max=1e-2, eta_min=0, last_epoch=0, warmup=None):
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self._cur_lr = eta_max
        self._eta_max = eta_max
        # super(CosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)
        self.warmup = warmup

    def step(self):
        self._cur_lr = self._get_lr()
        self.last_epoch += 1
        return self._cur_lr

    def _get_lr(self):
        if self.warmup is not None and self.warmup > 0:
            if self.last_epoch < self.warmup:
                return self._eta_max * ((self.last_epoch+1e-2) / self.warmup)
            elif self.last_epoch == self.warmup:
                return self._eta_max
        if self.last_epoch == 0:
            return self.eta_max
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self._cur_lr + (self.eta_max - self.eta_min) * \
                    (1 - math.cos(math.pi / self.T_max)) / 2
        return (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / \
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) * \
                (self._cur_lr - self.eta_min) + self.eta_min


class MultiStepLR(object):
    def __init__(self, eta_max, milestones, gamma=0.1, last_epoch=-1, warmup=None):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.last_epoch = last_epoch
        self._cur_lr = eta_max
        self._eta_max = eta_max
        # super(MultiStepLR, self).__init__(optimizer, last_epoch, verbose)
        self.warmup = warmup

    def step(self):
        self._cur_lr = self._get_lr()
        self.last_epoch += 1
        return self._cur_lr

    def _get_lr(self):
        if self.warmup is not None and self.warmup > 0:
            if self.last_epoch < self.warmup:
                return self._eta_max * ((self.last_epoch+1e-3) / self.warmup)
            elif self.last_epoch == self.warmup:
                return self._eta_max
        if self.last_epoch not in self.milestones:
            return self._cur_lr
        return self._cur_lr * self.gamma ** self.milestones[self.last_epoch]


def test_lr_sch(sch_name='cos'):
    lr_init = 0.1
    T = 150
    if sch_name == 'cos':
        sch = CosineAnnealingLR(T, lr_init, last_epoch=0, warmup=5)
    elif sch_name == 'multi_step':
        sch = MultiStepLR(lr_init, [50, 100], last_epoch=0, warmup=5)

    for step in range(150):
        lr = sch.step()
        if step % 20 == 0 or step < 20:
            print(f"[{step:3d}] lr={lr:.4f}")

    # resume
    print(f"Resume from step{step} with lr={lr:.4f}")
    T = 300
    if sch_name == 'cos':
        sch = CosineAnnealingLR(T, lr_init, last_epoch=step)
    elif sch_name == 'multi_step':
        sch = MultiStepLR(lr_init, [2, 4, 4, 50], last_epoch=step)
    for step in range(step, step+10):
        lr = sch.step()
        print(f"[{step:3d}] lr={lr:.4f}")


if __name__ == '__main__':
    test_lr_sch('cos')
