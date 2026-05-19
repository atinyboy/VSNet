import pickle

import numpy as np
import json
import random
import math

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from feeders import tools

flip_index = np.concatenate((range(0, 21),range(21, 42)), axis=0)

class Feeder(Dataset):
    def __init__(self, data_path, num_cls, data_type='j', repeat=1, partition=False):

        if 'test' in data_path:
            self.train_val = 'val'
        else:
            self.train_val = 'train'

        self.num_cls = num_cls
        self.time_steps = 64
        self.bone0 = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),(9,10), (10, 11),(12, 11), (13,14),(14,15),(15,16),
         (17,18),(18,19),(19,20),(0,17),(17,13),(13,9),(9,5)])
        self.bone = np.concatenate((self.bone0, self.bone0 + 21))

        self.label = []
        self.data = []
        self.data_path = data_path
        self.data_type = data_type
        self.partition = partition
        self.repeat = repeat
        self.load_data()

        if self.partition:
            self.left_arm = np.array([0, 17, 18, 19, 20])
            self.right_arm = self.left_arm + 21
            self.finger1 = np.array(range(1, 9))
            self.finger2 = np.array(range(9, 17))
            self.finger3 = self.finger1 + 21
            self.finger4 = self.finger2 + 21
            self.new_idx = np.concatenate((self.left_arm, self.right_arm, self.finger1, self.finger2, self.finger3, self.finger4),
                                          axis=-1)

    def load_data(self):
        # data: N C V T M
        with open(self.data_path, 'rb') as f:
            data_file = pickle.load(f)
        for data in data_file:
            if int(data['label']) < self.num_cls:
                # kp = [(x if x else [[0,0,0]]*21) + (y if y else [[0,0,0]]*21) for x,y in zip(data['hand1'], data['hand2'])]
                self.data.append(data['data'])
                self.label.append(data['label'])

        self.sample_name = [f'{self.train_val}_' + str(i) for i in range(len(self.data))]

    def __len__(self):
        return len(self.label) * self.repeat

    def __iter__(self):
        return self

    # 应用旋转和缩放
    def rand_view_transform(self, X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1, 0, 0], [0, math.cos(agx), math.sin(agx)], [0, -math.sin(agx), math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0, 1, 0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])
        X0 = np.dot(np.reshape(X, (-1, 3)), np.dot(Ry, np.dot(Rx, Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def __getitem__(self, index):
        label = self.label[index % len(self.label)]
        value = self.data[index % len(self.label)]
        T = value.shape[0]
        value = value[T // 10:T - T // 10, :, :]

        data_numpy = np.array(value)
        data_numpy = np.transpose(data_numpy, (2, 0, 1))
        # C, T, V = data_numpy.shape
        # data_numpy = np.reshape(data_numpy, (C, T, V, 1))

        if self.train_val == 'train':
            random.random()
            # data_numpy = tools.random_choose(data_numpy, self.window_size)

            # mirror
            if random.random() > 0.5:
                assert data_numpy.shape[2] == 42
                data_numpy = data_numpy[:, :, flip_index]
                data_numpy[0, :, :] = 512 - data_numpy[0, :, :]

            center = np.reshape(data_numpy[:, 5, 0], (3, 1, 1))
            data_numpy = data_numpy - center

            # normalization:
            # data_numpy = (data_numpy - self.mean_map) / self.std_map
            assert data_numpy.shape[0] == 3
            data_numpy[0, :, :] = data_numpy[0, :, :] - data_numpy[0, :, 0].mean(axis=0)
            data_numpy[1, :, :] = data_numpy[1, :, :] - data_numpy[1, :, 0].mean(axis=0)

            # random_shift:
            data_numpy[0, :, :] += random.random() * 20 - 10.0
            data_numpy[1, :, :] += random.random() * 20 - 10.0

            C, T, V = data_numpy.shape
            data_numpy = np.reshape(data_numpy, (C, T, V, 1))  # C T V M

            # random_move:
            data_numpy = tools.random_move(data_numpy)

            data = np.zeros((3, self.time_steps, 42, 1))

            length = data_numpy.shape[1]

            random_idx = random.sample(list(np.arange(length)) * 100, self.time_steps)
            random_idx.sort()
            data[:, :, :, :] = data_numpy[:, random_idx, :, :]  # C, T, V, M
            index_t = 2 * np.array(random_idx).astype(np.float32) / length - 1

        else:
            center = np.reshape(data_numpy[:, 5, 0], (3, 1, 1))
            data_numpy = data_numpy - center

            assert data_numpy.shape[0] == 3
            data_numpy[0, :, :] = data_numpy[0, :, :] - data_numpy[0, :, 0].mean(axis=0)
            data_numpy[1, :, :] = data_numpy[1, :, :] - data_numpy[1, :, 0].mean(axis=0)

            C, T, V = data_numpy.shape
            data_numpy = np.reshape(data_numpy, (C, T, V, 1))  # C T V M

            data = np.zeros((3, self.time_steps, 42, 1))

            length = data_numpy.shape[1]

            idx = np.linspace(0, length - 1, self.time_steps).astype(int)
            data[:, :, :, :] = data_numpy[:, idx, :, :]
            index_t = 2 * idx.astype(np.float32) / length - 1

        if 'b' in self.data_type:
            data_bone = np.zeros_like(data)
            for bone_idx in range(42):
                data_bone[:,:, self.bone[bone_idx][0], :] = data[:,:, self.bone[bone_idx][0], :] - data[:,:, self.bone[bone_idx][1], :]
            data = data_bone

        if 'm' in self.data_type:
            data_motion = np.zeros_like(data)
            data_motion[:,:-1, :, :] = data[:,1:, :, :] - data[:,:-1, :, :]
            data = data_motion

        if self.partition:
            data = data[:, :, self.new_idx, :]

        return data, index_t, label, index



    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    def init_seed(seed):
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    train_feeder_args = {
    'data_path': r'./data/slovo/test_1000.pkl',
    'num_cls': 100,
    'data_type': 'j',
    'repeat': 1,
    'partition': True}
    import os

    current_path = os.getcwd()
    os.chdir(os.path.join(current_path, '..'))
    data_loader = torch.utils.data.DataLoader(
        dataset=Feeder(**train_feeder_args),
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=True)
    # dataset = Feeder(**train_feeder_args)
    process = tqdm(data_loader)

    a = []
    for batch_idx, (data, index_t, label, index) in enumerate(process):
        a.append(data)
    a
