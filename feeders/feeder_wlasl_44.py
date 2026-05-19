import pickle

import numpy as np
import json
import random
import math

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from feeders import tools

flip_index = np.concatenate(([0,2,1,4,3],[26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46],
                             [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,48,47]), axis=0)

class Feeder(Dataset):
    def __init__(self, data_path, num_cls, data_type='j', repeat=1):

        with open(r'data/WLASL/WLASL_v0.3.json', 'r') as f1:
            wlasl = json.load(f1)

        data_dict = []
        for i, lab in enumerate(wlasl[:num_cls]):
            for ins in lab['instances']:
                if ins['split'] in data_path:
                    data_dict.append({'file_name': ins['video_id'], 'label': i})

        self.data_dict = data_dict

        if 'test' in data_path:
            self.train_val = 'val'
        else:
            self.train_val = 'train'

        self.time_steps = 32
        self.bone = np.array(
            [[0, 0],[1, 0], [2, 0], [3, 1], [4, 2], [5, 3], [6, 5], [7, 6],[8, 7], [9, 8],[10, 6],
             [11, 10], [12, 11], [13, 12], [14, 6], [15, 14], [16, 15], [17, 16],[18, 6],  [19, 18], [20, 19], [21, 20], [22, 6],[23, 22],
             [24, 23], [25, 24],[26, 4], [27, 26], [28, 27],  [29, 28], [30, 29],  [31, 27], [32, 31], [33, 32],[34, 33], [35, 27],[36, 35],
             [37, 36], [38, 37], [39, 27], [40, 39], [41, 40], [42, 41],[43, 27],  [44, 43], [45, 44], [46, 45], [47, 0], [48, 0]])

        self.label = []

        self.data_path = data_path
        self.data_type = data_type
        self.repeat = repeat
        self.load_data()

        self.left_arm = np.array([1,3,5,7,8,9])
        self.right_arm = np.array([2, 4,26,28,29,30])
        self.finger1 = np.array([10, 11, 12, 13,14, 15, 16, 17])
        self.finger2 = np.array([18, 19, 20, 21,22, 23, 24, 25])
        self.finger3 = np.array([10, 11, 12, 13,14, 15, 16, 17]) + 21
        self.finger4 = np.array([18, 19, 20, 21,22, 23, 24, 25]) + 21
        self.new_idx = np.concatenate((self.left_arm, self.right_arm, self.finger1, self.finger2, self.finger3, self.finger4),
                                      axis=-1)


    def load_data(self):
        # data: N C V T M
        self.data = []
        with open(self.data_path, 'rb') as f:
            data_file = pickle.load(f)
        for data in self.data_dict:
            file_name = data['file_name']
            if file_name in data_file:
                value = np.array(data_file[file_name])
                self.data.append(value)
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
                assert data_numpy.shape[2] == 49
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

            if 'b' in self.data_type:
                data_bone = np.zeros_like(data_numpy)
                for bone_idx in range(49):
                    data_bone[:, :, self.bone[bone_idx][0], :] = data_numpy[:, :, self.bone[bone_idx][0], :] - data_numpy[:, :,
                                                                                                             self.bone[
                                                                                                             bone_idx][
                                                                                                             1], :]
                data_numpy = data_bone

            if 'm' in self.data_type:
                data_motion = np.zeros_like(data_numpy)
                data_motion[:, :-1, :, :] = data_numpy[:, 1:, :, :] - data_numpy[:, :-1, :, :]
                data_numpy = data_motion

            data = np.zeros((3, self.time_steps, 49, 1))

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

            if 'b' in self.data_type:
                data_bone = np.zeros_like(data_numpy)
                for bone_idx in range(49):
                    data_bone[:, :, self.bone[bone_idx][0], :] = data_numpy[:, :, self.bone[bone_idx][0], :] - data_numpy[:, :,
                                                                                                                    self.bone[
                                                                                                                    bone_idx][
                                                                                                                    1], :]
                data_numpy = data_bone

            if 'm' in self.data_type:
                data_motion = np.zeros_like(data_numpy)
                data_motion[:, :-1, :, :] = data_numpy[:, 1:, :, :] - data_numpy[:, :-1, :, :]
                data_numpy = data_motion

            data = np.zeros((3, self.time_steps, 49, 1))

            length = data_numpy.shape[1]

            idx = np.linspace(0, length - 1, self.time_steps).astype(int)
            data[:, :, :, :] = data_numpy[:, idx, :, :]
            index_t = 2 * idx.astype(np.float32) / length - 1

        # if 'b' in self.data_type:
        #     data_bone = np.zeros_like(data)
        #     for bone_idx in range(49):
        #         data_bone[:,:, self.bone[bone_idx][0], :] = data[:,:, self.bone[bone_idx][0], :] - data[:,:, self.bone[bone_idx][1], :]
        #     data = data_bone
        #
        # if 'm' in self.data_type:
        #     data_motion = np.zeros_like(data)
        #     data_motion[:,:-1, :, :] = data[:,1:, :, :] - data[:,:-1, :, :]
        #     data = data_motion

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
        'data_path': './data/WLASL/test_100.pkl',
        'num_cls': 100,
        'data_type': 'j',
        'repeat': 1}
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
