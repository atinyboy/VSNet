import sys

import numpy as np

from graph.tools import get_groups

sys.path.extend(['../'])
from graph import tools


def get_new_graph(graph=1):
    if graph==1:
        new_graph = np.array([0,1,2,4,5,6,7,8,10,11,12,14,15,16, 18, 19, 20, 22,
                               23,24,26,27,28,30,31,32,34,35,36, 38, 39, 40,42, 43])
    elif graph==2:
        new_graph = np.array([0,1,2,4,5,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31,33])
    else:
        raise ValueError
    return new_graph

class Graph:
    def __init__(self, labeling_mode='spatial', graph='wlasl'):

        if graph == 'wlasl_24':
            num_node = 24
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (6, 7), (8, 9), (10, 11), (2, 8), (2, 10), (2, 12),
                                (12, 13), (2, 14), (14, 15), (16, 17), (6, 16), (18, 19), (6, 18), (20, 21), (6, 20), (22, 23), (6, 22)]

            inward = inward_ori_index
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward
        elif graph == 'wlasl_48':
            num_node = 48
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [(1, 2), (5, 6), (2, 3), (6, 7), (3, 8), (8, 9), (8, 16), (8, 24), (8, 32), (8, 40), (9, 10), (10, 11),
                                (16, 17), (17, 18), (18, 19), (24, 25), (25, 26), (26, 27), (32, 33), (33, 34), (34, 35), (40, 41),
                                (41, 42), (42, 43), (7, 12), (12, 13), (12, 20), (12, 28), (12, 36), (12, 44), (13, 14), (14, 15),
                                (20, 21), (21, 22), (22, 23), (28, 29), (29, 30), (30, 31), (36, 37), (37, 38), (38, 39), (44, 45),
                                (45, 46), (46, 47), (0, 1), (4, 5)]

            inward = inward_ori_index
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward

        elif graph == 'wlasl_36':
            num_node = 36
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [(0, 1), (6, 7), (1, 2), (7, 8), (2, 3), (2, 12), (2, 15), (2, 18), (2, 21), (3, 4), (4, 5),
                                (12, 13), (15, 16), (18, 19), (21, 22), (8, 9), (8, 24), (8, 27), (8, 30), (8, 33), (9, 10),
                                (10, 11), (24, 25), (27, 28), (13, 14), (16, 17), (19, 20), (22, 23), (25, 26), (28, 29), (31, 32),
                                (34, 35), (30, 31), (33, 34)]

            inward = inward_ori_index
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward

        elif graph == 'wlasl_44':
            num_node = 44
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [(0, 1), (6, 7), (1, 2), (7, 8), (2, 3), (2, 12), (2, 16), (2, 20), (2, 24), (3, 4), (4, 5), (12, 13),
             (13, 14), (14, 15), (16, 17), (17, 18), (18, 19), (20, 21), (21, 22), (22, 23), (24, 25), (25, 26),
             (26, 27), (8, 9), (8, 28), (8, 32), (8, 36), (8, 40), (9, 10), (10, 11), (28, 29), (29, 30), (30, 31),
             (32, 33), (33, 34), (34, 35), (36, 37), (37, 38), (38, 39), (40, 41), (41, 42), (42, 43)]

            inward = inward_ori_index
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward

        elif graph == 'wlasl_44_2':
            num_node = 44
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [(0, 1), (1,2),  (2,3),  (3,4),  (4,5),  (5,6),  (7,8),  (8,9),  (9,10),  (10,11),  (11,12),  (12,13),
                                (2,14),  (2,17), (14,15),  (15,16),  (17,18),  (18,19),  (19,20),  (21,22),  (22,23),  (23,24),
                                (2,21), (2,25), (25,26),  (26,27), (27,28),
                                (9,29), (29,30),  (30,31),  (9,32),  (32,33),  (33,34),  (34,35),  (9,36),  (36,37),
                                (37,38), (38,39),  (9,40),  (40,41),  (41,42),  (42,43)]

            inward = inward_ori_index
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward
        elif graph == 'wlasl_34':
            num_node = 34
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [(0, 1), (5, 6), (1, 2), (6, 7), (2, 10), (2, 13), (2, 16), (2, 19), (3, 4), (11, 12),
                                (14, 15), (17, 18), (20, 21), (7, 22), (7, 25), (7, 28), (7, 31), (8, 9), (23, 24), (26, 27),
                                (29, 30), (32, 33), (2, 3), (2, 11), (2, 14), (2, 17), (2, 20), (7, 8), (7, 23), (7, 26), (7, 29), (7, 32)]

            inward = inward_ori_index
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward

        elif graph == 'slovo_42':
            num_node = 42
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [[ 0,  1],[ 1,  2],[ 2,  3],[ 3,  4],[ 0,  5],[ 5,  6],[ 6,  7],[ 7,  8],[ 9, 10],[10, 11],
                                [11, 12],[13, 14],[14, 15],[15, 16],[17, 18],[18, 19],[19, 20],[ 0, 17],[13, 17],[ 9, 13],
                                [ 5,  9],[21, 22],[22, 23],[23, 24],[24, 25],[21, 26],[26, 27],[27, 28],[28, 29],[30, 31],
                                [31, 32],[32, 33],[34, 35],[35, 36],[36, 37],[38, 39],[39, 40],[40, 41],[21, 38],[34, 38],
                                [30, 34],[26, 30]]

            inward = [(i,j) for (i,j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward
        elif graph == 'slovo_34':
            num_node = 34
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (8, 9), (9, 10), (11, 12), (12, 13),
                                (14, 15), (15, 16), (5, 8), (8, 11), (11, 14), (0, 14), (17, 18), (18, 19), (19, 20), (20, 21),
                                (17, 22), (22, 23), (23, 24), (25, 26), (26, 27), (28, 29), (29, 30), (31, 32), (32, 33), (22, 25),
                                (25, 28), (28, 31), (17, 31)]

            inward = [(i,j) for (i,j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward

        elif graph == 'slovo_32':
            num_node = 32
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [(0, 1), (1, 2), (2, 3),  (0, 4), (4, 5), (5, 6), (0,7),(7,8),(8,9),(0,10),(11,12),(10,11),
                                (0,13),(14,15),(13,14),(16, 17), (17, 18), (18, 19), (16, 20), (20, 21), (21, 22), (16, 23),
                                (23, 24), (24, 25), (16, 26), (27, 28), (26, 27), (16, 29), (30, 31), (29, 30)]

            inward = [(i,j) for (i,j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]

        elif graph == 'slovo_24':
            num_node = 24
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (6, 7), (8, 9), (10, 11), (0, 10), (4, 6), (6, 8),
                                (8, 10), (12, 13), (13, 14), (14, 15), (12, 16), (16, 17), (18, 19), (20, 21), (22, 23),
                                (12, 22), (16, 18), (18, 20), (20, 22)]

            inward = [(i,j) for (i,j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward

        elif graph == 'wlasl_tree':
            num_node = 24
            groups = get_groups()
            left_node = groups[0]
            right_node = groups[1]
            cnt_node = [0,1,2,3,9,10,13,14,4,5,6,7,17,18,21,22]  # 左右虚拟连接
            l_self_link = [(i, i) for i in left_node]
            r_self_link = [(i, i) for i in right_node]
            cnt_self_link = [(i, i) for i in cnt_node]
            l_index = [(0, 1), (1, 2), (2, 3), (2, 8), (2, 10), (2, 12), (2, 14), (3, 9), (3, 11), (3, 13), (3, 15), (8, 9),(8, 11), (8, 13),
                       (8, 15), (10, 9), (10, 11), (10, 13), (10, 15), (12, 9), (12, 11), (12, 13), (12, 15), (14, 9), (14, 11), (14, 13), (14, 15)]
            r_index = [(4, 5), (5, 6), (6, 7), (6, 16), (6, 18), (6, 20), (6, 22), (7, 17), (7, 19), (7, 21), (7, 23), (16, 17), (16, 19), (16, 21), (16, 23),
                       (18, 17), (18, 19), (18, 21), (18, 23), (20, 17), (20, 19), (20, 21), (20, 23), (22, 17), (22, 19), (22, 21), (22, 23)]
            cnt_index = [(0, 4), (1, 5), (2, 6), (3, 7), (3, 18), (3, 17), (3, 21), (3, 22), (3, 7), (3, 18), (3, 22),
                         (9, 17), (9, 21), (10, 7), (10, 18), (10, 22), (13, 17), (13, 21), (14, 7), (14, 18), (14, 22)]

            l_inward = l_index
            l_outward = [(j, i) for (i, j) in l_inward]
            l_neighbor = l_inward + l_outward

            r_inward = r_index
            r_outward = [(j, i) for (i, j) in r_inward]
            r_neighbor = r_inward + r_outward

            cnt_inward = cnt_index
            cnt_outward = [(j, i) for (i, j) in cnt_inward]
            cnt_neighbor = cnt_inward + cnt_outward

            self_link = [l_self_link, r_self_link, cnt_self_link]
            inward = [l_inward, r_inward, cnt_inward]
            outward = [l_outward, r_outward, cnt_outward]
            neighbor = [l_neighbor, r_neighbor, cnt_neighbor]
        elif graph == 'kinetics':
            num_node = 18
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            inward = inward_ori_index
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward
        elif graph == 'ntu':
            num_node = 25
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                                (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                                (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
            inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward
        elif graph == 'ucla':
            num_node = 20
            self_link = [(i, i) for i in range(num_node)]
            inward_ori_index = [(10, 8), (8, 9), (11, 9), (0, 9), (1, 0), (2, 1), (3, 2), (4, 9), (5, 4), (6, 5), (7, 6),
                                (12, 10), (13, 12), (14, 13), (15, 14), (16, 10), (17, 16), (18, 17), (19, 18)]
            outward = [(i, j) for (i, j) in inward_ori_index]
            inward = [(j, i) for (i, j) in outward]
            neighbor = inward + outward
        else:
            raise ValueError

        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
