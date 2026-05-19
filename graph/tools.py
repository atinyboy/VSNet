import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    if isinstance(self_link[0], list):
        A = []
        for i in range(len(self_link)):
            I = edge2mat(self_link[i], num_node)
            In = normalize_digraph(edge2mat(inward[i], num_node))
            Out = normalize_digraph(edge2mat(outward[i], num_node))
            A_i = np.stack((I, In, Out))
            A.append(A_i)
        return np.stack(A, axis=0)
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


def get_groups(dataset='WLASL'):
    groups = []
    if dataset == 'WLASL':
        groups.append([0, 1, 2, 3, 8, 9, 10, 11,12, 13,14,15])
        groups.append([4,5,6,7,16,17,18,19, 20,21, 22,23])

    return groups
