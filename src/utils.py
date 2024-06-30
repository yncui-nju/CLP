import torch
import torch.nn as nn
import logging, os
import time
from tqdm import tqdm
import random
# from prettytable import PrettyTable
from torch.nn.init import xavier_normal_, uniform_
from torch.nn import Parameter
import numpy as np
from copy import deepcopy
import torch_geometric.utils.scatter as scatter
from collections import defaultdict
import torch.nn.functional as F


import time
import functools

def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 1 == 1:
            result = func(*args, **kwargs)
            return result
        else:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            try:
                class_name = args[0].__class__.__name__
                print(f"Method '{class_name}.{func.__name__}' executed in {execution_time} seconds.")
            except:
                print(f"Method '{func.__name__}' executed in {execution_time} seconds.")
            return result
    return wrapper


def same_seeds(seed):
    '''Set seed for reproduction'''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_param(shape):
    '''create learnable parameters'''
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param


def retype_parameters(args):
    '''
    '''
    args.learning_rate = float(args.learning_rate)
    args.batch_size = int(args.batch_size)
    args.neg_ratio = int(args.neg_ratio)
    args.margin = float(args.margin)
    args.gpu = int(args.gpu)
    try:
        args.ea_expand_training = True if args.ea_expand_training == 'True' else False
    except:
        raise('wrong_expand:', args.ea_expand_training)
    try:
        args.RPG = True if args.RPG == 'True' else False
    except:
        raise('wrong_expand:', args.RPG)
    args.seed = int(args.seed)
    args.emb_dim = int(args.emb_dim)
    args.ea_rate = float(args.ea_rate)
    args.use_attn = True if args.use_attn == 'True' else False
    args.use_RPG_triple = True if args.use_RPG_triple == 'True' else False
    args.use_augment = True if args.use_augment == 'True' else False


def load_fact(path, order='hrt'):
    '''
    Load (sub, rel, obj) from file 'path'.
    :param path: xxx.txt
    :return: fact list: [(s, r, o)]
    '''
    facts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split()
            if order=='hrt':
                s, r, o = line[0], line[1], line[2]
            elif order == 'htr':
                s, o, r = line[0], line[1], line[2]
            facts.append((s, r, o))
    return facts


def load_align_pair(path):
    '''
    Load (sub, rel, obj) from file 'path'.
    :param path: xxx.txt
    :return: fact list: [(s, r, o)]
    '''
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split()
            e1, e2 = line[0], line[1]
            pairs.append((e1, e2))
    return pairs


def build_edge_index(s, o):
    '''build edge_index using subject and object entity'''
    index = [s + o, o + s]
    return torch.LongTensor(index)


def init_param(model):
    for name, param in model.named_parameters():
        print(name, param.shape)
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            if '_embeddings' in name and model.args.scorer == 'RotatE':
                continue
            try:
                nn.init.xavier_normal_(param)
            except:
                nn.init.constant_(param, 0.0)


def merge_dicts(dict1, dict2):
    merged_dict = {**dict1, **dict2}  #
    return merged_dict



def find_topk_similar_entities_cross(A, B, topk):
    topk = int(topk)
    similarity_matrix = torch.mm(A, B.t())

    # Find the top-k most similar entities in B for each entity in A.
    topk_similar_entities_A_to_B = torch.topk(similarity_matrix, topk, dim=1)[1]

    # Find the top-k most similar entities in A for each entity in B.
    topk_similar_entities_B_to_A = torch.topk(similarity_matrix, topk, dim=0)[1].t()

    return topk_similar_entities_A_to_B, topk_similar_entities_B_to_A



def compute_coverage(r2s_A, r2s_B, topk_similar_entities_A_to_B, topk_similar_entities_B_to_A, A_start, B_start):
    num_relations_A = len(r2s_A)
    num_relations_B = len(r2s_B)

    coverage_tensor_A2B = torch.zeros(num_relations_A, num_relations_B)
    coverage_tensor_B2A = torch.zeros(num_relations_B, num_relations_A)
    topk_similar_entities_A_dict, heads_A_dict = dict(), dict()
    topk_similar_entities_B_dict, heads_B_dict = dict(), dict()

    for i, (r_A, entities_A) in enumerate(r2s_A.items()):

        heads_A = torch.LongTensor(list(entities_A)).to(topk_similar_entities_A_to_B.device) - A_start
        heads_A_dict[r_A] = heads_A

        topk_similar_entities_A_dict[r_A] = topk_similar_entities_A_to_B[heads_A].reshape(-1).unique()
    for i, (r_B, entities_B) in enumerate(r2s_B.items()):

        heads_B = torch.LongTensor(list(entities_B)).to(topk_similar_entities_A_to_B.device) - B_start
        heads_B_dict[r_B] = heads_B
        topk_similar_entities_B_dict[r_B] = topk_similar_entities_B_to_A[heads_B].reshape(-1).unique()

    for i, (r_A, entities_A) in enumerate(r2s_A.items()):
        topk_similar_entities_A = topk_similar_entities_A_dict[r_A]
        for j, (r_B, entities_B) in enumerate(r2s_B.items()):
            heads_B = heads_B_dict[r_B]
            coverage = topk_similar_entities_A.unsqueeze(1) == heads_B
            coverage_ratio = torch.sum(coverage.float()) / len(entities_A)
            coverage_tensor_A2B[i, j] = coverage_ratio

            heads_A = heads_A_dict[r_A]
            topk_similar_entities_B = topk_similar_entities_B_dict[r_B]
            coverage = topk_similar_entities_B.unsqueeze(1) == heads_A
            coverage_ratio = torch.sum(coverage.float()) / len(entities_B)
            coverage_tensor_B2A[j, i] = coverage_ratio
    return coverage_tensor_A2B, coverage_tensor_B2A


'''for RPG'''
def add_RPG_edge_using_cover(RPG_adj_1, threshold):
    def find_coordinates(tensor, threshold):
        def rel2inv(rel):
            if rel % 2 == 0:
                return rel+1
            else:
                return rel-1
        coordinates = []
        n = len(tensor)

        for i in range(n):
            for j in range(n):
                if tensor[i][j] >= threshold:
                    coordinates.append((i, rel2inv(j), j, tensor[i][j]))
        return coordinates

    RPG_edge = find_coordinates(RPG_adj_1, threshold)
    return RPG_edge


def rel2other_KG(rel, mid):
    if rel > mid:
        return rel-mid
    else:
        return rel


def create_batches(dataset, batch_size, according_to_relations=False):
    if not according_to_relations:
        batches = []
        num_batches = len(dataset) // batch_size + 1
        for i in range(num_batches):
            batches.append(dataset[i * batch_size: min((i + 1) * batch_size, len(dataset))])
    else:
        relation_samples_ = {}
        for triplet in dataset:
            relation = triplet[1]
            if relation not in relation_samples_:
                relation_samples_[relation] = []
            relation_samples_[relation].append(triplet)
        relation_samples = [val for key, val in relation_samples_.items()]
        batches = []
        for samples in relation_samples:
            random.shuffle(samples)
            num_batches = len(samples) // batch_size
            for i in range(num_batches):
                batch = samples[i * batch_size: (i + 1) * batch_size]
                batches.append(batch)

        remaining_samples = []
        for samples in relation_samples.values():
            remaining_samples.extend(samples[num_batches * batch_size:])

        random.shuffle(remaining_samples)

        while len(remaining_samples) > 0:
            sub_batch_size = min(batch_size, len(remaining_samples))
            sub_batch = remaining_samples[:sub_batch_size]
            batches.append(sub_batch)
            remaining_samples = remaining_samples[sub_batch_size:]

    return batches
