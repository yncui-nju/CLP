from src.utils import *
from ..utils import *
from copy import deepcopy as dcopy


class KnowledgeGraph():
    def __init__(self, args):
        self.args = args

        self.num_ent, self.num_rel = 0, 0
        self.entity2id, self.id2entity, self.relation2id, self.id2relation = dict(), dict(), dict(), dict()
        self.relation2inv = dict()
        self.entity_pairs = list()

        self.train, self.valid, self.test = list(), list(), list()
        self.edge_s, self.edge_r, self.edge_o = list(), list(), list()
        self.source = {i: Source(self.args) for i in range(len(self.args.source_list))}

        self.RPG_rel2edges = dict()
        self.expand_train = list()
        self.load_data()
        self.prepare_data()

    @timing_decorator
    def load_data(self):
        '''
        load data from all source file
        '''

        sr2o_all = dict()
        sr2o_valid = dict()
        self.r2s, self.r2o = dict(), dict()

        for ss_id, source in enumerate(self.args.source_list):
            edge_s, edge_r, edge_o = [], [], []
            '''load facts'''
            order = 'hrt'
            train_facts = load_fact(self.args.data_path + source + '/' + 'train.txt', order)
            valid_facts = load_fact(self.args.data_path + source + '/' + 'valid.txt', order)

            '''extract entities & relations from facts'''
            entities, relations = self.expand_entity_relation(train_facts)

            entities_id = self.ent_rel2id(entities, 'ent')
            relations_id = self.ent_rel2id(relations, 'rel')

            '''read train/valid data'''
            train = self.fact2id(train_facts)
            valid = self.fact2id(valid_facts)

            self.train += train
            self.valid += valid

            edge_s, edge_o, edge_r = self.expand_kg(train, 'train', edge_s, edge_o, edge_r, sr2o_all, sr2o_valid, self.r2s, self.r2o)
            self.edge_s = self.edge_s + edge_s
            self.edge_r = self.edge_r + edge_r
            self.edge_o = self.edge_o + edge_o
            _, _, _ = self.expand_kg(valid, 'valid', [], [], [], sr2o_all, sr2o_valid, None, None)

            '''store source'''
            self.store_source(ss_id, train, valid, edge_s, edge_o, edge_r, sr2o_all, sr2o_valid, entities_id, relations_id)

        '''read shared entities'''
        align_pairs = load_align_pair(self.args.data_path + 'known_shared_entities.txt')
        print('Alignment: ', len(align_pairs))
        self.ent2align = dict()
        for pair in align_pairs:
            e1, e2 = pair
            e1, e2 = self.entity2id[e1], self.entity2id[e2]
            self.ent2align[e1] = e2
            self.ent2align[e2] = e1
        self.expand_align(align_pairs)

        # 获取expand的集合
        for source_id, source in self.source.items():
            for fact in source.train:
                s, r, o = fact
                if s in self.ent2align.keys():
                    self.expand_train.append((self.ent2align[s], r, o))
                    self.expand_train.append((o, r+1, self.ent2align[s]))
                if o in self.ent2align.keys():
                    self.expand_train.append((s, r, self.ent2align[o]))
                    self.expand_train.append((self.ent2align[o], r+1, s))
        self.expand_train = set(self.expand_train)

        # for pyg scatter
        self.edge_s = torch.LongTensor(self.edge_s)
        self.edge_r = torch.LongTensor(self.edge_r)
        self.edge_o = torch.LongTensor(self.edge_o)

        self.sr2o_valid = sr2o_valid
        self.sr2o_all = sr2o_all

        self.r2s, self.r2o = self.get_r2e(self.train)

        # for RPG
        if self.args.RPG:
            self.get_RPG()
            self.RPG_r2e()
            self.update_attention_weight()


    def get_RPG(self):
        self.ori_RPG_adj_0 = torch.zeros([self.num_rel, self.num_rel], dtype=torch.float)
        self.ori_RPG_adj_1 = torch.zeros([self.num_rel, self.num_rel], dtype=torch.float)
        self.ori_RPG_cover = torch.eye(self.num_rel, dtype=torch.float)
        self.RPG_node = {r for r in range(self.num_rel)}
        self.RPG_node2entity = deepcopy(self.r2s)

        # get cover edge
        for i in range(self.num_rel):
            for j in range(self.num_rel):
                if i == j:
                    self.ori_RPG_adj_0[i, self.relation2inv[j]] = 1.0
                    self.ori_RPG_adj_0[j, self.relation2inv[i]] = 1.0
                    continue
                coverage1 = len(self.RPG_node2entity[i].intersection(self.RPG_node2entity[j]))/len(self.RPG_node2entity[i])
                coverage2 = len(self.RPG_node2entity[i].intersection(self.RPG_node2entity[j]))/len(self.RPG_node2entity[j])
                self.ori_RPG_cover[i, j] = coverage1
                self.ori_RPG_cover[j, i] = coverage2

        self.ori_RPG_adj_1 = torch.matmul(self.ori_RPG_cover, self.ori_RPG_adj_0)
        new_RPG_edge = add_RPG_edge_using_cover(self.ori_RPG_adj_1, self.args.lambda_1)
        self.ori_RPG_edge = new_RPG_edge

        self.RPG_edge = deepcopy(self.ori_RPG_edge)
        self.RPG_adj_0 = deepcopy(self.ori_RPG_adj_0)
        self.RPG_adj_1 = deepcopy(self.ori_RPG_adj_1)
        self.RPG_cover = deepcopy(self.ori_RPG_cover)
        print('Total RPG edges:', len(self.RPG_edge), '\tnature RPG edges:', self.num_rel)

    def RPG_r2e(self):
        self.args.logger.info('RPG graph edges: {}, new edges: {}'.format(len(self.RPG_edge), len(self.RPG_edge)-len(self.ori_RPG_edge)))
        self.RPG_r2s, self.RPG_r2o = self.get_r2e(self.RPG_edge)
        self.RPG_sr2o = dict()
        self.RPG_rel2edges = dict()
        for edge in self.RPG_edge:
            s, r, o = edge[0], edge[1], edge[2]
            item = self.RPG_sr2o.get((s, r), set())
            item.add(o)
            self.RPG_sr2o[(s, r)] = item

            item = self.RPG_sr2o.get((o, self.relation2inv[r]), set())
            item.add(s)
            self.RPG_sr2o[(o, self.relation2inv[r])] = item

            item = self.RPG_rel2edges.get(s, set())
            item.add(edge)
            self.RPG_rel2edges[s] = item
            item = self.RPG_rel2edges.get(o, set())
            item.add(edge)
            self.RPG_rel2edges[o] = item

        self.RPG_count = self.count_frequency(self.RPG_edge)


    def update_attention_weight(self):
        self.attention_weight = torch.zeros([self.num_rel, self.num_rel], dtype=torch.float).to(self.args.device)
        for qr in range(self.num_rel):
            qr_inv = self.relation2inv[qr]

            beta1, beta2 = 0.3, 0.3
            C = self.RPG_cover.to(self.args.device).clone()
            A = self.RPG_adj_1.to(self.args.device).clone()

            conf_matrix_1 = C[:, qr] * A[:, qr_inv]
            conf_matrix_multi = torch.matmul(C[:, qr].unsqueeze(1), A[:, qr_inv].unsqueeze(0))
            supp_matrix_1 = C[qr].t() * A[:, qr_inv]
            supp_matrix_2 = torch.matmul(C[qr].unsqueeze(1), A[:, qr_inv].unsqueeze(0))

            B = torch.zeros([A.size(0)]).to(self.args.device)

            conf_index_1 = torch.nonzero(conf_matrix_1 >= beta2)
            conf_index_multi = torch.nonzero(conf_matrix_multi>=beta2)
            supp_index_1 = torch.nonzero(supp_matrix_1 >= beta1)
            supp_index_2 = torch.nonzero(supp_matrix_2 >= beta1)

            if min(conf_index_1.shape) == 0:
                tuple_conf_1 = dict()
            else:
                tuple_conf_1 = {tuple(row):conf_matrix_1[row[0]] for row in conf_index_1.cpu().tolist()}
            if min(conf_index_multi.shape) == 0:
                tuple_conf_multi = dict()
            else:
                tuple_conf_multi = {tuple(row):conf_matrix_multi[row[0], row[1]] for row in conf_index_multi.cpu().tolist()}
            if min(supp_index_1.shape) == 0:
                tuple_supp_1 = set()
            else:
                tuple_supp_1 = set([tuple(row) for row in supp_index_1.cpu().tolist()])
            if min(supp_index_2.shape) == 0:
                tuple_supp_2 = set()
            else:
                tuple_supp_2 = set([tuple(row) for row in supp_index_2.cpu().tolist()])

            paths_1 = {path: conf for path, conf in tuple_conf_1.items() if path in tuple_supp_1}
            paths_2 = {path: conf for path, conf in tuple_conf_multi.items() if path in tuple_supp_2}

            for paths in [paths_1, paths_2]:
                for path in paths.keys():
                    for r in path:
                        if B[r] < paths[path]:
                            B[r] = paths[path]
                            if r % 2 == 0:
                                r_inv = r+1
                            else:
                                r_inv = r-1
                            B[r_inv] = paths[path]
            self.attention_weight[qr] = B

    def prepare_data(self):
        '''from RotatE'''
        self.count = self.count_frequency(self.train)
        self.sr2o_train = self.get_true_head_and_tail(self.train)

    def count_frequency(self, triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for triple in triples:
            h, r, t = triple[0], triple[1], triple[2]
            if (h, r) not in count:
                count[(h, r)] = start
            else:
                count[(h, r)] += 1
            if (t, r+1) not in count:
                count[(t, r+1)] = start
            else:
                count[(t, r+1)] += 1
        return count

    def get_true_head_and_tail(self, triples):
        '''
        Build a dictionary of true triples that will
        be used to filter negative sampling
        '''
        sr2o = dict()
        for h, r, t in triples:
            if (h, r) not in sr2o:
                sr2o[(h, r)] = []
            if (t, r+1) not in sr2o:
                sr2o[(t, r+1)] = []
            sr2o[(t, r+1)].append(h)
            sr2o[(h, r)].append(t)
        for (h, r) in sr2o.keys():
            sr2o[(h, r)] = np.array(list(set(sr2o[(h, r)])))
        return sr2o

    def load_test(self):
        if len(self.test) > 0:
            return

        cross_test_facts = load_fact(self.args.data_path + '/' + 'cross_test.txt', 'hrt')
        self.cross_test = self.fact2id(cross_test_facts)

        self.inner_test = []
        self.inner_test_align_0, self.inner_test_align_1, self.inner_test_align_2 = [], [], []
        for ss_id, source in enumerate(self.args.source_list):
            inner_test_facts = load_fact(self.args.data_path + '/' + source + '/' + 'test.txt', 'hrt')
            self.inner_test += self.fact2id(inner_test_facts)
        self.test = self.cross_test + self.inner_test
        _, _, _ = self.expand_kg(self.test, 'test', [], [], [], self.sr2o_all, None, None, None)
        self.split_test_to_source(self.test)

        # only for filter
        all_align_pairs = load_align_pair(self.args.data_path + 'all_shared_entities.txt')
        ent2align_all = dict()
        for pair in all_align_pairs:
            e1, e2 = pair
            ent2align_all[self.entity2id[e1]] = self.entity2id[e2]
            ent2align_all[self.entity2id[e2]] = self.entity2id[e1]
        self.sr2o_test = dict()
        for facts in [self.train, self.valid, self.test]:
            for fact in facts:
                s, r, o = fact
                item = self.sr2o_test.get((s, r), set())
                item.add(o)
                self.sr2o_test[(s, r)] = item
                item = self.sr2o_test.get((o, r + 1), set())
                item.add(s)
                self.sr2o_test[(o, r + 1)] = item
                if s in ent2align_all.keys():
                    item = self.sr2o_test.get((ent2align_all[s], r), set())
                    item.add(o)
                    self.sr2o_test[(ent2align_all[s], r)] = item
                    item = self.sr2o_test.get((o, r + 1), set())
                    item.add(ent2align_all[s])
                    self.sr2o_test[(o, r + 1)] = item
                if o in ent2align_all.keys():
                    item = self.sr2o_test.get((s, r), set())
                    item.add(ent2align_all[o])
                    self.sr2o_test[(s, r)] = item
                    item = self.sr2o_test.get((ent2align_all[o], r + 1), set())
                    item.add(s)
                    self.sr2o_test[(ent2align_all[o], r + 1)] = item

    def ent_rel2id(self, lst, mode='ent'):
        res = []
        for item in lst:
            if mode == 'ent':
                res.append(self.entity2id[item])
            else:
                res.append(self.relation2id[item])
        return res

    @timing_decorator
    def expand_entity_relation(self, facts):
        '''extract entities and relations from new facts'''
        new_entities, new_relations = set(), set()
        for (s, r, o) in facts:
            '''extract entities'''
            new_entities.add(s)
            new_relations.add(r)
            new_entities.add(o)
            if s not in self.entity2id.keys():
                self.entity2id[s] = self.num_ent
                self.id2entity[self.num_ent] = s
                self.num_ent += 1
            if o not in self.entity2id.keys():
                self.entity2id[o] = self.num_ent
                self.id2entity[self.num_ent] = o
                self.num_ent += 1

            '''extract relations'''
            if r not in self.relation2id.keys():
                self.relation2id[r] = self.num_rel
                self.relation2id[r + '_inv'] = self.num_rel + 1
                self.id2relation[self.num_rel] = r
                self.id2relation[self.num_rel + 1] = r + '_inv'
                self.relation2inv[self.num_rel] = self.num_rel + 1
                self.relation2inv[self.num_rel + 1] = self.num_rel
                self.num_rel += 2
        print('Entities {} Relations {}'.format(self.num_ent, self.num_rel))
        return new_entities, new_relations

    @timing_decorator
    def split_test_to_source(self, test):
        for idx, kg in self.source.items():
            max_id = max(kg.entities)
            min_id = min(kg.entities)
            for fact in test:
                h, r, t = fact
                if h <= max_id and h >= min_id:
                    kg.test.add(fact)

    @timing_decorator
    def fact2id(self, facts):
        '''(s name, r name, o name)-->(s id, r id, o id)'''
        fact_id = []
        for (s, r, o) in facts:
            fact_id.append((self.entity2id[s], self.relation2id[r], self.entity2id[o]))
        return fact_id

    @timing_decorator
    def expand_kg(self, facts, split, edge_s, edge_o, edge_r, sr2o_all, sr2o_valid, r2s, r2o):
        '''expand edge_index, edge_type (for GCN) and sr2o (to filter golden facts)'''
        def add_key2val(dict, key, val):
            '''add {key: value} to dict'''
            if key not in dict.keys():
                dict[key] = set()
            dict[key].add(val)
        edge_s.clear()
        edge_o.clear()
        edge_r.clear()
        for (h, r, t) in facts:
            if split == 'train':
                '''edge_index'''
                edge_s.append(h)
                edge_r.append(r)
                edge_o.append(t)
                edge_s.append(t)
                edge_r.append(r+1)
                edge_o.append(h)
                add_key2val(r2s, r, h)
                add_key2val(r2o, r, t)
                add_key2val(r2s, r+1, t)
                add_key2val(r2o, r+1, h)
            '''sr2o'''
            add_key2val(sr2o_all, (h, r), t)
            add_key2val(sr2o_all, (t, self.relation2inv[r]), h)
            if split in ['train', 'valid']:
                add_key2val(sr2o_valid, (h, r), t)
                add_key2val(sr2o_valid, (t, self.relation2inv[r]), h)
        return edge_s, edge_o, edge_r

    @timing_decorator
    def expand_align(self, align_pairs):
        for pair in align_pairs:
            if pair[0] not in self.entity2id.keys() or pair[1] not in self.entity2id.keys():
                continue
            self.entity_pairs.append((self.entity2id[pair[0]], self.entity2id[pair[1]]))

    @timing_decorator
    def store_source(self, ss_id, train_new, valid, edge_s, edge_o, edge_r, sr2o_all, sr2o_valid, entities, relations):
        '''store source data'''
        if ss_id > 0:
            self.source[ss_id].num_ent = self.num_ent - self.source[ss_id - 1].num_ent
            self.source[ss_id].num_rel = self.num_rel - self.source[ss_id - 1].num_rel
        else:
            self.source[ss_id].num_ent = self.num_ent
            self.source[ss_id].num_rel = self.num_rel

        '''train, valid and test data'''
        self.source[ss_id].train = dcopy(train_new)
        self.source[ss_id].valid = dcopy(valid)
        self.source[ss_id].entities = sorted(entities)
        self.source[ss_id].relations = sorted(relations)

        '''edge_index, edge_type (for GCN)'''
        self.source[ss_id].edge_s = dcopy(edge_s)
        self.source[ss_id].edge_r = dcopy(edge_r)
        self.source[ss_id].edge_o = dcopy(edge_o)

    def get_r2e(self, facts):
        r2o, r2s = dict(), dict()
        for fact in facts:
            s, r, o = fact[0], fact[1], fact[2]
            # r2o
            item = r2o.get(r, set())
            item.add(o)
            r2o[r] = item
            # r2s
            item = r2s.get(r, set())
            item.add(s)
            r2s[r] = item
            # r_inv2o
            item = r2o.get(r+1, set())
            item.add(o)
            r2o[r+1] = item
            # r_inv2s
            item = r2s.get(r+1, set())
            item.add(o)
            r2s[r+1] = item
        return r2s, r2o


class Source:
    def __init__(self, args):
        self.args = args
        self.num_ent, self.num_rel = 0, 0
        self.train, self.valid, self.test = list(), list(), set()
        self.cross_train = list()
        self.entities, self.relations = None, None

        self.edge_s, self.edge_r, self.edge_o = [], [], []
        self.cross_edge_s, self.cross_edge_r, self.cross_edge_o = [], [], []
        self.sr2o_all = dict()
        self.edge_index, self.edge_type = None, None




