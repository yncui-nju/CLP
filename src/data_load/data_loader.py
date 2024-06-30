from torch.utils.data import Dataset
from src.utils import *
from torch.utils.data import Dataset, DataLoader, BatchSampler


'''for Cross-KG Link Prediction'''
class CrossLinkPredictionTrainDatasetMarginLoss(Dataset):
    def __init__(self, args, kg, mode='head-batch'):
        self.args = args
        self.kg = kg
        self.mode = mode
        self.original_facts, self.original_is_expand, self.original_is_ea = self.build_facts()
        self.original_is_expand = torch.BoolTensor(self.original_is_expand).reshape(-1, 1)
        self.original_is_ea = torch.BoolTensor(self.original_is_ea).reshape(-1, 1)
        self.original_confidence = torch.ones([len(self.original_facts), 1], dtype=torch.float32)

        self.facts = deepcopy(self.original_facts)
        self.confidence = self.original_confidence.clone()
        self.is_expand = self.original_is_expand.clone()
        self.is_ea = self.original_is_ea.clone()

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, idx):
        '''
        :param idx: idx of the training fact
        :return: a positive facts and its negative facts
        '''
        fact = self.facts[idx]
        conf = self.confidence[idx]
        is_expand = self.is_expand[idx]
        is_ea = self.is_ea[idx]
        if is_expand:
            mf = list(self.kg.RPG_rel2edges.get(fact[1], set()))

        else:
            mf = list()
        '''negative sampling'''
        fact, label, subsampling_weight = self.corrupt(fact, is_ea)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        return fact, label, conf, self.mode, subsampling_weight, mf, torch.ones(len(fact), dtype=torch.bool)*is_expand, torch.ones(len(fact), dtype=torch.bool)*is_ea

    @staticmethod
    def collate_fn(data):
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        conf = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        subsampling_weight = torch.cat([_[4] for _ in data], dim=0)
        RPG_facts = []
        for _ in data:
            mf = _[5]
            RPG_facts += mf
        is_expands = torch.cat([_[6] for _ in data], dim=0)
        is_eas = torch.cat([_[7] for _ in data], dim=0)
        return fact[:,0], fact[:,1], fact[:,2], label, conf, mode, subsampling_weight, RPG_facts, is_expands, is_eas

    def build_facts(self):
        '''
        build training data for each snapshots
        :return: training data
        '''
        head_facts = list()
        is_expands = list()
        is_ea = list()
        all_relation, cross_relation_head, cross_relation_tail = set(), set(), set()
        for source_id, source in self.kg.source.items():
            expand_fact_num = 0
            for fact in source.train:
                s, r, o = fact
                all_relation.add(r)
                head_facts.append((s, r, o))
                is_expands.append(False)
                is_ea.append(False)
                if self.args.ea_expand_training:
                    if s in self.kg.ent2align.keys():
                        cross_relation_head.add(r)
                        head_facts.append((self.kg.ent2align[s], r, o))
                        is_expands.append(True)
                        is_ea.append(False)
                        expand_fact_num += 1
                    if o in self.kg.ent2align.keys():
                        cross_relation_tail.add(r)
                        head_facts.append((s, r, self.kg.ent2align[o]))
                        is_expands.append(True)
                        is_ea.append(False)

                        expand_fact_num += 1
            if self.args.ea_expand_training:
                self.args.logger.info('KG {} Expand Training Triples: {}'.format(source_id, expand_fact_num))
                self.args.logger.info('All Relation {}, cross relation head {}, cross relation tail {}.'.format(len(all_relation), len(cross_relation_head), len(cross_relation_tail)))
        return head_facts, is_expands, is_ea

    def corrupt(self, fact, is_ea):
        s, r, o = fact
        if is_ea:
            facts = [fact]
            label = [1]
            facts.extend([(s, r, o) for i in range(self.args.neg_ratio)])
            label += [-1 for i in range(self.args.neg_ratio)]
            return facts, label, torch.Tensor([0.1])
        s_temp, o_temp, r_temp = s, o, r
        try:
            subsampling_weight = self.kg.count[(s, r)] + self.kg.count[(o, self.kg.relation2inv[r])]
        except:
            try:
                subsampling_weight = self.kg.count[(self.kg.ent2align[s], r)] + self.kg.count[(o, self.kg.relation2inv[r])]
                s_temp = self.kg.ent2align[s]
            except:
                try:
                    subsampling_weight = self.kg.count[(s, r)] + self.kg.count[(self.kg.ent2align[o], self.kg.relation2inv[r])]
                    o_temp = self.kg.ent2align[o]
                except:
                    try:
                        try:
                            r_temp = self.same[r]
                            subsampling_weight = self.kg.count[(s, r_temp)] + self.kg.count[(o, self.kg.relation2inv[r_temp])]
                        except:
                            r_temp = self.kg.relation2inv[self.inverse[r]]
                            subsampling_weight = self.kg.count[(s, r_temp)] + self.kg.count[(o, self.kg.relation2inv[r_temp])]
                    except:
                        print(s, r, o)
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        facts = [fact]
        label = [1]
        negative_sample_size = 0
        negative_sample_list = []

        while negative_sample_size < self.args.neg_ratio:
            if self.mode == 'head-batch':
                negative_sample = np.random.randint(0, self.kg.num_ent, self.args.neg_ratio * 2)

                mask = np.in1d(
                    negative_sample,
                    self.kg.sr2o_train[(o_temp, self.kg.relation2inv[r_temp])],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                negative_sample = np.random.randint(0, self.kg.num_ent, self.args.neg_ratio * 2)

                mask = np.in1d(
                    negative_sample,
                    self.kg.sr2o_train[(s_temp, r_temp)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_size += negative_sample.size
            negative_sample_list.append(negative_sample)

        negative_sample = np.concatenate(negative_sample_list)[:self.args.neg_ratio]
        if self.mode == 'head-batch':
            facts.extend([(negative_sample[i], r, o) for i in range(self.args.neg_ratio)])
            label += [-1 for i in range(self.args.neg_ratio)]
        elif self.mode == 'tail-batch':
            facts.extend([(s, r, negative_sample[i]) for i in range(self.args.neg_ratio)])
            label += [-1 for i in range(self.args.neg_ratio)]

        return facts, label, subsampling_weight

    def add_facts_using_relations(self, same, inverse):
        self.same = {val:key for key, val in same.items()}
        self.inverse = {val:key for key, val in inverse.items()}
        facts, confidence, is_expands, is_ea = [], [], [], []
        for fact in self.kg.train:
            s, r, o = fact
            if r in same.keys():
                facts.append((s, same[r], o))
                confidence.append([0.5])
                is_expands.append(True)
                is_ea.append(False)
            if r in inverse.keys():
                facts.append((o, inverse[r], s))
                confidence.append([0.5])
                is_expands.append(True)
                is_ea.append(False)
        self.args.logger.info('Add {} facts using relations!'.format(len(facts)))
        self.facts = self.original_facts + facts
        self.confidence = torch.cat([self.original_confidence, torch.FloatTensor(confidence)], dim=0)
        self.is_expand = torch.cat([self.original_is_expand, torch.BoolTensor(is_expands).reshape(-1, 1)])
        self.is_ea = torch.cat([self.original_is_ea, torch.BoolTensor(is_ea).reshape(-1, 1)])


class RelationBatchSampler(BatchSampler):
    '''
    This class is created to save memory for the attn embedder.
    The training using attn embedder involves facts with the same relation placed within the same batch.
    '''
    def __init__(self, args, kg, dataset, batch_size, shuffle=True):
        self.args = args
        self.kg = kg
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.args.use_attn:
            #  to save memory, we group identical relations into the same batch
            self.relation_samples = self._separate_relations()
            super(RelationBatchSampler, self).__init__(self.relation_samples, self.batch_size, self.shuffle)
        else:
            super(RelationBatchSampler, self).__init__(self.dataset, self.batch_size, self.shuffle)

    def _separate_relations(self):
        relation_samples = {}
        for idx, sample in enumerate(self.dataset):
            try:
                relation = sample['fact'][1]
            except:
                relation = sample[1]
            try:
                if relation not in relation_samples.keys():
                    relation_samples[relation] = []
            except:
                print(relation)
            relation_samples[relation].append(idx)
        return relation_samples

    def __iter__(self):
        if not self.args.use_attn:
            dataset = [i for i in range(len(self.dataset))]
            batches = []
            if self.shuffle:
                random.shuffle(dataset)
            num_batches = len(dataset) // self.batch_size
            for i in range(num_batches):
                batch = list(dataset)[i * self.batch_size: (i + 1) * self.batch_size]
                batches.append(batch)
                # yield(batch)
            if len(dataset) % self.batch_size > 0:
                # yield self.dataset[num_batches * self.batch_size:]
                batches.append(list(dataset)[num_batches * self.batch_size:])
            if self.shuffle:
                random.shuffle(batches)
            return iter(batches)
        else:
            self.relation_indices = list(self.relation_samples.keys())
            if self.shuffle:
                random.shuffle(self.relation_indices)
            batches = []
            for relation in self.relation_indices:
                samples = self.relation_samples[relation]
                random.shuffle(samples)
                num_samples = len(samples)
                num_batches = num_samples // self.batch_size
                for i in range(num_batches):
                    batch = samples[i * self.batch_size: (i + 1) * self.batch_size]
                    batches.append(batch)
                if num_samples % self.batch_size > 0:
                    batches.append(samples[num_batches * self.batch_size:])
            if self.shuffle:
                random.shuffle(batches)
            return iter(batches)

    def __len__(self):
        if self.args.use_attn:
            num = 0
            for samples in self.relation_samples.values():
                num += len(samples)//self.batch_size
                if len(samples) % self.batch_size > 0:
                    num += 1
            return num
        else:
            return (len(self.dataset) - 1) // self.batch_size + 1


class CrossLinkPredictionValidDataset(Dataset):
    '''
    Dataloader for evaluation. For each snapshot, load the valid & test facts and filter the golden facts.
    '''
    def __init__(self, args, kg, mode='head-batch'):
        self.args = args
        self.kg = kg
        self.mode = mode

        '''prepare data for validation and testing'''
        self.original_valid = self.build_facts()
        self.valid = deepcopy(self.original_valid)

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        ele = self.valid[idx]
        fact, label, source_id = torch.LongTensor(ele['fact']), ele['label'], ele['source_id']

        label = self.get_label(list(label), source_id)
        return fact[0], fact[1], fact[2], label.float(), self.mode

    @staticmethod
    def collate_fn(data):
        s = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        o = torch.stack([_[2] for _ in data], dim=0)
        label = torch.stack([_[3] for _ in data], dim=0)
        mode = data[0][4]
        return s, r, o, label, mode

    def get_label(self, label, source_id):
        '''
        Filter the golden facts. The label 1.0 denote that the entity is the golden answer.
        :param label:
        :return: dim = test factnum * all seen entities
        '''
        y = np.zeros([self.kg.num_ent], dtype=np.float32)
        if source_id == 0:
            y[len(self.kg.source[0].entities):] = 1.0
        else:
            y[:len(self.kg.source[0].entities)] = 1.0
        for e2 in label: y[e2] = 1.0

        return torch.FloatTensor(y)

    def build_facts(self):
        '''
        build validation and test set using the valid & test data for each snapshots
        :return: validation set and test set
        '''
        valid = []
        for source_id, source in self.kg.source.items():
            for fact in source.valid:
                s, r, o = fact
                if self.mode == 'head-batch':
                    label = set(self.kg.sr2o_valid[(o, r+1)])
                elif self.mode == 'tail-batch':
                    label = set(self.kg.sr2o_valid[(s, r)])
                valid.append({'fact': (s, r, o), 'label': label, 'source_id': source_id})
        return valid

    def add_facts_using_relations(self, same, inverse):
        new_valid = []
        for source_id, source in self.kg.source.items():
            for fact in source.valid:
                s, r, o = fact
                if r in same:
                    if self.mode == 'head-batch':
                        label = set(self.kg.sr2o_valid[(o, r + 1)])
                    elif self.mode == 'tail-batch':
                        label = set(self.kg.sr2o_valid[(s, r)])
                    new_valid.append({'fact': (s, same[r], o), 'label': label, 'source_id': source_id})
                if r in inverse:
                    if self.mode == 'head-batch':
                        label = set(self.kg.sr2o_valid[(o, r + 1)])
                    elif self.mode == 'tail-batch':
                        label = set(self.kg.sr2o_valid[(s, r)])
                    new_valid.append({'fact': (o, inverse[r], s), 'label': label, 'source_id': source_id})
        self.valid = self.original_valid + new_valid


class CrossLinkPredictionTestDataset(Dataset):
    '''
    Dataloader for evaluation. For each snapshot, load the valid & test facts and filter the golden facts.
    '''
    def __init__(self, args, kg, mode='head-batch'):
        self.args = args
        self.kg = kg
        self.mode = mode

        '''prepare data for validation and testing'''
        self.test = self.build_facts()

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        ele = self.test[idx]
        fact, label, source_id, known = torch.LongTensor(ele['fact']), ele['label'], ele['source_id'], ele['known']
        type = torch.LongTensor([ele['type']])
        label = self.get_label(list(label), known)
        return fact[0], fact[1], fact[2], label.float(), self.mode, type

    @staticmethod
    def collate_fn(data):
        s = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        o = torch.stack([_[2] for _ in data], dim=0)
        label = torch.stack([_[3] for _ in data], dim=0)
        mode = data[0][4]
        types = torch.stack([_[5] for _ in data], dim=0)
        return s, r, o, label, mode, types

    def get_label(self, label, known):
        '''
        Filter the golden facts. The label 1.0 denote that the entity is the golden answer.
        :param label:
        :return: dim = test factnum * all seen entities
        '''
        if known:
            y = np.ones([self.kg.num_ent], dtype=np.float32)
        else:
            y = np.zeros([self.kg.num_ent], dtype=np.float32)
            for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)

    def build_facts(self):
        '''
        build validation and test set using the valid & test data for each snapshots
        :return: validation set and test set
        '''
        test = []
        inner_test = set(self.kg.inner_test)
        for source_id, source in self.kg.source.items():
            for fact in source.test:
                if fact in self.kg.expand_train:
                    continue
                else:
                    known = False
                s, r, o = fact
                if self.mode == 'tail-batch':
                    label = set(self.kg.sr2o_test[(s, r)])
                else:
                    label = set(self.kg.sr2o_test[(o, r+1)])
                if (s, r, o) in inner_test:
                    type = -1
                else:
                    type = 1
                test.append({'fact': (s, r, o), 'label': label, 'source_id': source_id, 'known': known, 'type': type})
        return test


class RPGTrainDataset(Dataset):
    def __init__(self, args, kg, mode='head-batch', neg_ratio=0):
        self.args = args
        self.kg = kg
        self.mode = mode
        self.neg_ratio = neg_ratio

        self.original_facts, self.original_confidence = self.build_facts()
        self.original_confidence = torch.FloatTensor(self.original_confidence).reshape(-1, 1)

        self.facts = deepcopy(self.original_facts)
        self.confidence = self.original_confidence.clone()
        self.mapping_weights = torch.ones_like(self.confidence, dtype=torch.float32)

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, idx):
        '''
        :param idx: idx of the training fact
        :return: a positive facts and its negative facts
        '''
        fact = self.facts[idx]
        conf = self.confidence[idx]
        mapping_weights = self.mapping_weights[idx]


        '''negative sampling'''
        fact, label, subsampling_weight = self.corrupt(fact)
        fact, label = torch.LongTensor(fact), torch.Tensor(label)
        return fact, label, conf, self.mode, subsampling_weight, mapping_weights

    @staticmethod
    def collate_fn(data):
        fact = torch.cat([_[0] for _ in data], dim=0)
        label = torch.cat([_[1] for _ in data], dim=0)
        conf = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        subsampling_weight = torch.cat([_[4] for _ in data], dim=0)
        mapping_weight = torch.cat([_[5] for _ in data], dim=0)

        return fact[:,0], fact[:,1], fact[:,2], label, conf, mode, subsampling_weight, mapping_weight

    @timing_decorator
    def build_facts(self):
        '''
        build training data for each snapshots
        :return: training data
        '''
        head_facts = list()
        confidence = list()
        for fact in self.kg.RPG_edge:
            s, r, o, c = fact
            if r % 2 == 0:
                head_facts.append((s, r, o))
                confidence.append(c)
        return head_facts, confidence

    def corrupt(self, fact):
        s, r, o = fact[0], fact[1], fact[2]
        s_temp, o_temp, r_temp = s, o, r
        subsampling_weight = self.kg.RPG_count[(s, r)] + self.kg.RPG_count[(o, self.kg.relation2inv[r])]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        facts = [(s, r, o)]
        label = [1]
        negative_sample_size = 0
        negative_sample_list = []

        if self.neg_ratio <= 0:
            return facts, label, subsampling_weight

        while negative_sample_size < self.neg_ratio:
            negative_sample = np.random.randint(0, self.kg.num_rel, self.neg_ratio * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.kg.RPG_sr2o[(o_temp, self.kg.relation2inv[r_temp])],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.kg.RPG_sr2o[(s_temp, r_temp)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_size += negative_sample.size
            negative_sample_list.append(negative_sample)

        negative_sample = np.concatenate(negative_sample_list)[:self.neg_ratio]
        # Concatenate negative samples, then merge them into the facts while updating the labels.
        if self.mode == 'head-batch':
            facts.extend([(negative_sample[i], r, o) for i in range(self.neg_ratio)])
            label += [-1 for i in range(self.neg_ratio)]
        elif self.mode == 'tail-batch':
            facts.extend([(s, r, negative_sample[i]) for i in range(self.neg_ratio)])
            label += [-1 for i in range(self.neg_ratio)]

        return facts, label, subsampling_weight

    def update_facts(self, facts):
        head_facts = list()
        confidence = list()
        visited_facts = dict()
        facts_mapping_num = list()
        num = 0
        for fact in facts:
            s, r, o, c = fact
            if r % 2 == 0:
                if (s, r, o) not in visited_facts:
                    head_facts.append((s, r, o))
                    confidence.append(c)
                    facts_mapping_num.append(1)
                    visited_facts[(s, r, o)] = num
                    num+=1
                else:
                    idx = visited_facts[(s, r, o)]
                    facts_mapping_num[idx] += 1
                    confidence[idx] = max(confidence[idx], c)
        self.facts = head_facts
        self.confidence = torch.FloatTensor(confidence).reshape(-1, 1)
        self.mapping_weights = torch.FloatTensor(facts_mapping_num).reshape(-1, 1)
        return head_facts, confidence, facts_mapping_num
