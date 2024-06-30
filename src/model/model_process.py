from ..data_load.data_loader import *
from torch.utils.data import DataLoader
import math


class TrainEvalBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg

    def _create_dataset(self):
        pass


'''Cross-Link Prediction'''
class CrossLinkPredictionTrainBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg
        '''prepare data'''
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.head_dataset = CrossLinkPredictionTrainDatasetMarginLoss(self.args, self.kg, mode='head-batch')
        self.tail_dataset = CrossLinkPredictionTrainDatasetMarginLoss(self.args, self.kg, mode='tail-batch')
        self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.facts, self.args.batch_size)
        self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.facts, self.args.batch_size)
        self.head_data_loader = DataLoader(self.head_dataset,
                                      batch_sampler=self.head_sampler,
                                      collate_fn=self.head_dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(self.args.seed)),
                                      pin_memory=True)
        self.tail_data_loader = DataLoader(self.tail_dataset,
                                           batch_sampler=self.tail_sampler,
                                           collate_fn=self.tail_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(self.args.seed)),
                                           pin_memory=True)
        if self.args.RPG:
            self.RPG_train_processor = RPGTrainBatchProcessor(self.args, self.kg)

    def process_epoch(self, model, optimizer):
        model.train()
        '''Start training'''

        head_data_loader = iter(self.head_data_loader)
        tail_data_loader = iter(self.tail_data_loader)
        success = False
        total_loss = 0.0
        while not success:
            total_loss = 0.0
            count_num = 0
            count_loss = 0
            count_batch = 0
            try:
                for idx_b, batch in enumerate(tqdm(range(2 * len(self.head_data_loader)))):
                    '''get loss'''
                    if idx_b % 2 == 0:
                        try:
                            iter_ = next(head_data_loader)
                        except:
                            continue
                        if iter_ is None:
                            continue
                        bh, br, bt, by, bc, mode, subsampling_weight, RPG_facts, is_expand, is_ea = iter_
                        mode = 'head-batch'
                    else:
                        try:
                            iter_ = next(tail_data_loader)
                        except:
                            continue
                        if iter_ is None:
                            continue
                        bh, br, bt, by, bc, mode, subsampling_weight, RPG_facts, is_expand, is_ea = iter_
                        mode = 'tail-batch'
                    current_samples_num = subsampling_weight.size(0)
                    bh = bh.to(self.args.device)
                    br = br.to(self.args.device)
                    bt = bt.to(self.args.device)
                    by = by.to(self.args.device)
                    bc = bc.to(self.args.device)
                    subsampling_weight = subsampling_weight.to(self.args.device)
                    if count_loss == 0:
                        optimizer.zero_grad()

                    jobs = {
                        'sub_emb': {'opt': 'ent_embedding', 'input': {"indexes": bh}, 'mode': mode},
                        'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": br}, 'mode': mode},
                        'obj_emb': {'opt': 'ent_embedding', 'input': {"indexes": bt}, 'mode': mode},
                    }
                    pred = model.forward(jobs=jobs, stage='train', mode=mode, margin=self.args.margin)
                    batch_loss = model.loss(pred, by, subsampling_weight, bc, neg_ratio=self.args.neg_ratio)
                    if self.args.use_RPG_triple and len(RPG_facts) > 0:
                        self.RPG_train_processor.set_RPG_edges(RPG_facts)
                        RPG_loss = self.RPG_train_processor.loss(model, optimizer)
                        batch_loss = self.args.alpha * RPG_loss + batch_loss

                    '''update'''
                    count_loss += batch_loss * current_samples_num
                    count_num += current_samples_num
                    count_batch += 1
                    if count_num >= self.args.batch_size or self.args.scorer == 'RotatE':
                        count_loss /= count_num
                        count_loss.backward()
                        optimizer.step()
                        total_loss += count_loss.item()
                        count_loss = 0
                        count_num = 0
                        count_batch = 0
                    '''post processing'''
                success = True
            except:
                import sys
                e = sys.exc_info()[0]
                # 如果是内存超出的错误，减小batch_size，否则直接报错并退出
                if 'CUDA out of memory' in str(e):
                    self.batch_size = self.batch_size // 2
                    self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.facts, self.batch_size)
                    self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.facts, self.batch_size)
                    self.head_data_loader = DataLoader(self.head_dataset,
                                                        batch_sampler=self.head_sampler,
                                                        collate_fn=self.head_dataset.collate_fn,
                                                        generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                        pin_memory=True)
                    self.tail_data_loader = DataLoader(self.tail_dataset,
                                                        batch_sampler=self.tail_sampler,
                                                        collate_fn=self.tail_dataset.collate_fn,
                                                        generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                        pin_memory=True)
                    print('Batch size is too large, reduce to', self.batch_size)
                else:
                    print('Error:', e)
                    break

        return total_loss

    def add_facts_using_relations(self, same, inverse):
        self.head_dataset.add_facts_using_relations(same, inverse)
        self.tail_dataset.add_facts_using_relations(same, inverse)
        self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.facts, self.args.batch_size)
        self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.facts, self.args.batch_size)
        self.head_data_loader = DataLoader(self.head_dataset,
                                           batch_sampler=self.head_sampler,
                                           collate_fn=self.head_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(self.args.seed)),
                                           pin_memory=True)
        self.tail_data_loader = DataLoader(self.tail_dataset,
                                           batch_sampler=self.tail_sampler,
                                           collate_fn=self.tail_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(self.args.seed)),
                                           pin_memory=True)


class CrossLinkPredictionValidBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        self.batch_size = self.args.test_batch_size
        '''prepare data'''
        self.head_dataset = CrossLinkPredictionValidDataset(args, kg, mode='head-batch')
        self.tail_dataset = CrossLinkPredictionValidDataset(args, kg, mode='tail-batch')
        self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.valid, self.args.test_batch_size)
        self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.valid, self.args.test_batch_size)
        self.head_data_loader = DataLoader(self.head_dataset,
                                      # shuffle=False,
                                      batch_sampler=self.head_sampler,
                                      # batch_size=self.batch_size,
                                      collate_fn=self.head_dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),
                                      pin_memory=True)
        self.tail_data_loader = DataLoader(self.tail_dataset,
                                           # shuffle=False,
                                           batch_sampler=self.tail_sampler,
                                           # batch_size=self.batch_size,
                                           collate_fn=self.tail_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(args.seed)),
                                           pin_memory=True)

    def process_epoch(self, model):
        model.eval()

        '''start evaluation'''
        success = False
        while not success:
            try:
                num = 0
                results = dict()
                for data_loader in [self.head_data_loader, self.tail_data_loader]:
                    for batch in data_loader:
                        sub, rel, obj, label, mode = batch
                        sub = sub.to(self.args.device)
                        rel = rel.to(self.args.device)
                        obj = obj.to(self.args.device)
                        label = label.to(self.args.device)
                        num += len(sub)

                        '''link prediction'''
                        if mode == 'tail-batch':
                            jobs = {
                                'sub_emb': {'opt': 'ent_embedding', 'input': {"indexes": sub}, 'mode': mode},
                                'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": rel}, 'mode': mode},
                                'obj_emb': {'opt': 'ent_embedding_all', 'input': {"indexes": obj}, 'mode': mode},
                            }
                            target = obj
                        else:
                            jobs = {
                                'sub_emb': {'opt': 'ent_embedding_all', 'input': {"indexes": sub}, 'mode': mode},
                                'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": rel}, 'mode': mode},
                                'obj_emb': {'opt': 'ent_embedding', 'input': {"indexes": obj}, 'mode': mode},
                            }
                            target = sub
                        pred = model(jobs=jobs, mode=mode, margin=self.args.margin)



                        b_range = torch.arange(pred.size()[0], device=self.args.device)
                        target_pred = pred[b_range, target]
                        pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)

                        pred[b_range, target] = target_pred

                        '''rank all candidate entities'''
                        ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, target]

                        '''get results'''
                        ranks = ranks.float()
                        results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                        results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                        results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)

                        for k in range(10):
                            results['hits{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get('hits{}'.format(k + 1), 0.0)
                success = True
            except:
                import sys
                e = sys.exc_info()[0]
                if 'CUDA out of memory' in str(e):
                    print('CUDA out of memory, try to reduce batch size by half')
                    self.batch_size = self.batch_size // 2
                    self.head_dataset = CrossLinkPredictionValidDataset(self.args, self.kg, mode='head-batch')
                    self.tail_dataset = CrossLinkPredictionValidDataset(self.args, self.kg, mode='tail-batch')
                    self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.valid,
                                                             self.args.batch_size)
                    self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.valid,
                                                             self.args.batch_size)
                    self.head_data_loader = DataLoader(self.head_dataset,
                                                       # shuffle=False,
                                                       batch_sampler=self.head_sampler,
                                                       # batch_size=self.batch_size,
                                                       collate_fn=self.head_dataset.collate_fn,
                                                       generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                       pin_memory=True)
                    self.tail_data_loader = DataLoader(self.tail_dataset,
                                                       # shuffle=False,
                                                       batch_sampler=self.tail_sampler,
                                                       # batch_size=self.batch_size,
                                                       collate_fn=self.tail_dataset.collate_fn,
                                                       generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                       pin_memory=True)
                else:
                    print('Unexpected error:', e)
                    break


        count = float(results['count'])
        for key, val in results.items():
            if key != 'count':
                results[key] = round(val / count, 4)
        return results

    def add_facts_using_relations(self, same, inverse):
        self.head_dataset.add_facts_using_relations(same, inverse)
        self.tail_dataset.add_facts_using_relations(same, inverse)
        self.head_data_loader = DataLoader(self.head_dataset,
                                           shuffle=False,
                                           batch_size=self.batch_size,
                                           collate_fn=self.head_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(self.args.seed)),
                                           pin_memory=True)
        self.tail_data_loader = DataLoader(self.tail_dataset,
                                           shuffle=False,
                                           batch_size=self.batch_size,
                                           collate_fn=self.tail_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(self.args.seed)),
                                           pin_memory=True)


class CrossLinkPredictionTestBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        self.batch_size = self.args.test_batch_size
        '''prepare data'''
        self.head_dataset = CrossLinkPredictionTestDataset(args, kg, mode='head-batch')
        self.tail_dataset = CrossLinkPredictionTestDataset(args, kg, mode='tail-batch')
        self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.test, self.args.test_batch_size)
        self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.test, self.args.test_batch_size)
        self.head_data_loader = DataLoader(self.head_dataset,
                                      # shuffle=False,
                                      batch_sampler=self.head_sampler,
                                      # batch_size=self.batch_size,
                                      collate_fn=self.head_dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),
                                      pin_memory=True)
        self.tail_data_loader = DataLoader(self.tail_dataset,
                                           # shuffle=False,
                                           batch_sampler=self.tail_sampler,
                                           # batch_size=self.batch_size,
                                           collate_fn=self.tail_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(args.seed)),
                                           pin_memory=True)

    def process_epoch(self, model):
        def update_results(results_, ranks_):
            results_['count'] = torch.numel(ranks_) + results_.get('count', 0.0)
            results_['mr'] = torch.sum(ranks_).item() + results_.get('mr', 0.0)
            results_['mrr'] = torch.sum(1.0 / ranks_).item() + results_.get('mrr', 0.0)
            for k in range(10):
                results_['hits{}'.format(k + 1)] = torch.numel(ranks_[ranks_ <= (k + 1)]) + results_.get(
                    'hits{}'.format(k + 1), 0.0)
            return results_
        def ave_results(results_):
            count = float(results_['count'])
            for key, val in results_.items():
                if key != 'count':
                    results_[key] = round(val / count, 4)
            return results_

        model.eval()
        success = False
        while not success:
            try:
                num = 0
                results_inner, results_cross, results_cross_relation, results_cross_entity = dict(), dict(), dict(), dict()
                '''start evaluation'''
                for data_loader in [self.head_data_loader, self.tail_data_loader]:
                    for batch in tqdm(data_loader):
                        sub, rel, obj, label, mode, types = batch
                        sub = sub.to(self.args.device)
                        rel = rel.to(self.args.device)
                        obj = obj.to(self.args.device)
                        label = label.to(self.args.device)
                        types = types.to(self.args.device)

                        num += len(sub)
                        '''link prediction'''
                        if mode == 'tail-batch':
                            jobs = {
                                'sub_emb': {'opt': 'ent_embedding', 'input': {"indexes": sub}, 'mode': mode},
                                'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": rel}, 'mode': mode},
                                'obj_emb': {'opt': 'ent_embedding_all', 'input': {"indexes": obj}, 'mode': mode},
                            }
                            target = obj
                        else:
                            jobs = {
                                'sub_emb': {'opt': 'ent_embedding_all', 'input': {"indexes": sub}, 'mode': mode},
                                'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": rel}, 'mode': mode},
                                'obj_emb': {'opt': 'ent_embedding', 'input': {"indexes": obj}, 'mode': mode},
                            }
                            target = sub
                        pred = model(jobs=jobs, mode=mode, margin=self.args.margin)

                        # pred = model(jobs=jobs, mode=mode, margin=self.args.margin)
                        b_range = torch.arange(pred.size()[0], device=self.args.device)

                        # # filter
                        target_pred = pred[b_range, target]
                        pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)

                        pred[b_range, target] = target_pred

                        '''rank all candidate entities'''
                        ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, target]
                        '''get results'''
                        ranks = ranks.float()

                        inner_idx = (types == -1).reshape(-1)
                        cross_idx = (types == 1).reshape(-1)
                        cross_relation_idx = ((sub<self.kg.source[0].num_ent) & (obj<self.kg.source[0].num_ent) & (rel>=self.kg.source[0].num_rel)) \
                                             + ((sub>=self.kg.source[0].num_ent)&(obj>=self.kg.source[0].num_ent)&(rel<self.kg.source[0].num_rel))
                        cross_entity_idx = (~cross_relation_idx) & cross_idx

                        ranks_inner = ranks[inner_idx]
                        ranks_cross = ranks[cross_idx]
                        ranks_cross_relation = ranks[cross_relation_idx]
                        ranks_cross_entity = ranks[cross_entity_idx]

                        results_inner = update_results(results_inner, ranks_inner)
                        results_cross = update_results(results_cross, ranks_cross)
                        results_cross_relation = update_results(results_cross_relation, ranks_cross_relation)
                        results_cross_entity = update_results(results_cross_entity, ranks_cross_entity)

                success = True
            except Exception as e:
                # 如果是内存不足的问题，减小batch_size，否则直接报错
                if 'CUDA out of memory' in str(e):
                    self.args.logger.info('Error: {}'.format(e))
                    self.args.logger.info('Retry...')
                    self.batch_size = int(self.batch_size / 2)
                    self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.test,
                                                             self.batch_size)
                    self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.test,
                                                             self.batch_size)
                    self.head_data_loader = DataLoader(self.head_dataset,
                                                       # shuffle=False,
                                                       batch_sampler=self.head_sampler,
                                                       # batch_size=self.batch_size,
                                                       collate_fn=self.head_dataset.collate_fn,
                                                       generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                       pin_memory=True)
                    self.tail_data_loader = DataLoader(self.tail_dataset,
                                                       # shuffle=False,
                                                       batch_sampler=self.tail_sampler,
                                                       # batch_size=self.batch_size,
                                                       collate_fn=self.tail_dataset.collate_fn,
                                                       generator=torch.Generator().manual_seed(int(self.args.seed)),
                                                       pin_memory=True)
                else:
                    self.args.logger.info('Error: {}'.format(e))
                    break



        results_inner = ave_results(results_inner)
        results_cross = ave_results(results_cross)
        results_cross_relation = ave_results(results_cross_relation)
        results_cross_entity = ave_results(results_cross_entity)

        self.args.logger.info('cross_relation_results:{}'.format(results_cross_relation))
        self.args.logger.info('cross__entity_results:{}'.format(results_cross_entity))
        self.args.logger.info('inner_results:{}'.format(results_inner))
        self.args.logger.info('cross_results:{}'.format(results_cross))

        rr = results_inner
        self.args.logger.info('{}\t{}\t{}\t{}\t{}'.format(rr['mrr'], rr['hits1'], rr['hits3'], rr['hits5'], rr['hits10']))
        rr = results_cross
        self.args.logger.info('{}\t{}\t{}\t{}\t{}'.format(rr['mrr'], rr['hits1'], rr['hits3'], rr['hits5'], rr['hits10']))
        rr = results_cross_relation
        self.args.logger.info('{}\t{}\t{}\t{}\t{}'.format(rr['mrr'], rr['hits1'], rr['hits3'], rr['hits5'], rr['hits10']))
        rr = results_cross_entity
        self.args.logger.info('{}\t{}\t{}\t{}\t{}'.format(rr['mrr'], rr['hits1'], rr['hits3'], rr['hits5'], rr['hits10']))
        return results_cross


class RPG_filler():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg
        '''prepare data'''

    def process_epoch(self, model):
        model.eval()
        '''start'''
        ent_embeddings = model.encoder.embed_ent_all(scorer=model.decoder.scorer)

        final_ent_embeddings = ent_embeddings
        A = final_ent_embeddings[:self.kg.source[0].num_ent]
        B = final_ent_embeddings[self.kg.source[0].num_ent:]

        A = torch.nn.functional.normalize(A, dim=1)
        B = torch.nn.functional.normalize(B, dim=1)

        r2s_A, r2s_B = {r: s for r, s in self.kg.r2s.items() if r < self.kg.source[0].num_rel}, {r: s for r, s in
                                                                                                 self.kg.r2s.items() if
                                                                                                 r >= self.kg.source[
                                                                                                     0].num_rel}

        A2B_topk, B2A_topk = find_topk_similar_entities_cross(A, B, self.args.topk)
        A2B_cover, B2A_cover = compute_coverage(r2s_A, r2s_B, A2B_topk, B2A_topk, A_start=0, B_start=self.kg.source[0].num_ent)

        self.kg.RPG_cover[:self.kg.source[0].num_rel, self.kg.source[0].num_rel:] = A2B_cover*2.0   # top-k will miss many alignments, multiplying by 2 to adjust.
        self.kg.RPG_cover[self.kg.source[0].num_rel:, :self.kg.source[0].num_rel] = B2A_cover*2.0
        self.kg.RPG_cover[self.kg.RPG_cover>1] = 1
        self.kg.RPG_adj_1 = torch.matmul(self.kg.RPG_cover, self.kg.RPG_adj_0)

        A_relation = {i for i in range(self.kg.source[0].num_rel)}
        B_relation = {i+self.kg.source[0].num_rel for i in range(self.kg.num_rel - self.kg.source[0].num_rel)}
        A2B_relation = self.get_relation(A2B_cover, B2A_cover, A_relation, B_relation, threshold=self.args.lambda_2)
        B2A_relation = self.get_relation(B2A_cover, A2B_cover, B_relation, A_relation, threshold=self.args.lambda_2)

        self.kg.RPG_edge = add_RPG_edge_using_cover(self.kg.RPG_adj_1, self.args.lambda_1)
        self.kg.RPG_r2e()
        self.kg.update_attention_weight()

        same_prototypes = self.get_co_relation(A2B_relation, B2A_relation)
        same_relations_bi = self.get_corelation_bidirection(same_prototypes)
        same, inverse, scores_same, scores_inverse = dict(), dict(), dict(), dict()
        for r1, r2 in same_relations_bi:
            if r1 % 2 == 0 and r2 % 2 == 0:
                min_ = 100000
                if r1 in same:
                    min_ = min(min_, same[r1])
                if r2 in same:
                    min_ = min(min_, same[r2])
                if r2 >= self.kg.source[0].num_rel:
                    if (r1 not in scores_same and r2 not in scores_same) or (min(A2B_cover[r1, r2-len(A_relation)], B2A_cover[r2-len(A_relation), r1]) > min_):
                        same[r1] = r2
                        same[r2] = r1
                        scores_same[r1] = A2B_cover[r1, r2-len(A_relation)]
                        scores_same[r2] = B2A_cover[r2 - len(A_relation), r1]
                else:
                    if (r1 not in scores_same and r2 not in scores_same) or (B2A_cover[r1-len(A_relation), r2]+A2B_cover[r2, r1-len(A_relation)] > min_):
                        same[r1] = r2
                        same[r2] = r1
                        scores_same[r1] = B2A_cover[r1-len(A_relation), r2]
                        scores_same[r2] = A2B_cover[r2, r1 - len(A_relation)]
            elif r1 % 2 == 0 and r2 % 2 == 1:
                min_ = 10000
                if r1 in scores_inverse:
                    min_ = min(min_, scores_inverse[r1])
                if r2 in scores_inverse:
                    min_ = min(min_, scores_inverse[r2])
                if r2 >= self.kg.source[0].num_rel:
                    if (r1 not in scores_inverse and self.kg.relation2inv[r2] not in scores_inverse) or min(A2B_cover[rel2other_KG(r1, self.kg.source[0].num_rel), rel2other_KG(r2, self.kg.source[0].num_rel)], B2A_cover[rel2other_KG(r2, self.kg.source[0].num_rel), rel2other_KG(r1, self.kg.source[0].num_rel)]) > min_:
                        inverse[r1] = self.kg.relation2inv[r2]
                        inverse[self.kg.relation2inv[r2]] = r1
                        scores_inverse[r1] = A2B_cover[rel2other_KG(r1, self.kg.source[0].num_rel), rel2other_KG(r2, self.kg.source[0].num_rel)]
                        scores_inverse[self.kg.relation2inv[r2]] = B2A_cover[rel2other_KG(r2, self.kg.source[0].num_rel), rel2other_KG(r1, self.kg.source[0].num_rel)]
                else:
                    if (r1 not in scores_inverse and self.kg.relation2inv[r2] not in scores_inverse) or min(B2A_cover[rel2other_KG(r1, self.kg.source[0].num_rel), rel2other_KG(r2, self.kg.source[0].num_rel)], A2B_cover[rel2other_KG(r2, self.kg.source[0].num_rel), rel2other_KG(r1, self.kg.source[0].num_rel)]) > min_:
                        inverse[r1] = self.kg.relation2inv[r2]
                        inverse[self.kg.relation2inv[r2]] = r1
                        scores_inverse[r1] = B2A_cover[rel2other_KG(r1, self.kg.source[0].num_rel), rel2other_KG(r2, self.kg.source[0].num_rel)]
                        scores_inverse[self.kg.relation2inv[r2]] = A2B_cover[rel2other_KG(r2, self.kg.source[0].num_rel), rel2other_KG(r1, self.kg.source[0].num_rel)]


        filtered_same, filtered_inverse = dict(), dict()
        for r1, r2 in same.items():
            if same[r2] == r1:
                filtered_same[r1] = r2
                filtered_same[r2] = r1

        for r1, r2 in inverse.items():
            if inverse[r2] == r1:
                filtered_inverse[r1] = r2
                filtered_inverse[r2] = r1

        print('same', filtered_same)
        print('inverse', filtered_inverse)

        return filtered_same, filtered_inverse

    def find_topk_similar_entities(self, embeddings, topk):
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        similarity_matrix.fill_diagonal_(float('-inf'))
        topk_similar_entities = torch.topk(similarity_matrix, topk, dim=1)[1]
        return topk_similar_entities

    def get_relation(self, A2B_cover, B2A_cover, A_relations, B_relations, threshold=0.5):
        result_dict = {}
        follow = dict()

        for i, a_relation in enumerate(A_relations):
            result_dict[a_relation] = set()

            for j, b_relation in enumerate(B_relations):
                cover_rate = A2B_cover[i][j]

                if cover_rate >= threshold/2:
                    result_dict[a_relation].add(b_relation)
        return result_dict

    def get_co_relation(self, A2B_relation, B2A_relation):
        same_relations = set()
        for rA, rBs in A2B_relation.items():
            for rB in rBs:
                if rA in B2A_relation[rB]:
                    same_relations.add((rA, rB))
                    same_relations.add((rB, rA))
        for rB, rAs in B2A_relation.items():
            for rA in rAs:
                if rB in A2B_relation[rA]:
                    same_relations.add((rA, rB))
                    same_relations.add((rB, rA))
        return same_relations

    def get_corelation_bidirection(self, same_prototypes):
        def rel2inv(r):
            if r % 2 == 0:
                return r+1
            else:
                return r-1
        same_relations_bi = set()
        for r1, r2 in same_prototypes:
            if (rel2inv(r1), rel2inv(r2)) in same_prototypes:
                same_relations_bi.add((r1, r2))
                same_relations_bi.add((r2, r1))

        return same_relations_bi



class RPGTrainBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg
        self.neg_ratio = 0
        '''prepare data'''
        self.head_dataset = RPGTrainDataset(self.args, self.kg, mode='head-batch', neg_ratio=self.neg_ratio)
        self.tail_dataset = RPGTrainDataset(self.args, self.kg, mode='tail-batch', neg_ratio=self.neg_ratio)

    def process_epoch(self, model, optimizer):
        model.train()
        '''Start training'''
        total_loss = 0.0
        head_data_loader = iter(self.head_data_loader)
        tail_data_loader = iter(self.tail_data_loader)
        count_num = 0
        count_loss = 0
        count_batch = 0
        for idx_b, batch in enumerate(tqdm(range(2 * len(self.head_data_loader)))):
            '''get loss'''
            if idx_b % 2 == 0:
                bh, br, bt, by, bc, mode, subsampling_weight = next(head_data_loader)
                mode = 'head-batch'
            else:
                bh, br, bt, by, bc, mode, subsampling_weight = next(tail_data_loader)
                mode = 'tail-batch'

            bh = bh.to(self.args.device)
            br = br.to(self.args.device)
            bt = bt.to(self.args.device)
            by = by.to(self.args.device)
            bc = bc.to(self.args.device)

            subsampling_weight = subsampling_weight.to(self.args.device)
            current_samples_num = subsampling_weight.size(0)
            optimizer.zero_grad()

            jobs = {
                'sub_emb': {'opt': 'ent_embedding_prototype', 'input': {"indexes": bh}, 'mode': mode},
                'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": br}, 'mode': mode},
                'obj_emb': {'opt': 'ent_embedding_prototype', 'input': {"indexes": bt}, 'mode': mode},
            }
            pred = model.forward(jobs=jobs, stage='train', mode=mode, margin=2.0)
            batch_loss = model.loss(pred, by, subsampling_weight, bc, neg_ratio=self.neg_ratio)

            batch_loss = batch_loss

            '''update'''
            count_loss += batch_loss * current_samples_num
            count_num += current_samples_num
            count_batch += 1
            if count_num >= self.args.batch_size or self.args.scorer == 'RotatE':
                count_loss /= count_num
                count_loss.backward()
                optimizer.step()
                total_loss += count_loss.item()
                count_loss = 0
                count_num = 0
                count_batch = 0
        return total_loss

    def loss(self, model, optimizer):
        model.train()
        '''Start training'''
        total_loss = 0.0
        losses = []
        nums = []
        head_data_loader = iter(self.head_data_loader)
        tail_data_loader = iter(self.tail_data_loader)
        for idx_b, batch in enumerate(range(2*math.ceil(len(self.head_dataset.facts)/self.args.batch_size))):
            '''get loss'''
            if idx_b % 2 == 0:
                bh, br, bt, by, bc, mode, subsampling_weight, mapping_weight = next(head_data_loader)
                mode = 'head-batch'
            else:
                bh, br, bt, by, bc, mode, subsampling_weight, mapping_weight = next(tail_data_loader)
                mode = 'tail-batch'

            bh = bh.to(self.args.device)
            br = br.to(self.args.device)
            bt = bt.to(self.args.device)
            by = by.to(self.args.device)
            bc = bc.to(self.args.device)

            subsampling_weight = subsampling_weight.to(self.args.device)
            mapping_weight = mapping_weight.to(self.args.device)

            jobs = {
                'sub_emb': {'opt': 'ent_embedding_prototype', 'input': {"indexes": bh}, 'mode': mode},
                'rel_emb': {'opt': 'rel_embedding', 'input': {"indexes": br}, 'mode': mode},
                'obj_emb': {'opt': 'ent_embedding_prototype', 'input': {"indexes": bt}, 'mode': mode},
            }
            pred = model.forward(jobs=jobs, stage='train', mode=mode, margin=self.args.margin)
            batch_loss = model.loss(pred, by, subsampling_weight, bc, neg_ratio=self.neg_ratio)/torch.mean(mapping_weight)
            losses.append(batch_loss * bh.reshape(-1, 1).size(0))
            nums.append(bh.reshape(-1, 1).size(0))
            '''post processing'''
        try:
            total_loss = torch.sum(torch.stack(losses))/sum(nums)
        except:
            total_loss = 0
        return total_loss

    def set_RPG_edges(self, facts):
        self.head_dataset.update_facts(facts)
        self.tail_dataset.update_facts(facts)
        self.head_sampler = RelationBatchSampler(self.args, self.kg, self.head_dataset.facts, self.args.batch_size)
        self.tail_sampler = RelationBatchSampler(self.args, self.kg, self.tail_dataset.facts, self.args.batch_size)
        self.head_data_loader = DataLoader(self.head_dataset,
                                           batch_sampler=self.head_sampler,
                                           collate_fn=self.head_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(self.args.seed)),
                                           pin_memory=True)
        self.tail_data_loader = DataLoader(self.tail_dataset,
                                           batch_sampler=self.tail_sampler,
                                           collate_fn=self.tail_dataset.collate_fn,
                                           generator=torch.Generator().manual_seed(int(self.args.seed)),
                                           pin_memory=True)









