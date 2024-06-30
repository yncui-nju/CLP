from src.utils import *


class KgeEmbedder(torch.nn.Module):
    def __init__(self, args, kg):
        super().__init__()
        self.args = args
        self.kg = kg
        self.num_ent = kg.num_ent
        self.num_rel = kg.num_rel

        self.ent_dim = args.emb_dim
        self.rel_dim = args.emb_dim

    def embed_ent(self, indexes):
        pass

    def embed_rel(self, indexes):
        pass

    def embed_ent_all(self, indexes=None):
        pass

    def embed_rel_all(self, indexes=None):
        pass


class LookupEmbedder(KgeEmbedder):
    def __init__(self, args, kg):
        super().__init__(args, kg)

    def _create_embedding(self):
        self.ent_embeddings = nn.Embedding(self.num_ent, self.ent_dim).to(self.args.device)
        self.rel_embeddings = nn.Embedding(self.num_rel, self.rel_dim).to(self.args.device)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.args.margin + 2.0) / self.ent_dim]),
            requires_grad=False
        )
        uniform_(self.ent_embeddings.weight, a=-self.embedding_range.item(), b=self.embedding_range.item())
        uniform_(self.rel_embeddings.weight, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def embed_ent(self, indexes, scorer=None, query_rel=None, mode=None):
        return self.ent_embeddings(indexes)

    def embed_rel(self, indexes):
        return self.rel_embeddings(indexes)

    def embed_ent_all(self, indexes=None, scorer=None, query_rel=None, mode=None):
        return self.ent_embeddings.weight

    def embed_rel_all(self, indexes=None):
        rel_embeddings = self.rel_embeddings.weight
        return torch.cat([rel_embeddings[0::2], rel_embeddings[0::2]], dim=-1).reshape(-1, rel_embeddings.size(-1))

    def embed_ent_prototype(self, indexes, kg=None, scores=None, scorer=None, query_rel=None, mode=None):
        '''
        将每个关系的头实体集合的mean作为prototype
        :param indexes: 需要编码的关系
        :param kg: 指示（p, r, p)中r所属的kg，方便llemapping
        :return: 各关系的prototype
        '''
        if kg is None or kg.all() or not kg.any():
            ent_embeddings = self.embed_ent_all(scorer=scorer, query_rel=query_rel, mode=mode)
            edge_s = self.kg.edge_s.to(self.args.device)
            edge_r = self.kg.edge_r.to(self.args.device)

            s_embeddings = torch.index_select(ent_embeddings, 0, edge_s)
            proto_embeddings = scatter(src=s_embeddings, index=edge_r, dim=0, dim_size=self.kg.num_rel, reduce='mean')
            return proto_embeddings[indexes]
        else:  # for llemapping
            edge_s = self.kg.edge_s.to(self.args.device)
            edge_r = self.kg.edge_r.to(self.args.device)
            ent_embeddings_1, ent_embeddings_2 = self.embed_ent_all_double(kg=kg)
            # for kg 1
            s_embeddings_1 = torch.index_select(ent_embeddings_1, 0, edge_s)
            proto_embeddings_1 = scatter(src=s_embeddings_1, index=edge_r, dim=0, dim_size=self.kg.num_rel, reduce='mean')
            # for kg 2
            s_embeddings_2 = torch.index_select(ent_embeddings_2, 0, edge_s)
            proto_embeddings_2 = scatter(src=s_embeddings_2, index=edge_r, dim=0, dim_size=self.kg.num_rel, reduce='mean')

            # fill prototype embeddings
            proto = torch.zeros([indexes.size(0), ent_embeddings_1.size(-1)], dtype=torch.float).to(self.args.device)
            proto[~kg] = proto_embeddings_1[indexes[~kg]]
            proto[kg] = proto_embeddings_2[indexes[kg]]
            return proto

    def forward(self, **kwargs):
        jobs = kwargs['jobs']
        res = dict()
        for job, value in jobs.items():
            opt_name = value['opt']
            if opt_name == 'ent_embedding':
                opt = self.embed_ent
            elif opt_name == 'rel_embedding':
                opt = self.embed_rel
            elif opt_name == 'ent_embedding_all':
                opt = self.embed_ent_all
            elif opt_name == 'rel_embedding_all':
                opt = self.embed_rel_all
            elif opt_name == 'ent_embedding_prototype':
                opt = self.embed_ent_prototype
            else:
                raise('Invalid embedding opration!')
            input = value['input']
            res[job] = opt(input['indexes'])
        return res


class LookupEmbedderAttn(LookupEmbedder):
    def __init__(self, args, kg):
        super().__init__(args, kg)
        self.drop = torch.nn.Dropout(p=0.0, inplace=False)

    def forward(self, **kwargs):
        jobs = kwargs['jobs']
        scorer = kwargs['scorer']
        mode = kwargs['mode']
        res = dict()
        for job, value in jobs.items():
            opt_name = value['opt']
            if opt_name == 'rel_embedding':
                query_rel = value['input']['indexes']
        for job, value in jobs.items():
            opt_name = value['opt']
            if opt_name == 'ent_embedding':
                opt = self.embed_ent
            elif opt_name == 'rel_embedding':
                opt = self.embed_rel
            elif opt_name == 'ent_embedding_all':
                opt = self.embed_ent_all
            elif opt_name == 'rel_embedding_all':
                opt = self.embed_rel_all
            elif opt_name == 'ent_embedding_prototype':
                opt = self.embed_ent_prototype
            else:
                raise('Invalid embedding opration!')
            input = value['input']
            if 'ent' in opt_name:
                res[job] = opt(indexes=input['indexes'], scorer=scorer, query_rel=query_rel, mode=mode)
            else:
                res[job] = opt(indexes=input['indexes'])
        return res

    def embed_ent(self, indexes, scorer=None, query_rel=None, mode=None):
        ent_embeddings_attn = self.embed_ent_all(scorer=scorer, query_rel=query_rel, mode=mode)
        return ent_embeddings_attn[indexes]

    def embed_ent_all(self, indexes=None, scorer=None, query_rel=None, mode=None):
        ent_embeddings = super().embed_ent_all(mode=mode)
        rel_embeddings = super().embed_rel_all()
        edge_s = self.kg.edge_s.to(self.args.device)
        edge_r = self.kg.edge_r.to(self.args.device)
        edge_o = self.kg.edge_o.to(self.args.device)

        s_embeddings = torch.index_select(ent_embeddings, 0, edge_s)
        r_embeddings = torch.index_select(rel_embeddings, 0, edge_r)

        if query_rel is not None:
            if mode == 'head-batch':
                qr = query_rel[0] + 1
                qr_inv = query_rel[0]
            else:
                qr = query_rel[0]
                qr_inv = query_rel[0] + 1
            if self.args.valid:
                B = self.kg.attention_weight[qr]
            else:
                B = self.kg.best_attention_weight[qr]
            if B.size(0) < self.kg.num_rel:
                B = torch.cat([B.unsqueeze(1), B.unsqueeze(1)], dim=-1).reshape(-1)
            B[qr] = 0.5
            B[qr_inv] = 0.5
            B = torch.repeat_interleave(torch.max(B[0::2], B[1::2]), 2)
        else:
            B = torch.ones_like(edge_r, dtype=torch.float).to(self.args.device)
        B += 0.1
        co_relation = torch.index_select(B, 0, edge_r).reshape(-1,1)
        ent_embeddings_attn_sum = scatter(src=self.drop(scorer.decode(s=s_embeddings, r=r_embeddings, modes=(edge_r % 2) == 1)*co_relation),
                                      index=edge_o, dim=0, dim_size=ent_embeddings.size(0), reduce='sum')
        weights_attn_sum = scatter(src=co_relation, index=edge_o, dim=0, reduce='sum', dim_size=ent_embeddings.size(0))

        ent_embeddings_attn = ent_embeddings_attn_sum / (weights_attn_sum+1e-10)
        ent_embeddings_attn = ent_embeddings_attn / 2 + ent_embeddings / 2
        return ent_embeddings_attn








