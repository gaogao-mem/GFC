import torch
import torch.nn as nn
import math
from transformers import AutoModel
from transformers import RobertaModel, BertModel
from transformers import BertTokenizer, RobertaTokenizer


class GFC(nn.Module):
    def __init__(self, args, ent2id, rel2id, id2ent, id2rel, triples, sub_map, rel_emb):
        super().__init__()
        self.args = args
        self.num_steps = 2
        self.id2ent = id2ent
        self.id2rel = id2rel
        self.rel_emb = rel_emb
        self.sub_map = sub_map
        self.max_active = args.max_active
        self.ent_act_thres = args.ent_act_thres
        self.num_relations = len(rel2id)

        Tsize = len(triples)
        Esize = len(ent2id)
        idx = torch.LongTensor([i for i in range(Tsize)])
        self.Msubj = torch.sparse.FloatTensor(
            torch.stack((idx, triples[:,0])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, Esize]))
        self.Mobj = torch.sparse.FloatTensor(
            torch.stack((idx, triples[:,2])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, Esize]))
        self.Mrel = torch.sparse.FloatTensor(
            torch.stack((idx, triples[:,1])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, self.num_relations]))
        print('triple size: {}'.format(Tsize), 'entity size: {} {}'.format(Esize, len(self.id2ent)), 'relation size: {}'.format(self.num_relations))
        try:
            if args.bert_name == "bert-base-uncased":
                self.bert_encoder = BertModel.from_pretrained('/root/autodl-tmp/GFC/models/bert-base-uncased')
                self.tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/GFC/models/bert-base-uncased')
            elif args.bert_name == "roberta-base":
                self.bert_encoder = RobertaModel.from_pretrained('/root/autodl-tmp/GFC/models/roberta-base')
                self.tokenizer = RobertaTokenizer.from_pretrained('/root/autodl-tmp/GFC/models/roberta-base')
            else:
                raise ValueError("please input the right name of pretrained model")
        except ValueError as e:
            raise e
        dim_hidden = self.bert_encoder.config.hidden_size
        self.rel_classifier1 = nn.Linear(dim_hidden, 1)
        self.rel_classifier2 = nn.Linear(dim_hidden, self.num_relations)
        self.key_layer = nn.Linear(dim_hidden, dim_hidden)
        self.hop_att_layer = nn.Sequential(
            nn.Linear(dim_hidden, 1)
            # nn.Tanh()
        )

        self.high_way = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.Sigmoid()
        )

    def follow(self, e, r):
        x = torch.sparse.mm(self.Msubj, e.t()) * torch.sparse.mm(self.Mrel, r.t())
        return torch.sparse.mm(self.Mobj.t(), x).t() # [bsz, Esize]

    def newfollow(self, e, pair, p):
        """
        Args:
            e [num_ent]: entity scores
            pair [rsz, 2]: pairs that are taken into consider
            p [rsz]: transfer probabilities of each pair
        """
        sub, obj = pair[:, 0], pair[:, 1]
        obj_p = e[sub] * p
        out = torch.index_add(torch.zeros_like(e), 0, obj, obj_p)
        return out

    def forward(self, heads, questions, answers=None, entity_range=None):
        # print('forward')
        # print('entity size: ', len(self.id2ent), heads.shape, answers.shape, entity_range.shape)
        for param in self.bert_encoder.parameters():
          param.requires_grad = True
        q = self.bert_encoder(**questions)
        q_embeddings, q_word_h = q.pooler_output, q.last_hidden_state # (bsz, dim_h), (bsz, len, dim_h)
        bsz, dim_h = q_embeddings.size()
        device = heads.device
        last_e = heads
        word_attns = []
        rel_probs = []
        ent_probs = []
        ctx_h_list = []
        q_word_h_hop = q_word_h
        q_word_h_dist_ctx = [0]

        path_infos = [] # [bsz, num_steps]
        for i in range(bsz):
            path_infos.append([])
            for j in range(self.num_steps):
                path_infos[i].append(None)

        for t in range(self.num_steps):
            h_key = self.key_layer(q_word_h_hop)  # [bsz, max_q, dim_h]
            q_logits = torch.matmul(h_key, q_word_h.transpose(-1, -2)) # [bsz, max_q, dim_h] * [bsz, dim_h, max_q] = [bsz, max_q, max_q]
            q_logits = q_logits.transpose(-1, -2)

            q_dist = torch.softmax(q_logits, 2)  # [bsz, max_q, max_q] 
            q_dist = q_dist * questions['attention_mask'].float().unsqueeze(1)  # [bsz, max_q, max_q]*[bsz, max_q]
            q_dist = q_dist / (torch.sum(q_dist, dim=2, keepdim=True) + 1e-6) # [bsz, max_q, max_q] 
            hop_ctx = torch.matmul(q_dist, q_word_h_hop) 
            if t == 0:
                z = 0
            else:
                z = self.high_way(q_word_h_dist_ctx[-1]) 
            if t == 0:
                q_word_h_hop = q_word_h + hop_ctx
            else:
                q_word_h_hop = q_word_h + hop_ctx + z*q_word_h_dist_ctx[-1]# [bsz, max_q, max_q]*[bsz, max_q, dim_h] = [bsz, max_q, dim_h]
            q_word_h_dist_ctx.append(hop_ctx + z*q_word_h_dist_ctx[-1])

            q_word_att = torch.sum(q_dist, dim=1, keepdim=True)  # [bsz, 1, max_q]  # 2改为1
            q_word_att = torch.softmax(q_word_att, 2)
            q_word_att = q_word_att * questions['attention_mask'].float().unsqueeze(1)  # [bsz, 1, max_q]*[bsz, max_q]
            q_word_att = q_word_att / (torch.sum(q_word_att, dim=2, keepdim=True) + 1e-6)  # [bsz, max_q, max_q]
            word_attns.append(q_word_att)  # bsz,1,q_max
            ctx_h = (q_word_h_hop.transpose(-1,-2) @ q_word_att.transpose(-1,-2)).squeeze(2)  # [bsz, dim_h, max_q] * [bsz, max_q,1]

            ctx_h_list.append(ctx_h)

            r_stack = []
            e_stack = []
            cnt_trunc = 0
            for i in range(bsz):
                # print("last_e: ", last_e[i], last_e[i].shape)
                sort_score, sort_idx = torch.sort(last_e[i], dim=0, descending=True)
                # print("sort_score: ", sort_score[0], sort_score.shape, sort_idx[0], sort_idx.shape)
                e_idx = sort_idx[sort_score.gt(self.ent_act_thres)].tolist()
                if len(e_idx) == 0:
                    # print('no active entity at step {}'.format(t))
                    e_idx = sort_idx[:50].tolist()
                    '''for idx in sort_idx:
                        idx = idx.item()
                        if idx in self.sub_map:
                            e_idx.append(idx)
                            print('found one: ', idx)
                            break'''

                rels = []
                pair = []
                for j in e_idx:
                    for p, o in self.sub_map[j]:
                        rels.append(p)
                        pair.append((j, o))
                        if len(rels) == self.max_active: # limit the number of next-hop
                            cnt_trunc += 1
                            # print('trunc: {}'.format(cnt_trunc))
                            break
                    if len(rels) == self.max_active:
                        break

                # print('rels: ', len(rels))
                '''if len(rels) == 0:
                    print('no active entity at step {}'.format(t))
                    for idx in sort_idx:
                        idx = idx.item()
                        if idx in self.sub_map:
                            for p, o in self.sub_map[idx]:
                                rels.append(p)
                                pair.append((idx, o))
                                break'''
                if len(rels) == 0:
                    e_stack.append(last_e[i])
                    continue

                rels = torch.LongTensor(rels).to(device)
                pair = torch.LongTensor(pair).to(device)

                # print('step {}, desc number {}'.format(t, len(rg)))
                rels_embeddings = self.rel_emb[rels]
                # print('rels_embedding', rels_embeddings.shape, len(rels))
                d_logit = self.rel_classifier1(ctx_h[i:i+1] * rels_embeddings).squeeze(1) # [rsz,]
                d_prob = torch.sigmoid(d_logit) # [rsz,]
                # print('d_prob:', torch.sum(d_prob), d_prob)
                e_stack.append(self.newfollow(last_e[i], pair, d_prob))  # [Esize]

                '''# expand rsz to all relations
                rel_prob = torch.index_add(torch.zeros(self.num_relations).to(device), 0, rels, d_prob) # [num_relations,]
                print('d_prob', d_prob.shape, d_prob[0], 'rel_prob: ', rel_prob.shape, rel_prob[rels[0]])
                print(rels)
                print(d_prob)
                print(rel_prob)
                # transfer probability
                r_stack.append(rel_prob)'''
                
                if not self.training:
                    rels_text = [self.id2rel[idx] for idx in rels]
                    # collect path
                    act_idx = torch.where(d_prob>0.9)
                    # print(act_idx)
                    if act_idx[0].numel() > 0:
                        act_idx=act_idx[0].tolist()
                        # print(rels_text[act_idx[0]])
                    else:
                        act_idx = []
                    path_infos[i][t] = [(pair[_][0], rels_text[_], pair[_][1]) for _ in act_idx]

            '''r_stack = torch.stack(r_stack, dim=0)  # [bsz, num_relations]
            print("r_stack ", r_stack.shape)'''

            '''last_e1 = self.follow(last_e, r_stack)  # faster than index_add [bsz, num_ent]
            print("last_e1 ", last_e1.shape)'''
            last_e = torch.stack(e_stack, dim=0)
            # print("last_e2 ", last_e.shape)
            '''# whether last_e is the same with last_e2
            print('the same?: ',torch.sum(torch.abs(last_e1 - last_e2)))'''


            # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float() 
            z = (m * last_e + (1-m)).detach()
            last_e = last_e / z
            ent_probs.append(last_e)

        hop_res = torch.stack(ent_probs, dim=1) # [bsz, num_hop, num_ent]
        # print('hop_res: ', hop_res)

        ctx_h_history = torch.stack(ctx_h_list, dim=2)  # [bsz, dim_h, num_hop]
        hop_logit = self.hop_att_layer(ctx_h_history.transpose(-1, -2))  # bsz, num_hop, 1
        hop_attn = torch.softmax(hop_logit.transpose(-1, -2), 2).transpose(-1, -2)  # bsz, num_hop, 1
        # print('hop_attn: ', hop_attn)

        last_e = torch.sum(hop_res * hop_attn, dim=1) # [bsz, num_ent]
        # print('last_e: ', last_e, torch.sum(last_e, dim=1))

        if not self.training:
            return {
                'e_score': last_e,
                'word_attns': word_attns,
                'rel_probs': rel_probs,
                'ent_probs': ent_probs,
                'hop_attn': hop_attn.squeeze(2)
            }
        else:
            '''weight = answers * 99 + 1 
            print(torch.sum(entity_range * weight))
            loss = torch.sum(entity_range * weight * torch.pow(last_e - answers, 2)) / torch.sum(entity_range * weight)
            '''
            weight = answers * 999 + 1
            loss = torch.mean(weight * torch.pow(last_e - answers, 2))
            # print(loss)

            return {'loss': loss}
