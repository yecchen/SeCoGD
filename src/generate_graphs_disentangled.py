import numpy as np
import os
import pickle
import dgl
import torch
from tqdm import tqdm
import argparse
from collections import defaultdict

def load_quadruples(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            contextid = int(line_split[4])
            quadrupleList.append([head, rel, tail, time, contextid])
            times.add(time)

    times = list(times)
    times.sort()

    return np.array(quadrupleList), np.asarray(times)


def get_data_with_t(data, tim):
    x = data[np.where(data[:,3] == tim)].copy()
    x = np.delete(x, 3, 1)  # drops time column
    return x

def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm

def r2e(triplets, num_rels):
    src, rel, dst, contextid = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r+num_rels))
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst, contextid) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        r_to_e[rel+num_rels].add(src)
        r_to_e[rel+num_rels].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx


def get_big_graph(triples, num_nodes, num_rels, K):
    g_list = [] # len = K

    for k in range(K):
        curr_triplets = triples[np.where(triples[:, 3]==k)]
        if len(curr_triplets) == 0:
            g = dgl.DGLGraph()
            g.add_nodes(num_nodes)
            norm = comp_deg_norm(g).view(-1, 1)
            node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
            g.ndata['id'] = node_id
            g.ndata['norm'] = norm
            g.uniq_r = np.array([])
            g.r_to_e = []
            g.r_len = []
            g_list.append(g)
            continue

        src, rel, dst, contextid = curr_triplets.transpose()
        src_double, dst_double = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel_double = np.concatenate((rel, rel + num_rels))

        g = dgl.DGLGraph()
        g.add_nodes(num_nodes)
        g.add_edges(src_double, dst_double)

        norm = comp_deg_norm(g).view(-1, 1)
        node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)

        g.ndata['id'] = node_id
        g.ndata['norm'] = norm
        g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
        g.edata['type'] = torch.LongTensor(rel_double)

        uniq_r, r_len, r_to_e = r2e(curr_triplets, num_rels)
        g.uniq_r = uniq_r
        g.r_to_e = r_to_e
        g.r_len = r_len

        g_list.append(g)

    return tuple(g_list)

def get_entid2contextid(train_data):
    entid2contextid = dict()
    train_data = train_data.tolist()
    for idx, (head, rel, tail, time, contextid) in tqdm(enumerate(train_data), total=len(train_data)):
        if head not in entid2contextid:
            entid2contextid[head] = set()
        if tail not in entid2contextid:
            entid2contextid[tail] = set()
        entid2contextid[head].add(contextid)
        entid2contextid[tail].add(contextid)
    return entid2contextid

def get_relid2contextid(train_data, num_r):
    relid2contextid = dict()
    train_data = train_data.tolist()
    for idx, (head, rel, tail, time, contextid) in tqdm(enumerate(train_data), total=len(train_data)):
        rel_rev = rel + num_r
        if rel not in relid2contextid:
            relid2contextid[rel] = set()
        if rel_rev not in relid2contextid:
            relid2contextid[rel_rev] = set()
        relid2contextid[rel].add(contextid)
        relid2contextid[rel_rev].add(contextid)
    return relid2contextid

def get_hypergraph_adj(itemid2contextid, K):
    num_item= len(itemid2contextid)
    hyper_adj = []
    for itemid in range(num_item):
        curr_adj = []
        contextids = itemid2contextid[itemid]
        for r in range(K):
            r_adj = [0.0] * K
            for c in range(K):
                if r in contextids and c in contextids:
                    r_adj[c] = 1.0
            curr_adj.append(r_adj)
        hyper_adj.append(torch.Tensor(curr_adj) / len(contextids))
    hyper_adj = torch.block_diag(*hyper_adj)
    return hyper_adj

def main(args):
    graph_dict = {}

    data_path = args.datapath
    K = args.K

    train_data, train_times = load_quadruples(data_path, 'train_w_contextid.txt')
    val_data, val_times = load_quadruples(data_path, 'valid_w_contextid.txt')
    test_data, test_times = load_quadruples(data_path, 'test_w_contextid.txt')

    with open(os.path.join(data_path, 'stat.txt'), 'r') as f:
        line = f.readline()
        num_nodes, num_r = line.strip().strip('\n').split("\t")
        num_nodes = int(num_nodes)
        num_r = int(num_r)
    print(num_nodes, num_r)


    print('---generate entity hypergraph')
    eid2contextid = get_entid2contextid(train_data)
    ent_hypergraph_adj = get_hypergraph_adj(eid2contextid, K)
    torch.save(ent_hypergraph_adj, data_path + '/hypergraph_ent.pt')


    print('---generate relation hypergraph')
    relid2contextid = get_relid2contextid(train_data, num_r)
    hypergraph_adj_overall = get_hypergraph_adj(relid2contextid, K)
    torch.save(hypergraph_adj_overall, data_path + '/hypergraph_rel.pt')


    print('---generate knowledge graph')
    with tqdm(total=len(train_times), desc="Generating graphs for training") as pbar:
        for tim in train_times:
            data = get_data_with_t(train_data, tim)
            graph_dict[tim] = get_big_graph(data, num_nodes, num_r, K)
            pbar.update(1)

    with tqdm(total=len(val_times), desc="Generating graphs for validating") as pbar:
        for tim in val_times:
            data = get_data_with_t(val_data, tim)
            graph_dict[tim] = get_big_graph(data, num_nodes, num_r, K)
            pbar.update(1)
        
    with tqdm(total=len(test_times), desc="Generating graphs for testing") as pbar:
        for tim in test_times:
            data = get_data_with_t(test_data, tim)
            graph_dict[tim] = get_big_graph(data, num_nodes, num_r, K)
            pbar.update(1)

    with open(os.path.join(data_path, 'graph_dict_each_context.pkl'), 'wb') as fp:
        pickle.dump(graph_dict, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate disentangled graphs')
    parser.add_argument("--datapath", type=str, default="../data_disentangled/EG_LDA_K5",
                        help="disentangled dataset to generate disentangled graphs")
    parser.add_argument("--K", type=int, default=5,
                        help="number of contexts")
    args = parser.parse_args()

    main(args)
