import argparse
import os
import sys
import pickle
import logging

from torch.utils.tensorboard import SummaryWriter

import torch
import json
import numpy as np
from tqdm import tqdm
import random

sys.path.append("..")
from src import utils
from src.seco import SeCo


def test(args, model, model_name,
         history_times, query_times, graph_dict,
         test_list, test_context_list,
         all_ans_list, head_ents,
         use_cuda, mode='eval'):
    """
    :param model: model used to test
    :param model_name: model state file name
    :param history_times: all time stamps in dataset
    :param query_times: all time stamps in testing dataset
    :param graph_dict: all graphs per day in dataset
    :param test_list: test triple snaps list
    :param test_context_list: test context prob list
    :param all_ans_list: dict used for time-aware filtering (key and value are all int variable not tensor)
    :param head_ents: extremely frequent head entities causing popularity bias
    :param use_cuda: use cuda or cpu
    :param mode: 'eval' used in training process; or 'test' used for testing the best checkpoint
    :return: mrr for event object prediction
    """
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        logging.info("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint[
            'epoch']))  # use best stat checkpoint
        logging.info("\n" + "-" * 10 + "start testing" + "-" * 10 + "\n")
        model.load_state_dict(checkpoint['state_dict'])

    rank_filter_list, mrr_filter_list = [], []
    tags, tags_all = [], []

    model.eval()

    with torch.no_grad():

        for time_idx, test_snap in enumerate(tqdm(test_list)):
            query_time = query_times[time_idx]
            query_idx = np.where(history_times == query_time)[0].item()
            input_time_list = history_times[query_idx - args.train_history_len: query_idx]
            history_glist = [graph_dict[tim] for tim in input_time_list]

            # load test triplets: ( (s, r, o), ... ), len = all triplet in the same day
            test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
            test_triples_input = test_triples_input.to(args.gpu)
            # load test contexts: hard: ( (1,0,0,0,0), ... ), ... ), len = all triplet in the same day
            test_context_input = test_context_list[time_idx]
            test_context_input = torch.from_numpy(test_context_input).float().cuda() if use_cuda else torch.from_numpy(
                test_context_input).float()
            test_context_input = test_context_input.to(args.gpu)

            test_triples, final_score = model.predict(history_glist, test_triples_input, use_cuda,
                                                    test_contexts=test_context_input)

            mrr_filter, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000)

            popularity_tag = list(map(lambda x: utils.popularity_map(x, head_ents), test_triples))
            tags_all.append(popularity_tag)

            rank_filter_list.append(rank_filter)
            mrr_filter_list.append(mrr_filter)

    mrr_filter_all = utils.cal_ranks(rank_filter_list, tags_all, mode)

    return mrr_filter_all


def run_experiment(args):
    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(
        data.train)  # [((s, r, o), ...), ...] len = num_date, do not contain inverse triplets
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)
    train_times = np.array(sorted(set(data.train[:, 3])))
    val_times = np.array(sorted(set(data.valid[:, 3])))
    test_times = np.array(sorted(set(data.test[:, 3])))
    history_times = np.concatenate((train_times, val_times, test_times), axis=None)

    num_nodes = data.num_nodes
    num_rels = data.num_rels
    print(num_nodes, num_rels)

    # data for time-aware filtering
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes)
    # [ { e1: {r: (e2)} } ], len = uniq_t in given dataset

    # load popularity bias data
    print("loading popularity bias data")
    head_ents = json.load(open('../data/{}/head_ents.json'.format(args.dataset), 'r'))

    # load disentangled graph data (sub-embeddings)
    disentangled_dataset = '{}_{}_K{}'.format(args.dataset, args.context, args.k_contexts)
    print("loading context data from " + disentangled_dataset)
    context_data = utils.load_data(disentangled_dataset)

    train_context_list, valid_context_list, test_context_list = None, None, None
    if args.score_aggregation == 'hard':
        print("loading context onehot list")
        # [[(1,0,0,0,0) onehot of contextid, ...], ...] len = num_date
        train_context_list = utils.split_context_by_time_onehot(context_data.train, args.k_contexts)
        valid_context_list = utils.split_context_by_time_onehot(context_data.valid, args.k_contexts)
        test_context_list = utils.split_context_by_time_onehot(context_data.test, args.k_contexts)
    elif args.score_aggregation == 'avg':
        print("loading context average list")
        # [[(0.2, 0.2, 0.2, 0.2, 0.2), ...], ...] len = num_date
        train_context_list = utils.split_context_by_time_avg(context_data.train, args.k_contexts)
        valid_context_list = utils.split_context_by_time_avg(context_data.valid, args.k_contexts)
        test_context_list = utils.split_context_by_time_avg(context_data.test, args.k_contexts)

    # load hyper graph
    hyper_adj_ent, hyper_adj_rel = None, None
    if args.hypergraph_ent:
        print("loading hypergraph adjacency matrix: entity")
        hyper_adj_ent = torch.load("../data_disentangled/{}/hypergraph_ent.pt".format(disentangled_dataset))
    if args.hypergraph_rel:
        print("loading hypergraph adjacency matrix: relation")
        hyper_adj_rel = torch.load("../data_disentangled/{}/hypergraph_rel.pt".format(disentangled_dataset))

    # logging
    print("build results directories")
    hypergraph_ent_naming = '_hgent{}'.format(args.n_layers_hypergraph_ent) if args.hypergraph_ent else ''
    hypergraph_rel_naming = '_hgrel{}'.format(args.n_layers_hypergraph_rel) if args.hypergraph_rel else ''
    hypergraph_naming = hypergraph_ent_naming + hypergraph_rel_naming

    score_naming = '_' + args.score_aggregation
    encoder_naming = '{}_n{}_h{}'.format(args.encoder, args.n_layers, args.n_hidden)
    train_naming = '_t{}_lr{}_wd{}'.format(args.train_history_len, args.lr, args.wd)
    model_name = encoder_naming + hypergraph_naming + score_naming + train_naming

    log_path = '../results/{}/{}'.format(args.dataset, disentangled_dataset)
    filename = '../results/{}/{}/{}{}.log'.format(
        args.dataset, disentangled_dataset, model_name, args.alias)

    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    logging.basicConfig(level=logging.INFO, filename=filename)

    # runs
    run_path = '../runs_search' if args.param_search else '../runs'
    run_path += "/" + args.dataset + "/" + disentangled_dataset + "/" + model_name + args.alias

    if not os.path.isdir(run_path):
        os.makedirs(run_path)

    run = SummaryWriter(run_path)

    # models
    model_path = '../models/{}/{}'.format(args.dataset, disentangled_dataset)
    model_state_file = model_path + '/' + model_name + args.alias
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    logging.info("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    # create stat
    model = SeCo(args.decoder,
                 args.encoder,
                 num_nodes,
                 num_rels,
                 hyper_adj_ent,
                 hyper_adj_rel,
                 args.n_layers_hypergraph_ent,
                 args.n_layers_hypergraph_rel,
                 args.k_contexts,
                 args.n_hidden,
                 sequence_len=args.train_history_len,
                 num_bases=args.n_bases,
                 num_hidden_layers=args.n_layers,
                 dropout=args.dropout,
                 self_loop=args.self_loop,
                 layer_norm=args.layer_norm,
                 input_dropout=args.input_dropout,
                 hidden_dropout=args.hidden_dropout,
                 feat_dropout=args.feat_dropout,
                 use_cuda=use_cuda,
                 gpu=args.gpu)

    print(model)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()
        if args.hypergraph_ent:
            model.hyper_adj_ent = model.hyper_adj_ent.to(args.gpu)
        if args.hypergraph_rel:
            model.hyper_adj_rel = model.hyper_adj_rel.to(args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    graph_dict = None
    print("loading train, valid, test graphs...")
    with open(os.path.join('../data_disentangled/' + disentangled_dataset, 'graph_dict_each_context.pkl'), 'rb') as fp:
        graph_dict = pickle.load(fp)

    if args.test and os.path.exists(model_state_file):
        print("----------------------------------------start testing----------------------------------------\n")
        test(args,
             model=model,
             model_name=model_state_file,
             history_times=history_times,
             query_times=test_times,
             graph_dict=graph_dict,
             test_list=test_list,
             test_context_list=test_context_list,
             all_ans_list=all_ans_list_test,
             head_ents=head_ents,
             use_cuda=use_cuda,
             mode="test")
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(
            model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_val_mrr, best_test_mrr = 0, 0
        best_epoch = 0
        accumulated = 0
        for epoch in range(args.n_epochs):
            model.train()
            losses = []

            idx = [_ for _ in range(len(train_list))]
            batch_cnt = len(idx)
            epoch_anchor = epoch * batch_cnt
            random.shuffle(idx)  # shuffle based on time

            for batch_idx, train_sample_num in enumerate(tqdm(idx)):
                batch_anchor = epoch_anchor + batch_idx
                if train_sample_num == 0: continue  # make sure at least one history graph
                # train_list : [((s, r, o) on the same day)], len = uniq_t in train
                output = train_list[train_sample_num]  # all triplets in the next day to be predicted
                # train_context_list: hard: [[(1,0,0,0,0) onehot of contextid, ... on the same day]], len = uniq_t in train
                output_context = train_context_list[train_sample_num]
                # hard: [(1,0,0,0,0) onehot of contextid, ... on the same day], len = 1
                output_context = torch.from_numpy(output_context).float().cuda() if use_cuda else torch.from_numpy(
                    output_context).float()
                if train_sample_num - args.train_history_len < 0:
                    input_list = train_times[0: train_sample_num]
                else:
                    input_list = train_times[train_sample_num - args.train_history_len: train_sample_num]

                # generate history graph
                history_glist = [graph_dict[tim] for tim in input_list]  # [(g), ...], len = valid history length

                output = torch.from_numpy(output).long().cuda() if use_cuda else torch.from_numpy(output).long()

                loss = model(history_glist, output, use_cuda, output_context)

                losses.append(loss.item())
                run.add_scalar('loss/loss_all', loss.item(), batch_anchor)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print("Epoch {:04d}, AveLoss: {:.4f}, BestValMRR {:.4f}, BestTestMRR: {:.4f}, Model: {}, Dataset: {} "
                  .format(epoch, np.mean(losses), best_val_mrr, best_test_mrr, model_name, args.dataset))

            # validation and test
            if (epoch + 1) and (epoch + 1) % args.evaluate_every == 0:
                val_mrr = test(args,
                               model=model,
                               model_name=model_state_file,
                               history_times=history_times,
                               query_times=val_times,
                               graph_dict=graph_dict,
                               test_list=valid_list,
                               test_context_list=valid_context_list,
                               all_ans_list=all_ans_list_valid,
                               head_ents=head_ents,
                               use_cuda=use_cuda,
                               mode="eval")
                run.add_scalar('val/mrr', val_mrr, epoch)

                test_mrr = test(args,
                                model=model,
                                model_name=model_state_file,
                                history_times=history_times,
                                query_times=test_times,
                                graph_dict=graph_dict,
                                test_list=test_list,
                                test_context_list=test_context_list,
                                all_ans_list=all_ans_list_test,
                                head_ents=head_ents,
                                use_cuda=use_cuda,
                                mode="eval")
                run.add_scalar('test/mrr', test_mrr, epoch)

                if val_mrr < best_val_mrr:
                    accumulated += 1
                    if epoch >= args.n_epochs:
                        print("Max epoch reached! Training done.")
                        break
                    if accumulated >= args.patience:
                        print("Early stop triggered! Training done at epoch{}, best epoch is {}".format(epoch,
                                                                                                        best_epoch))
                        break
                else:
                    accumulated = 0
                    best_val_mrr = val_mrr
                    best_test_mrr = test_mrr
                    best_epoch = epoch
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

        print('--- test best epoch model at epoch {}'.format(best_epoch))
        test(args,
             model=model,
             model_name=model_state_file,
             history_times=history_times,
             query_times=test_times,
             graph_dict=graph_dict,
             test_list=test_list,
             test_context_list=test_context_list,
             all_ans_list=all_ans_list_test,
             head_ents=head_ents,
             use_cuda=use_cuda,
             mode="test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SeCoGD')

    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="which country's dataset to use: EG/ IR/ IS")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")

    # configuration for context
    parser.add_argument("--context", type=str, default='',
                        help="context clustering method: LDA/ KMeans / GMM")
    parser.add_argument("--k_contexts", type=int, default=5,
                        help="number of contexts to disentangle the sub-embeddings")

    # configuration for cross-context hypergraph
    parser.add_argument("--hypergraph_ent", action='store_true', default=False,
                        help="add hypergraph between disentangled nodes")
    parser.add_argument("--hypergraph_rel", action='store_true', default=False,
                        help="add hypergraph between disentangled relations")
    parser.add_argument("--n_layers_hypergraph_ent", type=int, default=1,
                        help="number of propagation rounds on entity hypergraph")
    parser.add_argument("--n_layers_hypergraph_rel", type=int, default=1,
                        help="number of propagation rounds on relation hypergraph")
    parser.add_argument("--score_aggregation", type=str, default='hard',
                        help="score aggregation strategy: hard/ avg")

    # configuration for context specific encoder
    parser.add_argument("--encoder", type=str, default="rgcn",
                        help="method of encoder: rgcn/ compgcn")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--n_hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--n_bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--self_loop", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer_norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input_dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden_dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat_dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train_history_len", type=int, default=3,
                        help="history length")

    # configuration for stat training
    parser.add_argument("--n_epochs", type=int, default=40,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--patience", type=int, default=5,
                        help="early stop patience")
    parser.add_argument("--evaluate_every", type=int, default=1,
                        help="perform evaluation every n epochs")
    parser.add_argument("--param_search", action='store_true', default=False,
                        help="perform parameter search, affects runs saving path")
    parser.add_argument("--alias", type=str, default='',
                        help="model naming alias, better start with _")

    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-6,
                        help="weight decay")
    parser.add_argument("--grad_norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    args = parser.parse_args()
    print(args)

    run_experiment(args)
    sys.exit()
