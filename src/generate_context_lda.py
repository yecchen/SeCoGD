import json
import argparse
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import gensim
from gensim import corpora
from gensim.test.utils import datapath
Lda = gensim.models.ldamodel.LdaModel

SEED = 2023

def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
#             d[int(line[1])] = line[0]
            d[line[0]] = int(line[1])
    return d

def main(project_path, c, K_TOPICS):
    print('------generate context data for country {}'.format(c))

    train_md5_list = json.load(open('{}/data/{}/train_md5_list.json'.format(project_path, c), 'r'))
    md5_list = json.load(open('{}/data/{}/md5_list.json'.format(project_path, c), 'r'))
    all_docs_lda = json.load(open('{}/data/{}/docs_cleaned_tokens.json'.format(project_path, c), 'r'))
    train_docs_lda = json.load(open('{}/data/{}/train_docs_cleaned_tokens.json'.format(project_path, c), 'r'))

    md52docid = {}
    for idx, md5 in enumerate(md5_list):
        md52docid[md5] = idx

    # dictionary
    fdic = '{}/data/{}/lda_bow_dictionary'.format(project_path, c)
    dictionary = corpora.Dictionary(train_docs_lda)
    dictionary.save_as_text(datapath(fdic))

    doc_term_matrix = [dictionary.doc2bow(doc) for doc in train_docs_lda]

    ldamodel = Lda(doc_term_matrix, num_topics=K_TOPICS, id2word=dictionary, random_state=SEED)
    fmodel = '{}/data/{}/lda_bow_model_K{}'.format(project_path, c, K_TOPICS)
    ldamodel.save(datapath(fmodel))

    print(ldamodel.print_topics(num_topics=K_TOPICS, num_words=10))

    # generate train, valid, test with topicid
    data_df = pd.read_csv(project_path + '/data/{}/{}.csv'.format(c, c), sep='\t')
    topicid = []
    md5_lists = data_df['Md5_list'].tolist()

    all_docs_topic_weight = []
    for idx, doc in tqdm(enumerate(all_docs_lda), total=len(all_docs_lda)):
        curr_doc = dictionary.doc2bow(doc)
        curr_probs = ldamodel[curr_doc]
        curr_weight = [0.0] * 5
        for topic_id, topic_prob in curr_probs:
            curr_weight[topic_id] = topic_prob
        all_docs_topic_weight.append(curr_weight.copy())

    for idx, md5_list in tqdm(enumerate(md5_lists), total=len(md5_lists)):
        md5s = md5_list.split(', ')
        if len(md5s) == 0:
            curr_md5 = md5s[0]
            curr_topic_weight = all_docs_topic_weight[md52docid[curr_md5]]
        else:
            curr_topic_weight = []
            for curr_md5 in md5s:
                curr_topic_weight.append(all_docs_topic_weight[md52docid[curr_md5]].copy())
            curr_topic_weight = np.array(curr_topic_weight)
            curr_topic_weight = np.sum(curr_topic_weight, axis=0) / len(md5s)
        topicid.append(np.argmax(curr_topic_weight))

    save_path = project_path + '/data_disentangled/{}_LDA_K{}'.format(c, K_TOPICS)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    data_df['topicid'] = topicid
    data_df.to_csv(path_or_buf='{}/data_disentangled/{}_LDA_K{}/{}.csv'.format(
        project_path, c, K_TOPICS, c), sep='\t', index=False)

    train_split = int((data_df['timid'].max() + 1) * 0.8)
    valid_split = int((data_df['timid'].max() + 1) * 0.9)

    ent2id = _read_dictionary(project_path + '/data/{}/entity2id.txt'.format(c))
    rel2id = _read_dictionary(project_path + '/data/{}/relation2id.txt'.format(c))
    with open('/{}/data_disentangled/{}_LDA_K{}/stat.txt'.format(project_path, c, K_TOPICS), 'w') as f:
        f.write(str(len(ent2id)) + '\t' + str(len(rel2id)))

    output_df = pd.DataFrame()
    output_df['actor1id'] = data_df['Actor1Name'].map(lambda x: ent2id[x])
    output_df['eventid'] = data_df['EventCode'].map(lambda x: rel2id[str(x)])
    output_df['actor2id'] = data_df['Actor2Name'].map(lambda x: ent2id[x])
    output_df['timid'] = data_df['timid']
    output_df['topicid'] = data_df['topicid']

    output_train = output_df[output_df['timid'] <= train_split]
    output_val = output_df[(output_df['timid'] > train_split) & (output_df['timid'] <= valid_split)]
    output_test = output_df[output_df['timid'] > valid_split]

    output_train.to_csv(save_path + '/train_w_topicid.txt', header=None, index=None, sep='\t', mode='a')
    output_val.to_csv(save_path + '/valid_w_topicid.txt', header=None, index=None, sep='\t', mode='a')
    output_test.to_csv(save_path + '/test_w_topicid.txt', header=None, index=None, sep='\t', mode='a')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate event and structured data with context information')
    parser.add_argument("--project_path", type=str, default="..",
                        help="project root path")
    parser.add_argument("--c", type=str, default="EG",
                        help="country: EG, IR, or IS")
    parser.add_argument("--K_TOPICS", type=int, default=5,
                        help="number of contexts")
    args = parser.parse_args()

    main(args.project_path, args.c, args.K_TOPICS)