import pandas as pd
from collections import Counter
import os
import csv
import argparse
import json


def main(project_path, country):
    print('------generate structured data for country {}'.format(country))

    data_path = project_path + '/data'

    data = pd.read_csv('{}/{}/{}.csv'.format(data_path, country, country), sep='\t')

    # construct train, valid, and test structured files
    unique_time = data['date'].unique() # 2584 days
    train_split = int((data['timid'].max() + 1) * 0.8)
    valid_split = int((data['timid'].max() + 1) * 0.9)

    unique_url = set()
    for md5_list in data['Md5_list']:
        unique_url.update(md5_list.split(', '))
    json.dump(list(unique_url), open('{}/{}/md5_list.json'.format(data_path, country), 'w'), indent=4)

    unique_entity = data['Actor1Name'].append(data['Actor2Name']).unique()
    unique_relation = data['EventCode'].unique()
    ent2id = {name: idx for idx, name in enumerate(unique_entity)}
    rel2id = {name: idx for idx, name in enumerate(unique_relation)}

    output_df = pd.DataFrame()
    output_df['actor1id'] = data['Actor1Name'].map(lambda x: ent2id[x])
    output_df['eventid'] = data['EventCode'].map(lambda x: rel2id[x])
    output_df['actor2id'] = data['Actor2Name'].map(lambda x: ent2id[x])
    output_df['timid'] = data['timid']
    output_df['Md5_list'] = data['Md5_list']

    output_train = output_df[output_df['timid'] <= train_split]
    output_val = output_df[(output_df['timid'] > train_split) & (output_df['timid'] <= valid_split)]
    output_test = output_df[output_df['timid'] > valid_split]

    unique_train_url = set()
    for md5_list in output_train['Md5_list']:
        unique_train_url.update(md5_list.split(', '))
    json.dump(list(unique_train_url), open('{}/{}/train_md5_list.json'.format(data_path, country), 'w'), indent=4)

    # popularity bias
    train_stat_ent = Counter(output_train['actor1id'].append(output_train['actor2id']))
    freq_head_ents = sum(train_stat_ent.values()) // 3
    train_stat_ent = sorted(train_stat_ent.items(), key=lambda item: item[1], reverse=True)
    head_ents = []
    total_freq_ent = 0
    for entity_id, freq in train_stat_ent:
        if total_freq_ent <= freq_head_ents:
            head_ents.append(entity_id)
            total_freq_ent += freq
        else:
            break

    output_path = '{}/{}'.format(data_path, country)
    output_train.to_csv(output_path + '/train_w_md5s.txt', header=None, index=None, sep='\t')
    output_val.to_csv(output_path + '/valid_w_md5s.txt', header=None, index=None, sep='\t')
    output_test.to_csv(output_path + '/test_w_md5s.txt', header=None, index=None, sep='\t')

    del output_train['Md5_list']
    del output_val['Md5_list']
    del output_test['Md5_list']

    output_train.to_csv(output_path + '/train.txt', header=None, index=None, sep='\t')
    output_val.to_csv(output_path + '/valid.txt', header=None, index=None, sep='\t')
    output_test.to_csv(output_path + '/test.txt', header=None, index=None, sep='\t')

    with open((os.path.join(output_path, 'entity2id.txt')), 'w') as f:
        temp_csv_writer = csv.writer(f, delimiter='\t')
        for k, v in ent2id.items():
            temp_csv_writer.writerow([k, v])

    with open((os.path.join(output_path, 'relation2id.txt')), 'w') as f:
        temp_csv_writer = csv.writer(f, delimiter='\t')
        for k, v in rel2id.items():
            temp_csv_writer.writerow([k, v])

    with open((os.path.join(output_path, 'stat.txt')), 'w') as f:
        f.write(str(len(ent2id)) + '\t' + str(len(rel2id)))

    json.dump(head_ents, open(output_path + '/head_ents.json', 'w'), indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate structured data')
    parser.add_argument("--project_path", type=str, default="..",
                        help="project root path")
    parser.add_argument("--c", type=str, default="EG",
                        help="country: EG, IR, or IS")

    args = parser.parse_args()

    main(args.project_path, args.c)