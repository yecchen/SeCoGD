from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import string
import json
import argparse
from tqdm import tqdm


def main(project_path, country):
    print('------generate clean text data for country {}'.format(country))

    docs_title_paragraph = json.load(open('{}/data/{}/docs_title_paragraph.json'.format(project_path, country), 'r'))
    # [ [titile, [p1, p2, ...]], ...]
    md5_list = json.load(open('{}/data/{}/md5_list.json'.format(project_path, country), 'r'))
    train_md5_list = json.load(open('{}/data/{}/train_md5_list.json'.format(project_path, country), 'r'))

    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    have_unicode = lambda body: any(ord(x) > 127 for x in body)
    manuals = set(["'s", '``', "''"])

    def clean(tokens):
        final_tokens = []
        for token in tokens:
            if token in stop or token in exclude or have_unicode(token) or token in manuals:
                continue
            else:
                final_tokens.append(lemma.lemmatize(token))
        return final_tokens

    docs_cleaned_tokens = []  # [ [token1, token2, ...], ...]

    for idx, md5 in tqdm(enumerate(md5_list), total=len(md5_list)):
        sents, cleaned_tokens = [], []
        doc_title_paragraph = docs_title_paragraph[idx]
        sents.append(doc_title_paragraph[0])  # title
        for paragraph in doc_title_paragraph[1]:
            sents += sent_tokenize(paragraph)
        for sent in sents:
            cleaned_sent_tokens = clean(sent.split())
            cleaned_tokens += cleaned_sent_tokens
        docs_cleaned_tokens.append(cleaned_tokens.copy())

    json.dump(docs_cleaned_tokens, open('{}/data/{}/docs_cleaned_tokens.json'.format(project_path, country), 'w'),
              indent=4)

    md52docid = {}
    for idx, md5 in enumerate(md5_list):
        md52docid[md5] = idx

    train_docs_cleaned_tokens = []  # [ [token1, token2, ...], ...]

    for idx, md5 in tqdm(enumerate(train_md5_list), total=len(train_md5_list)):
        train_docs_cleaned_tokens.append(docs_cleaned_tokens[md52docid[md5]])

    json.dump(train_docs_cleaned_tokens,
              open('{}/data/{}/train_docs_cleaned_tokens.json'.format(project_path, country), 'w'), indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate clean text data')
    parser.add_argument("--project_path", type=str, default="..",
                        help="project root path")
    parser.add_argument("--c", type=str, default="EG",
                        help="country: EG, IR, or IS")

    args = parser.parse_args()

    main(args.project_path, args.c)