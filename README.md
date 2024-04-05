# SeCoGD
This is the code and data for KDD'23 paper SeCoGD:
["Context-aware Event Forecasting via Graph Disentanglement"](https://arxiv.org/abs/2308.06480) by Yunshan Ma, Chenchen Ye, Zijian Wu, Xiang Wang, Yixin Cao, and Tat-Seng Chua.

## Environment

The code is tested to be runnable under the environment with
python=3.9; pytorch=1.12; cuda=11.3.

To create environment, you could use commands as below:
```
conda create --name seco python=3.9

conda activate seco

conda install pytorch==1.12.0 -c pytorch
pip install pandas
pip install tensorboard
conda install tqdm
conda install -c dglteam dgl-cuda11.3
pip install matplotlib
pip install nltk
pip install gensim
```

## Dataset

The input data files need to be unzipped first:
```
unzip data.zip
unzip data_disentangled.zip
```

The folder `./data` contains the original event data and the structured data.  
The folder `./data_disentangled` contains extra context data.   
We analyze the CAMEO event ontology and convert it into python dictionaries in `./data/CAMEO`.



## Structured Data Generation

---
**_NOTE:_**
- `[country_name]` is `EG` or `IR` or `IS`.
- In the dataset, each md5 corresponds to a news URL. This conversion is done by `md5 = hashlib.md5(URL.encode()).hexdigest()`.
- All scripts are running under `./src`.
---

The original event data is stored in `./data/[country_name]/[country_name].csv`.  
The generated structured data files are also put under `./data/[country_name]`.

**To self-generate the structured data:**
- Run the data generation script:
    ```
    python generate_structured_data.py --c [country_name]
    ```

## Context Generation

---
**_NOTE:_**
- `[context_generation_method]` is `LDA` here, and it can be replaced with other text clustering methods, such as K-Means and GMM.
- `[K]` is the number of contexts, i.e. the number of topics in LDA. We provide data for `K=5` as example.
---

The new event data with context information is stored in `./data_disentangled/[country_name]_[context_generation_method]_K[K]/[country_name].csv`.  
The files for generated structured data with context information are also put under `./data_disentangled/[country_name]_[context_generation_method]_K[K]`.


**To self-generate the contexts:**
1. Prepare text data:
  - The original news articles:
    - Location: `./data/[country_name]/docs_title_paragraph.json` 
    - Format: `[ [title, [paragraph1, paragraph2, ...]], ... ]`
    - Order: `./data/[country_name]/md5_list.json`
  - Clean text data for LDA model learning and topic prediction:
    ```
    python generate_clean_text.py --c [country_name]
    ```
  - The cleaned text data will be saved as:
    - Location: `./data/[country_name]/train_docs_cleaned_tokens.json` and `./data/[country_name]/docs_cleaned_tokens.json`
    - Format: `[ [token1, token2, ...], ...]`
    - Order: `./data/[country_name]/train_md5_list.json` and `./data/[country_name]/md5_list.json` respectively.
  
2. Run context generation script:

    ```
    python generate_context_lda.py --c [country_name] --K_TOPICS [K]
    ```


## Experiment for Context-aware Event Forecasting

1. Create graph data:
    ```
    python generate_graphs.py --datapath ../data/[country_name]
    python generate_graphs_disentangled.py --datapath ../data_disentangled/[country_name]_[context_generation_method]_K[K] --K [K]
    ```

2. Train SeCoGD model:
    ```
    python main.py -d [country_name] \
    --context [context_generation_method] \
    --k_contexts [K] \
    --hypergraph_ent \
    --n_layers_hypergraph_ent 1 \
    --hypergraph_rel \
    --n_layers_hypergraph_rel 1 \
    --score_aggregation hard \
    --encoder rgcn \
    --n_layers 2 \
    --n_hidden 200 \
    --self_loop \
    --layer_norm \
    --train_history_len 3 \
    --lr 0.001 \
    --wd 1e-6
    ```

3. View results:
- Loss and runs:
    ```
    tensorboard --logdir=./runs/
    ```
- Logging results will be saved under: `./results/[country_name]/[country_name]_[context_generation_method]_K[K]/`
