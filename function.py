import pandas as pd
import numpy as np
import re
import time
import json
from datetime import datetime
from datetime import timedelta
import pickle
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from textrank4zh import TextRank4Sentence
import matplotlib.pyplot as plt
from gensim.models import word2vec
from collections import Counter

def data_filter(df):

    df = df[df['content'] != ''].copy()
    df = df.dropna(subset=['content']).copy()
    df = df.reset_index(drop=True)
    return df

def get_data(df, last_time, delta):
    last_time = datetime.strptime(last_time, '%Y/%m/%d %H:%M:%S')
    delta = timedelta(delta)
    try:
        df['time'] = df['time'].map(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))
    except TypeError:
        pass
    df = df[df['time'].map(lambda x: (x <= last_time) and (x > last_time - delta))].copy()
    print('df.shape=', df.shape)
    if df.shape[0] == 0:
        print('No Data!')
        return df
    df = df.sort_values(by=['time'], ascending=[0])
    df['time'] = df['time'].map(lambda x: datetime.strftime(x, '%Y/%m/%d %H:%M:%S'))
    df = df.reset_index(drop=True)
    return df


def clean_content_blank(content):
    content = re.sub(r'\?+', ' ', content)
    content = re.sub(r'\u3000', '', content)
    content = content.strip()
    content = re.sub(r'[ \t\r\f]+', ' ', content)
    content = re.sub(r'\n ', '\n', content)
    content = re.sub(r' \n', '\n', content)
    content = re.sub(r'\n+', '\n', content)
    return content


def clean_content(content):

    content = clean_content_blank(content)
    content = content.lower()
    content = re.sub(r'https?://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', content)
    texts = ['rt']
    for text in texts:
        content = re.sub(text, '', content)
    content = re.sub(r'\n+', '\n', content)
    return content


def get_num_en_ch(text):
    text = re.sub(r'[^@0-9A-Za-z\u4E00-\u9FFF]+', ' ', text)
    text = text.strip()
    return text


def userdict_cut(text, userdict_path=None):

    if userdict_path is not None:
        jieba.load_userdict(userdict_path)
    words = jieba.cut(text)
    return words


def stop_words_cut(words, stop_words_path):
    with open(stop_words_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
        stopwords.append(' ')
        words = [word for word in words if word not in stopwords]
    return words

def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)

def get_word_library(list1):
    list2 = flat(list1)
    list3 = list(set(list2))
    return list3

def get_single_frequency_words(list1):

    list2 = flat(list1)
    cnt = Counter(list2)
    list3 = [i for i in cnt if cnt[i] == 1]
    return list3


def get_most_common_words(list1, top_n=None, min_frequency=1):

    list2 = flat(list1)
    cnt = Counter(list2)
    list3 = [i[0] for i in cnt.most_common(top_n) if cnt[i[0]] >= min_frequency]
    return list3


def get_num_of_value_no_repeat(list1):

    num = len(set(list1))
    return num


def feature_extraction(series, vectorizer='CountVectorizer', vec_args=None):

    vec_args = {'max_df': 1.0, 'min_df': 1} if vec_args is None else vec_args
    vec_args_list = ['%s=%s' % (i[0],
                                "'%s'" % i[1] if isinstance(i[1], str) else i[1]
                                ) for i in vec_args.items()]
    vec_args_str = ','.join(vec_args_list)
    vectorizer1 = eval("%s(%s)" % (vectorizer, vec_args_str))
    matrix = vectorizer1.fit_transform(series)
    return matrix

def get_cluster(matrix, cluster='DBSCAN', cluster_args=None):

    cluster_args = {'eps': 0.5, 'min_samples': 5, 'metric': 'cosine'} if cluster_args is None else cluster_args
    cluster_args_list = ['%s=%s' % (i[0],
                                    "'%s'" % i[1] if isinstance(i[1], str) else i[1]
                                    ) for i in cluster_args.items()]
    cluster_args_str = ','.join(cluster_args_list)
    cluster1 = eval("%s(%s)" % (cluster, cluster_args_str))
    cluster1 = cluster1.fit(matrix)
    return cluster1

def get_labels(cluster):
    labels = cluster.labels_
    return labels

def label2rank(labels_list):
    series = pd.Series(labels_list)
    list1 = series[series != -1].tolist()
    n = len(set(list1))
    cnt = Counter(list1)
    key = [cnt.most_common()[i][0] for i in range(n)]
    value = [i for i in range(1, n + 1)]
    my_dict = dict(zip(key, value))
    my_dict[-1] = -1
    rank_list = [my_dict[i] for i in labels_list]
    return rank_list

def get_non_outliers_data(df, label_column='label'):
    df = df[df[label_column] != -1].copy()
    return df


def get_data_sort_labelnum(df, label_column='label', top=1):
    assert top > 0, 'top不能小于等于0！'
    labels = df[label_column].tolist()
    cnt = Counter(labels)
    label = cnt.most_common()[top - 1][0] if top <= len(set(labels)) else -2
    df = df[df[label_column] == label].copy() if label != -2 else pd.DataFrame(columns=df.columns)
    return df


def list2wordcloud(list1, save_path, font_path):
    text = ' '.join(list1)
    wc = WordCloud(font_path=font_path, width=800, height=600, margin=2,
                   ranks_only=True, max_words=200, collocations=False).generate(text)
    wc.to_file(save_path)


def get_key_sentences(text, num=1):

    tr4s = TextRank4Sentence(delimiters='\n')
    tr4s.analyze(text=text, source='all_filters')
    abstract = '\n'.join([item.sentence for item in tr4s.get_key_sentences(num=num)])
    return abstract


def feature_reduction(matrix, pca_n_components=50, tsne_n_components=2):

    data_pca = PCA(n_components=pca_n_components).fit_transform(matrix) if pca_n_components is not None else matrix
    data_pca_tsne = TSNE(n_components=tsne_n_components).fit_transform(
        data_pca) if tsne_n_components is not None else data_pca
    #print('data_pca_tsne.shape=', data_pca_tsne.shape)
    return data_pca_tsne


def get_wordvec(model, word):
    try:
        model.wv.get_vector(word)
        return True
    except:
        return False

def get_word_and_wordvec(model, words):
    word_list = [i for i in words if get_wordvec(model, i)]
    wordvec_list = [model.wv[i].tolist() for i in words if get_wordvec(model, i)]
    return word_list, wordvec_list

def get_top_words(words, label, label_num):
    df = pd.DataFrame()
    df['word'] = words
    df['label'] = label
    for i in range(label_num):
        df_ = df[df['label'] == i]
        print(df_['word'][:30])

def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def process(filepath =  'data.csv',eps_var = 0.6,min_samples_var = 10):
    df = pd.read_csv(filepath,index_col=0)[:5000]

    df['text'] = df['text'].apply(clean_content_blank)
    df['text'] = df['text'].apply(clean_content)
    df['content'] = df['text'].apply(get_num_en_ch)
    df['content_cut'] = df['text'].map(lambda x:x.split(' '))
    word_library_list = get_word_library(df['content_cut'])
    single_frequency_words_list = get_single_frequency_words(df['content_cut'])
    max_features = (len(word_library_list) - len(single_frequency_words_list))
    matrix = feature_extraction(df['content'], vectorizer='TfidfVectorizer',
                                             vec_args={'max_df': 0.95, 'min_df': 1, 'max_features': max_features})

    dbscan = get_cluster(matrix, cluster='DBSCAN',
                        cluster_args={'eps': eps_var, 'min_samples': min_samples_var, 'metric': 'cosine'})
    labels = get_labels(dbscan)
    df['label'] = labels
    ranks = label2rank(labels)
    df['rank'] = ranks
    df['matrix'] = matrix.toarray().tolist()

    df_non_outliers = df[df['label'] != -1].copy()
    rank_num = get_num_of_value_no_repeat(df_non_outliers['rank'])


    df_non_outliers[df_non_outliers['rank'] == 1]

    data_pca_tsne = feature_reduction(df_non_outliers['matrix'].tolist(),pca_n_components=3, tsne_n_components=2)



    df_non_outliers['pca_tsne'] = data_pca_tsne.tolist()
    del df_non_outliers['matrix']
    data_pca_tsne = df_non_outliers['pca_tsne']
    label = df_non_outliers['label']

    return df_non_outliers,rank_num,data_pca_tsne,label
    


def draw_clustering_analysis_barh(rank_num, value, yticks, title):
    plt.figure(figsize=(13, 6), dpi=100)
    plt.subplot(122)
    ax = plt.gca()
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.invert_yaxis()
    plt.barh(range(1, rank_num + 1), value, align='center', linewidth=0)
    plt.yticks(range(1, rank_num + 1), yticks)
    for a, b in zip(value, range(1, rank_num + 1)):
        plt.text(a + 1, b, '%.0f' % a, ha='left', va='center')
    plt.title(title)
    plt.savefig( '2.jpg')
    plt.show()
    
def draw_clustering_analysis_pie(rank_num, value, yticks,title):
    plt.figure(figsize=(13, 6), dpi=100)
    plt.subplot(132)
    plt.pie(value, explode=[0.2] * rank_num, labels=yticks, autopct='%1.2f%%', pctdistance=0.7)
    plt.title(title)
    plt.show()

    
