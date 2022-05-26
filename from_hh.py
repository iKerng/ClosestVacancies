import hh_token
import pandas as pd
from os import path, getcwd, getenv, environ
import sqlite3 as sql
import requests as rq
import bs4 as bs
import numpy as np

from nltk.tokenize import WordPunctTokenizer
from nltk.stem import snowball
from nltk.corpus import stopwords
from nltk import download
from pymorphy2 import MorphAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from transformers import AutoTokenizer, AutoModel
import torch


def clean_tokenize(text=''):
    text = bs.BeautifulSoup(text, 'html.parser').get_text().lower()
    tokenizer = WordPunctTokenizer()
    ls_token = tokenizer.tokenize(text)
    rus_stem = snowball.SnowballStemmer(language='russian')
    morph = MorphAnalyzer()
    ls_new = []
    for word in ls_token:
        # приводим слово к нормальной форме
        morphed = morph.parse(word)[0].normal_form
        # stem_word = rus_stem.stem(morphed)
        # отбираем слова, которые несут смысловую нагрузку
        if morphed not in stopwords.words('russian') and morphed.isalpha():
            ls_new.append(rus_stem.stem(morphed))
    return ' '.join(ls_new)

def run_tfidf(cleared_user_text='', vacancies=pd.df(columns=['cleared_vacs'])):
    print('TF-IDF: запускаем обработку')
    tfidf_vec = TfidfVectorizer(ngram_range=(2, 2), analyzer='word')
    # обучаемся
    tfidf_vec.fit([' '.join(vacancies['cleared_vacs'].to_list()) + ' ' + ' '.join(cleared_user_text)])
    # формируем векторы
    user_vec = tfidf_vec.transform([cleared_user_text])
    vacs_vec = tfidf_vec.transform(vacancies.cleared_vacs.to_list())
    # сравниваем векторы и получаем список схожести
    result_pair = cosine_similarity(user_vec, vacs_vec)[0]
    # ранжируем результаты сравнения
    tf_idf_top10 = pd.Series(result_pair).sort_values(ascending=False).index.to_list()[0:10]
    print('TF-IDF: завершили')

    return vacancies['url'].iloc[tf_idf_top10].to_list()

def run_word2vec(cleared_user_text='', vacancies=pd.df(columns=['cleared_vacs'])):
    print('Word2Vec: запускаем обработку')
    # задаем параметры для модели
    max_epochs = 30
    vec_size = 125
    alpha = 0.025
    data = [cleared_user_text] + vacancies['cleared_vacs'].to_list()
    # todo: добавить комментарии по модели word2vec
    tagged_data = [TaggedDocument(words=sentence.split(' '), tags=[str(i)]) for i, sentence in enumerate(data)]
    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)
    # todo: добавить комментарии по модели word2vec
    model.build_vocab(tagged_data)
    # todo: добавить комментарии по модели word2vec
    for epoch in range(max_epochs):
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs + 1)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha
    # токенизируем сообщение пользователя
    user_data = cleared_user_text.split(' ')
    # формируем вектор пользовательского описания
    user_vec = model.infer_vector(user_data)
    # формируем векторы описания вакансий
    vac_vecs = [model.infer_vector(x.split(' ')) for x in vacancies['cleared_vacs'].to_list()]
    # сравниваем векторы и получаем список схожести
    res = model.dv.cosine_similarities(user_vec, vac_vecs)
    # ранжируем результаты сравнения
    res_top_list = pd.Series(res).sort_values(ascending=False).index.to_list()[:10]
    print('Word2Vec: завершили')

    return vacancies['url'].iloc[res_top_list]

def run_sbert(cleared_user_text='', vacancies=pd.df(columns=['cleared_vacs'])):
    print('S-BERT: запускаем обработку')
    # todo: добавить комментарии по модели sbert
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    all_texts = [cleared_user_text] + vacancies['cleared_vacs'].to_list()
    # todo: добавить комментарии по модели sbert
    tokens = tokenizer(all_texts,
                       max_length=128,
                       truncation=True,
                       padding='max_length',
                       return_tensors='pt')
    # todo: добавить комментарии по модели sbert
    outputs = model(**tokens)
    # todo: добавить комментарии по модели sbert
    embeddings = outputs.last_hidden_state
    # todo: добавить комментарии по модели sbert
    mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
    # todo: добавить комментарии по модели sbert
    masked_embeddings = embeddings * mask
    # todo: добавить комментарии по модели sbert
    summed = torch.sum(masked_embeddings, 1)
    # todo: добавить комментарии по модели sbert
    counted = torch.clamp(mask.sum(1), min=1e-9)
    # todo: добавить комментарии по модели sbert
    mean_pooled = summed / counted
    # todo: добавить комментарии по модели sbert
    mean_pooled = mean_pooled.detach().numpy()
    # todo: добавить комментарии по модели sbert
    scores = np.zeros((mean_pooled.shape[0], mean_pooled.shape[0]))
    # todo: добавить комментарии по модели sbert
    scores[0, :] = cosine_similarity([mean_pooled[0]], mean_pooled)[0]
    # получаем список схожести
    res = list((scores[0])[1:])
    # ранжируем результаты сравнения
    list_res = pd.Series(res).sort_values(ascending=False).index.to_list()[:10]
    print('S-BERT: завершили')

    return vacancies['url'].iloc[list_res]

def hh_api_get_vacancies(params=dict(), user_text='', id=0):
    # sqlite connect params
    db_path = path.abspath(getcwd()) + '\\data\\vacancies.db'
    print(db_path)
    connect = sql.connect(db_path)

    # default parametrs for search vacancies
    token = hh_token.get_token()
    link_api_hh = 'https://api.hh.ru/'
    date_from = '1970-01-02'
    page = 0
    per_page = 100
    limit = 1000
    ls_columns = ['id', 'name', 'url', 'api_url']
    filter_vacs_dict = {'Authorization': token, 'per_page': per_page, 'page': page, 'date_from': date_from
        , 'order_by': 'publication_time', }
    filter_vacs_dict.update(params)

    # запускаем обработку вакансий
    if int(getenv('is_razmetka')):
    # для разметки данных берем данные локально из БД
        vacancies = pd.read_sql('select * from vacancies_'+str(id), con=connect, index_col='id')
        environ['model'] = '0'
    else:
    # а для режима пром - берем данные с HH.ru
        vacancies = pd.DataFrame()
        rs_get = rq.get(url=link_api_hh+'vacancies/', params=filter_vacs_dict).json()
        ls_vacancies = rs_get.get('items')
        count_vacs = rs_get.get('found')
        iters = [x for x in range(int(count_vacs/per_page)+1)][1:int(limit/per_page)]
        for iter_page in iters:
            filter_vacs_dict.update({'page': iter_page})
            ls_vacancies = ls_vacancies + rq.get(url=link_api_hh+'vacancies/', params=filter_vacs_dict).json().get('items')
        for i in range(len(ls_vacancies)):
            if type(ls_vacancies[i]) == type(list()):
                print(i)
                continue
            new_line = pd.DataFrame(
                [[(ls_vacancies[i]).get('id'), (ls_vacancies[i]).get('name')
                     , (ls_vacancies[i]).get('alternate_url'), (ls_vacancies[i]).get('url')]]
                , columns=ls_columns
                , index=[i])
            vacancies = pd.concat([vacancies, new_line], axis=0)
        vacancies['description'] = vacancies['api_url'].apply(lambda x: rq.get(x).json().get('description'))
        vacancies.to_sql('vacancies_'+str(id), con=connect, if_exists='replace', index=False)
    connect.close()

    # предобработка текста
    vacancies['cleared_vacs'] = vacancies['description'].apply(lambda x: clean_tokenize(x))
    cleared_user_text = clean_tokenize(user_text)

    # рабочая модель (для разметки используем весь набор)
    mode_rezhim = int(getenv('model'))
    if mode_rezhim == 0:
        tfidf = run_tfidf(cleared_user_text, vacancies)
        word2vec = run_word2vec(cleared_user_text, vacancies)
        run_sber = run_sbert(cleared_user_text, vacancies)
        return tfidf + word2vec + run_sber
    elif mode_rezhim == 1:
        return run_tfidf(cleared_user_text, vacancies)
    elif mode_rezhim == 2:
        return run_word2vec(cleared_user_text, vacancies)
    elif mode_rezhim == 3:
        return run_sbert(cleared_user_text, vacancies)
    else:
        return ['Проблема с настройками. Поиск невозможен']


if __name__ == '__main__':
    # данные для разметки
    download('stopwords')
    user_text = 'хочу быть data scientist, окончил школу data analytic и школу data scientist корпоративного ' \
                'университета сбербанка. очень хорошо знаю sql от написания сложных запросов с использование ' \
                'оконных функций, регулярных выражений, иерархических запросов, до pl/sql скриптов. Есть профиль ' \
                'с работами на github, работал со всем необходимыми библиотеками: pandas, numpy, seaborn, plotly, ' \
                'mathplotlib, sklearn, есть базовые работы с deep learning на MNIST, есть опыт в работе с ' \
                'библиотеками для NLP: TF-IDF, word2vec, BERT, SBERT. Имею огромное желание развиваться ' \
                'направлении DA и DS'
    tfidf, w2v, sbert = hh_api_get_vacancies(user_text=user_text, id=int(getenv('bot_admin')))
    tfidf = '\r\n'.join(tfidf)
    w2v = '\r\n'.join(w2v)
    sbert = '\r\n'.join(sbert)
    print(f"Результат подбора по версии TF-IDF:\r\n{tfidf}")
    print(f"Результат подбора по версии Word2Vec:\r\n{tfidf}")
    print(f"Результат подбора по версии BERT:\r\n{tfidf}")