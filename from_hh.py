import hh_token
import pandas as pd
from os import path, getcwd
import sqlite3 as sql
import requests as rq
import bs4 as bs

from nltk.tokenize import WordPunctTokenizer
from nltk.stem import snowball
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_tokenize(text=''):
    bs.BeautifulSoup(text.lower())
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


def hh_api_get_vacancies(params=dict(), user_text='', id=0):
    # sqlite connect params
    db_path = path.abspath(getcwd()) + '\\data\\vacancies.db'
    connect = sql.connect(db_path)
    cursor = connect.cursor()


    # default parametrs for search vacancies
    token = hh_token.get_token()
    link_api_hh = 'https://api.hh.ru/'
    date_from = '1970-01-02'
    page = 0
    # todo в проме - 100
    per_page = 100
    # todo в проме - 1000
    limit = 1000
    ls_columns = ['id', 'name', 'url', 'api_url']
    filter_vacs_dict = {'Authorization':token, 'per_page':per_page, 'page':page, 'date_from':date_from
        , 'order_by':'publication_time', }
    filter_vacs_dict.update(params)

    # для продакшена
    # vacancies = pd.DataFrame()
    # rs_get = rq.get(url=link_api_hh+'vacancies/', params=filter_vacs_dict).json()
    # ls_vacancies = rs_get.get('items')
    # count_vacs = rs_get.get('found')
    # iters = [x for x in range(int(count_vacs/per_page)+1)][1:int(limit/per_page)]
    # for iter_page in iters:
    #     filter_vacs_dict.update({'page': iter_page})
    #     ls_vacancies = ls_vacancies + rq.get(url=link_api_hh+'vacancies/', params=filter_vacs_dict).json().get('items')
    # # with open(path.abspath(getcwd()) + '\\data\\ls_vacancies.txt', 'w', encoding='utf8') as f:
    # #     f.write(str(ls_vacancies))
    # #     f.close()
    # for i in range(len(ls_vacancies)):
    #     if type(ls_vacancies[i]) == type(list()):
    #         print(i)
    #         continue
    #     new_line = pd.DataFrame([[(
    #                                   ls_vacancies[i]).get('id')
    #                                  , (ls_vacancies[i]).get('name')
    #                                  , (ls_vacancies[i]).get('alternate_url')
    #                                  , (ls_vacancies[i]).get('url')]]
    #                             , columns=ls_columns
    #                             , index=[i])
    #     vacancies = pd.concat([vacancies, new_line], axis=0)
    # vacancies['description'] = vacancies['api_url'].apply(lambda x: rq.get(x).json().get('description'))
    # vacancies.to_sql('vacancies_'+str(id), con=connect, if_exists='replace', index=False)
    # vacancies = pd.read_sql('select * from vacancies_'+str(id), con=connect, index_col='id')
    # для продакшена


    # для разработки
    vacancies = pd.read_sql('select * from vacancies', con=connect, index_col='id')
    connect.close()
    # todo сделать предоброботку текста вакансий DONE
    vacancies['cleared_vacs'] = vacancies['description'].apply(lambda x: clean_tokenize(x))
    cleared_user_text = clean_tokenize(user_text)
    # todo реализовать подбор с помощью TF-IDF
    tfidf_vec = TfidfVectorizer(ngram_range=(2, 2), analyzer='word')
    tfidf_vec.fit([' '.join(vacancies['cleared_vacs'].to_list()) + ' ' + ' '.join(cleared_user_text)])
    user_vec = tfidf_vec.transform(cleared_user_text)
    vacs_vec = tfidf_vec.transform(vacancies.cleared_vacs.to_list())
    result_pair = cosine_similarity(user_vec, vacs_vec)[0]
    tf_idf_top10 = pd.Series(result_pair).sort_values(ascending=False).index.to_list()[0:10]
    tf_idf_top10_urls = vacancies['url'].iloc[tf_idf_top10].to_list()
    # todo реализовать подбор с помощью word2vec
    # todo реализовать подбор с помощью S-BERT
    # /для разработки
    return tf_idf_top10_urls