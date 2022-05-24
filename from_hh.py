import hh_token
import pandas as pd
from os import path, getcwd
import sqlite3 as sql
import requests as rq



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
    vacancies = pd.DataFrame()
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
    vacancies = pd.read_sql('select * from vacancies', con=connect, index_col='id')
    connect.close()
    # todo сделать предоброботку текста вакансий
    # todo реализовать подбор с помощью TF-IDF
    # todo реализовать подбор с помощью word2vec
    # todo реализовать подбор с помощью S-BERT
    return vacancies['url'].head(10).to_list()