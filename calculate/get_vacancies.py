from calculate import hh_token
import pandas as pd
from os import path, getcwd, getenv, environ
import sqlite3 as sql
import requests as rq


def hh_api_get_vacancies(params: object = None, user_id: object = 0) -> object:
    # sqlite connect params
    if params is None:
        params = dict()
    db_path = path.abspath(getcwd()) + '/data/vacancies.db'
    connect = sql.connect(db_path)

    # default parametrs for search vacancies
    token = hh_token.get_token()
    link_api_hh = 'https://api.hh.ru/'
    area = 113 # РФ
    date_from = '1970-01-02'
    page = 0
    per_page = 100
    limit = 1000
    ls_columns = ['id', 'name', 'url', 'api_url']
    filter_vacs_dict = {'Authorization': token, 'per_page': per_page, 'page': page, 'date_from': date_from,
                        'order_by': 'publication_time', 'area': area}
    filter_vacs_dict.update(params)
    print(f'Параметры запроса к апи HH: {filter_vacs_dict}')
    count_vacs = 0

    # запускаем обработку вакансий
    if int(getenv('is_razmetka')):
        # для разметки данных берем данные локально из БД
        print('Находимся в режиме разметки, данные берём из БД')
        vacancies = pd.read_sql('select * from vacancies_'+str(user_id), con=connect, index_col='id')
        environ['model'] = '0'
        count_vacs = len(vacancies)
    else:
        print('Находимся в режиме работы с HH.ru')
        # а для режима пром - берем данные с HH.ru
        vacancies = pd.DataFrame()
        rs_get = rq.get(url=link_api_hh+'vacancies/', params=filter_vacs_dict).json()
        ls_vacancies = rs_get.get('items')
        count_vacs = rs_get.get('found')
        iters = [x for x in range(int(count_vacs/per_page)+1)][1:int(limit/per_page)]
        for iter_page in iters:
            filter_vacs_dict.update({'page': iter_page})
            ls_vacancies = ls_vacancies + rq.get(url=link_api_hh+'vacancies/',
                                                 params=filter_vacs_dict).json().get('items')
        for i in range(len(ls_vacancies)):
            if isinstance(ls_vacancies[i], list):
                continue
            new_line = pd.DataFrame(
                [[(ls_vacancies[i]).get('id'), (ls_vacancies[i]).get('name'),
                  (ls_vacancies[i]).get('alternate_url'), (ls_vacancies[i]).get('url')]],
                columns=ls_columns,
                index=[i])
            vacancies = pd.concat([vacancies, new_line], axis=0)
        vacancies['description'] = vacancies['api_url'].apply(lambda x: rq.get(x).json().get('description'))
        vacancies.to_sql('vacancies_'+str(user_id), con=connect, if_exists='replace', index=False)
    connect.close()
    return vacancies, count_vacs
