import sqlite3 as sql
from os import path, getcwd
import pandas as pd
import requests as rq
import hh_token

# обновление словарей городов и регионов
def get_regions(areas=dict(), reload=0):
    db_path = path.abspath(getcwd()) + '\\data\\vacancies.db'
    connect = sql.connect(db_path)
    if (not len(areas) == 0) and reload == 1:
        pd.DataFrame([x for x in areas])[['id', 'parent_id', 'name']].to_sql('region', con=connect, if_exists='replace'
                                                                             , index=False)
        ls_cities = []
        for region in areas:
            for city in region.get('areas'):
                ls_cities.append(city)
        pd.DataFrame(ls_cities)[['id', 'parent_id', 'name']].to_sql('cities', con=connect, if_exists='replace'
                                                                    , index=False)
    connect.close()

# обновление словарей параметров вакансий
def get_dictionaries(dictionaries=dict(), reload=0):
    db_path = path.abspath(getcwd()) + '\\data\\vacancies.db'
    connect = sql.connect(db_path)
    ls_dicts = ['employment', 'experience', 'schedule', 'vacancy_type', 'working_days', 'currency', 'employer_type']
    if (not len(dictionaries) == 0) and reload == 1:
        for name_dict in ls_dicts:
            pd.DataFrame(dictionaries.get(name_dict)).to_sql(name=name_dict, con=connect, if_exists='replace'
                                                             , index=False)
    connect.close()

# функция обновления словарей
def reload_dict(dict_name, reload=1):
    token = hh_token.get_token()
    headers = {'Authorization': token}
    api_hh = 'https://api.hh.ru/'
    db_path = path.abspath(getcwd()) + '\\data\\vacancies.db'
    connect = sql.connect(db_path)
    # условие обновления словарей: все или только конкретный, например справочник регионов
    if reload:
        if dict_name == 'areas':
            areas = rq.get(url=api_hh + dict_name, headers=headers).json()[0].get('areas')
            get_regions(areas, 1)
        elif dict_name == 'dictionaries':
            dictionaries = rq.get(url=api_hh + dict_name, headers=headers).json()
            get_dictionaries(dictionaries, 1)
        elif dict_name == 'all':
            areas = rq.get(url=api_hh + 'areas', headers=headers).json()[0].get('areas')
            get_regions(areas, 1)
            dictionaries = rq.get(url=api_hh + 'dictionaries', headers=headers).json()
            get_dictionaries(dictionaries, 1)
    connect.close()

