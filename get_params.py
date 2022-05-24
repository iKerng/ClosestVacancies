from os import path, getcwd, getenv
import sqlite3 as sql
from pandas import DataFrame

from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from from_hh import hh_api_get_vacancies as get_vacs
from preload import reload_dict, get_regions, reload_dict


def sql_def():
    db_path = path.abspath(getcwd()) + '\\data\\vacancies.db'
    connect = sql.connect(db_path)
    return connect.cursor()

# todo написать функцию админскую, по которой делать обновление словарей

class OrderParams(StatesGroup):
    waiting_start = State()
    waiting_schedule = State()
    waiting_city = State()
    waiting_region = State()
# todo: employer_type
# todo: experience
# todo: подумать, какие еще справочники использовать
    waiting_search_word_keys = State()
    waiting_description = State()


async def choose_schedule(msg: types.Message, state: FSMContext):
    print('choose_schedule')
    cur = sql_def()
    cols = DataFrame(cur.execute("SELECT name FROM pragma_table_info('schedule')"))[0].to_list()
    df_schedule = DataFrame(cur.execute('''SELECT * FROM SCHEDULE'''), columns=cols)
    cur.close()
    work_type_office = df_schedule['name'].to_list()
    work_type_office.remove('Удаленная работа')
    if msg.text == 'Удаленная работа':
        await state.update_data(schedule=df_schedule[df_schedule['name'] == msg.text]['id'].to_list()[0])
        await OrderParams.waiting_search_word_keys.set()
        await msg.answer('Поздравляю! Остался последний шаг. Пришлите сообщение с описанием вакансии, которую Вы ищите.'
                   , reply_markup=types.ReplyKeyboardRemove())
    elif work_type_office.count(msg.text):
        await state.update_data(schedule=df_schedule[df_schedule['name'] == msg.text]['id'].to_list()[0])
        await msg.reply('Отлично! Давайте теперь определим город в котором будет осуществляться подбор вакансии.\r\n' +
                        'Для этого отправьте ответным сообщение наименование города.'
                        , reply_markup=types.ReplyKeyboardRemove())
        await OrderParams.next()
    else:
        await msg.reply("Данного формата работы нет в списке параметров поиска. Произведите выбор заново.")



async def choose_city(msg: types.Message, state: FSMContext):
    print('choose_city')
    cur = sql_def()
    cols_cities = DataFrame(cur.execute("SELECT name FROM pragma_table_info('cities')"))[0].to_list()
    df_cities = DataFrame(cur.execute("SELECT DISTINCT parent_id, name FROM cities"), columns=cols_cities[1:])
    cols_region = DataFrame(cur.execute("SELECT name FROM pragma_table_info('region')"))[0].to_list()
    df_region = DataFrame(cur.execute("SELECT * FROM region c"), columns=cols_region)
    cur.close()
    city = df_cities['name'].str.lower().to_list().count(msg.text.lower())
    region = df_region['name'].str.lower().to_list().count(msg.text.lower())
    if city > 0 or region:
        await state.update_data(city=msg.text.lower())
        count_cities = df_cities['name'].to_list().count(msg.text.lower())
        if count_cities > 1:
            list_regions = df_region[df_region['id'].isin(df_cities[df_cities['name'].str.lower() == msg.text.lower()]\
                                                          ['parent_id'].to_list())]['name'].to_list()
            region_buttons = types.ReplyKeyboardMarkup(resize_keyboard=True)
            region_buttons.add(*list_regions)
            await msg.reply('Мы нашли данные город в нескольких регионах. Выберите Ваш регион'
                            , reply_markup=region_buttons)
            await state.update_data(city=msg.text)
            await OrderParams.next()
        else:
            if region:
                city_id = df_region[df_region['name'].str.lower() == msg.text.lower()]['id'].to_list()[0]
            else:
                city_id = df_cities[df_cities['name'].str.lower() == msg.text.lower()]['id'].to_list()[0]
            await msg.answer('Поздравляю! Остался последний шаг. Пришлите сообщение с описанием вакансии, '\
                             + 'которую Вы ищите.')
            await state.update_data(area=city_id)
            await OrderParams.waiting_search_word_keys.set()
    else:
        await msg.reply('Увы, но мне не удалось найти город с таким названием. Вероятно Вы допустили опечатку, '\
                        + 'попробуйте снова')



async def choose_region(msg: types.Message, state: FSMContext):
    print('choose_region')
    cur = sql_def()
    city = (await state.get_data()).get('city')
    df_city_region = DataFrame(cur.execute("""
        SELECT 
            c.name as city_name
            , c.id as city_id
            , c.parent_id as region_id
            , r.name as region_name 
        FROM 
            cities c 
            JOIN region r ON r.id = c.parent_id 
        WHERE 
            1=1
            AND c.name = '""" + city + """' 
            AND r.name = '""" + msg.text + """'
        """), columns=['city_name', 'city_id', 'region_id', 'region_name'])
    cur.close()
    quantity = len(df_city_region)
    if quantity:
        city_id = df_city_region['city_id'].to_list()[0]
        await state.update_data(region=msg.text.lower())
        await state.update_data(area=city_id)
        await OrderParams.next()
    else:
        await msg.reply('Вероятно Вы указали регион не из предложенных вариантов. Выберите регион нажав на кнопку')
    await msg.answer('Осталось немного. Напишите какую роль Вы хотите выполнять, например: '\
                     + '"тестироващик ПО" или "аналитик данных"'
               , reply_markup=types.ReplyKeyboardRemove())


async def search_vacs_word_keys(msg: types.Message, state: FSMContext):
    print('choose_vacs_word_keys')
    await state.update_data( )
    await msg.answer('Данные получены, производим обработку... Необходимо подождать некоторое время... '\
                     + 'Дождитесь сообщения о завершении.')
    await state.update_data(text='"' + msg.text + '"')
    print(await state.get_data())
    await msg.answer('Теперь опишите функицональные обязанности, которые Вы хотите выполнять.')
    await OrderParams.next()

async def analyze_description(msg: types.Message, state: FSMContext):
    print('analyze_description')
    # todo здесь будет ключевая логика по обработке текста с помощью NLP
    await msg.answer('Механизм подбора вакансий запущен. К сожалению, необходимо немного подождать, '\
                     + 'пока я подберу для Вас подходящие вакансии. Я обязательно Вам напишу, '\
                     + 'как только закончу подбор вакансий')
    ls_result = get_vacs(params=(await state.get_data()), user_text=msg.text, id=msg.id)
    await msg.reply('Поздравляю! Подбор вакансии по заданному описанию с применением одного из направлений машинного '\
                    + 'обучение, а именно NLP, завершен!')
    # todo в проме убрать slice из цикла
    for i, url in enumerate(ls_result[:2]):
        await msg.answer(str(i+1) + ') ' + url)
    await state.finish()
    await msg.answer('Для нового поиска нажмите /start')


# функция определения состояния
# async def check_state(state: FSMContext):
#     current_state = await state.get_state()  # текущее машинное состояние пользователя
#     if current_state in OrderParams:  # registration - название класса состояний
#         print('Пользователь в одном из состояний регистрации')
#     if current_state == 'OrderParams:waiting_start':
#         print('Пользователь находиться в состоянии - waiting_start')
#     if current_state == 'OrderParams:waiting_start':
#         print('Пользователь находиться в состоянии - waiting_schedule')
#     if current_state == 'OrderParams:waiting_city':
#         print('Пользователь находиться в состоянии - waiting_city')
#     if current_state == 'OrderParams:waiting_region':
#         print('Пользователь находиться в состоянии - waiting_region')


def register_continue(dp: Dispatcher):
    dp.register_message_handler(choose_schedule, state=OrderParams.waiting_schedule)
    dp.register_message_handler(choose_city, state=OrderParams.waiting_city)
    dp.register_message_handler(choose_region, state=OrderParams.waiting_region)
    dp.register_message_handler(analyze_description, state=OrderParams.waiting_description)
    dp.register_message_handler(search_vacs_word_keys, state=OrderParams.waiting_search_word_keys)

