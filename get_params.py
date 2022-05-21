from os import path, getcwd, getenv
import sqlite3 as sql
from pandas import DataFrame

from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage

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
    waiting_description = State()
# todo: employer_type
# todo: experience
# todo: подумать, какие еще справочники использовать


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
        await OrderParams.waiting_description.set()
        await msg.answer('Поздравляю! Остался последний шаг. Пришлите сообщение с описанием вакансии, которую вы ищите.'
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
    if df_cities['name'].to_list().count(msg.text) > 0 or df_region['name'].to_list().count(msg.text) > 0:
        count_cities = df_cities['name'].to_list().count(msg.text)
        await state.update_data(city=msg.text)
        if count_cities > 1:
            list_regions = df_region[df_region['id'].isin(df_cities[df_cities['name'] == msg.text]['parent_id'].to_list())]['name'].to_list()
            region_buttons = types.ReplyKeyboardMarkup(resize_keyboard=True)
            region_buttons.add(*list_regions)
            await msg.reply('Мы нашли данные город в нескольких регионах. Выберите Ваш регион', reply_markup=region_buttons)
            await OrderParams.next()
        else:
            await OrderParams.waiting_description.set()
            await msg.answer('Поздравляю! Остался последний шаг. Пришлите сообщение с описанием вакансии, которую вы ищите.')
    else:
        await msg.reply('Увы, но мне не удалось найти город с таким названием. Вероятно Вы допустили опечатку, попробуйте снова')



async def choose_region(msg: types.Message, state: FSMContext):
    print('choose_region')
    cur = sql_def()
    city = await state.get_data()
    quantity = cur.execute("SELECT count(*) AS quantity FROM cities c JOIN region r ON r.id = c.parent_id WHERE c.name = '" + city.get('city')
                + "' AND r.name = '" + msg.text +"'")
    cur.close()
    if quantity:
        await state.update_data(region=msg.text)
        await state.next()
    else:
        await msg.reply('Вероятно Вы указали регион не из предложенных вариантов. Выберите регион нажав на кнопку')
    msg.answer('Поздравляю! Остался последний шаг. Пришлите сообщение с описанием вакансии, которую вы ищите.'
               , reply_markup=types.ReplyKeyboardRemove())



async def analyze_description(msg: types.Message, state: FSMContext):
    print('analyze_description')
    # todo здесь будет ключевая логика по обработке текста с помощью NLP
    await msg.reply('Поздравляю! Подбор вакансии по заднному описанию с применение одного из направлений машинного обучение, а именно NLP, завершен!')
    ls_vacancies = ['https://hh.ru/vacancy/49423067']
    for i in ls_vacancies:
        await msg.answer(i)
    await state.finish()


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

