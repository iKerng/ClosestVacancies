from os import path, getcwd, getenv
import sqlite3 as sql
import random

from pandas import DataFrame
import nltk

from aiogram import types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup

from calculate.get_vacancies import hh_api_get_vacancies as get_vacs
from calculate.run_nlp_predict import nlp_predict
from calculate.log_writer import to_log

nltk.download('punkt')
db_path = path.abspath(getcwd()) + '/data/vacancies.db'

# todo добавить логирование

def sql_def():
    db_path = path.abspath(getcwd()) + '/data/vacancies.db'
    connect = sql.connect(db_path)
    return connect.cursor()


class OrderParams(StatesGroup):
    waiting_start = State()
    waiting_schedule = State()
    waiting_city = State()
    waiting_region = State()
    # todo: добавить еще справочники
    waiting_search_word_keys = State()
    waiting_description = State()


async def cmd_start_new(msg: types.Message):
    user_id = str(msg.from_user.id)
    connect =  sql.connect(db_path)
    # проверяем наличие пользователя в БД.
    # если пользователь имеется, то забираем параметр доступа, а также данные по принадлежности к админам
    if connect.execute(f"select count(*) from access where user_id = {user_id}").fetchall()[0][0]:
        access = connect.execute(f"select access from access where user_id = {user_id}").fetchall()[0][0]
        is_admin = connect.execute(f"select is_admin from access where user_id = {user_id}").fetchall()[0][0]
        connect.close()
        if not access:
            await msg.answer('Вы заблокированы админом. Доступ к сервису запрещен.')
            to_log(log_user=msg.from_user.username, log_user_id=user_id,
                   log_text=f"Заблокированный пользователь пытается получить доступ к сервису")
    # если пользователя нет
    else:
        to_log(log_user=msg.from_user.username,
               log_user_id=msg.from_user.id,
               log_text=f"Новый Пользователь отправил команду /start")
        to_log(log_user=msg.from_user.username, log_user_id=user_id,
               log_text=f"Пользователь отправил запрос на получение доступа к сервису")
        new_user_id = msg.from_user.id
        new_user_name = msg.from_user.full_name
        # отправляем пользователю сообщение о том, что заявка на доступ отправлена на рассмотрение
        await msg.answer('Информация о Вас поступила администратору. Пожалуйста, '
                         'ожидайте получения доступа к сервису бота')
        # создаем две кнопки: разрешить доступ, или запретить
        add_whitelist = f'Добавить в WhiteList пользователя {msg.from_user.first_name} с ID {str(new_user_id)}'
        add_blacklist = f'Заблокировать пользователя {msg.from_user.first_name} с ID [{str(new_user_id)}]'
        inl_kb = [types.InlineKeyboardButton(add_whitelist, callback_data='add_user')
            , types.InlineKeyboardButton(add_blacklist, callback_data='block_user')]
        keyboard = types.InlineKeyboardMarkup(row_width=1)
        keyboard.add(*inl_kb)
        admin_id = int(getenv('bot_admin'))
        # отправляем заявку на предоставление доступа главному админу
        await msg.bot.send_message(chat_id=admin_id,
                                   text=f'Пользователь с именем [{new_user_name}] и ID: [{new_user_id}] хочет '
                                        f'воспльзоваться услугой бота', reply_markup=keyboard)


async def cmd_start(msg: types.Message, state: FSMContext):
    to_log(log_user=msg.from_user.username,
           log_user_id=msg.from_user.id,
           log_text=f"Пользователь отправил команду /start")
    # Если приходит новая команда /start, то мы прекращаем все предыдущие состояния
    await state.finish()
    await OrderParams.waiting_schedule.set()
    cur = sql_def()
    # запрос к БД на получение списка колонок
    cols = DataFrame(cur.execute('''pragma table_info ('schedule')'''))[1].to_list()
    # запрос к БД на получение значений словаря режима работы
    df_schedule = DataFrame(cur.execute('''SELECT * FROM SCHEDULE'''), columns=cols)
    cur.close()
    work_type_office = df_schedule['name'].to_list() + ['Пропустить']
    work_type_button = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    work_type_button.add(*work_type_office)
    await msg.reply(
        f"{msg.from_user.first_name}, Вас приветствует бот по подбору вакансий с использованием "
        f"машинного обучения на базе Natural Language Processing\r\n\r\nДля более точного подбора вакансии настоятельно"
        f" рекомендую воспользоваться командой /help\r\n\r\n"
        f"Для поиска вакансий необходимо предварительно получить некоторые данные.\r\n\r\n"
        f"Выберите формат работы", reply_markup=work_type_button)


async def cmd_cancel(msg: types.Message, state: FSMContext):
    # функция отмены текущего процесса
    to_log(log_user=msg.from_user.username,
           log_user_id=msg.from_user.id,
           log_text=f"Пользователь отправил команду /cancel")
    await state.finish()
    await msg.answer("Действие отменено", reply_markup=types.ReplyKeyboardRemove())


async def cmd_help(msg: types.Message):
    to_log(log_user=msg.from_user.username,
           log_user_id=msg.from_user.id,
           log_text=f"Пользователь отправил команду /help")
    desc_role = 'Описание работы фильтра "Наименование роли"'
    desc_func = 'Рекомендация по вводу текста функциональных обязанностей'
    desc_main = 'Основное описание помощника'
    send_text = 'Данный бот производит поиск вакансий наиболее схожих с описанием, предоставленным пользователем на ' \
                'этапе описания функциональных обязанностей. Бот использует методы машинного обучения по сопоставлению ' \
                'двух текстов (текста от пользователя и текста описания вакансий, выгруженных с сайта HH.ru на ' \
                'основании заполненных фильтров (все фильтры необязательны и их указание можно пропустить тем не ' \
                'менее рекомендую их заполнять, так как будет более релевантный результат)).\r\n' \
                '1) Сначала нужно указать тип режима работы (если режим удаленный, то следующая стадия ' \
                'будет 3)\r\n' \
                '2) Необходимо указаать город\r\n' \
                '2.1) Если существует несколько городов в разных областях с одинаковым городом, ' \
                'то Вам будет предложен выбор региона, в котором был найден город'  \
                '3) Необходимо указать "Наименование роли" (по данному тексту будет производиться отбор ' \
                'вакансий). Более подробное описание доступно по кнопке\r\n' \
                '4) Необходимо описать функциональные обязанности, которые Вы хотите выполнять (более подробное' \
                ' описание доступно по кнопке).\r\n\r\n' \
                'Так же у бота реализованы команды /cancel или просто написать "отмена" или "отменить" в чат для ' \
                'отмены текущего поиска.\r\n\r\n' \
                'P.S.\r\nКнопки не будут работать, если Вы уже запустили процесс подбора вакансий'
    inl_kb = [types.InlineKeyboardButton(desc_role, callback_data='role_desc'),
              types.InlineKeyboardButton(desc_func, callback_data='func_desc'),
              types.InlineKeyboardButton(desc_main, callback_data='return_desc')]
    keyboard = types.InlineKeyboardMarkup(row_width=1)
    keyboard.add(*inl_kb)
    await msg.answer(text=send_text,
                     reply_markup=keyboard)


# функция выбора режим работы.
# Если удаленный режим, то переходим дальше к получению
async def choose_schedule(msg: types.Message, state: FSMContext):
    to_log(log_user=msg.from_user.username,
           log_user_id=msg.from_user.id,
           log_text=f"Пользователь выбирал формат работы: {msg.text}")
    cur = sql_def()
    # получаем список колонок
    cols = DataFrame(cur.execute("pragma table_info('schedule')"))[1].to_list()
    # получаем DF со словарем режима работы
    df_schedule = DataFrame(cur.execute('''SELECT * FROM SCHEDULE'''), columns=cols)
    cur.close()
    work_type_office = df_schedule['name'].to_list()
    work_type_office.remove('Удаленная работа')
    # если режим работы удаленный - то все равно из какого города работать
    skip_step = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    skip_step.add('Пропустить')
    if msg.text == 'Удаленная работа':
        await state.update_data(schedule=df_schedule[df_schedule['name'] == msg.text]['id'].to_list()[0])
        # следовательно переходим сразу к получению названия вакансии
        await OrderParams.waiting_search_word_keys.set()
        await msg.answer('Напишите, какую роль Вы хотите выполнять, например: '
                         '"тестироващик ПО" или "аналитик данных", более подробно ищите в /help (не работает во '
                         'время запущенного процесса подбора вакансий)',
                         reply_markup=types.ReplyKeyboardRemove())
    elif work_type_office.count(msg.text):
        await state.update_data(schedule=df_schedule[df_schedule['name'] == msg.text]['id'].to_list()[0])
        await msg.reply('Давайте теперь определим город в котором будет осуществляться подбор вакансий.\r\n'
                        'Для этого отправьте ответным сообщение наименование города.',
                        reply_markup=skip_step)
        await OrderParams.next()
    elif msg.text.lower() == 'пропустить':
        await msg.reply('Давайте теперь определим город в котором будет осуществляться подбор вакансий.\r\n'
                        'Для этого отправьте ответным сообщение наименование города.',
                        reply_markup=skip_step)
        await OrderParams.next()
    else:
        await msg.reply("Данного формата работы нет в списке параметров поиска. Произведите выбор заново.")


async def choose_city(msg: types.Message, state: FSMContext):
    to_log(log_user=msg.from_user.username,
           log_user_id=msg.from_user.id,
           log_text=f"Пользователь выбирал город поиска вакансий: {msg.text}")
    # формируем словари регионов и городов
    cur = sql_def()
    cols_cities = DataFrame(cur.execute("pragma table_info('cities')"))[1].to_list()
    df_cities = DataFrame(cur.execute("SELECT DISTINCT parent_id, name FROM cities"), columns=cols_cities[1:])
    cols_region = DataFrame(cur.execute("pragma table_info('region')"))[1].to_list()
    df_region = DataFrame(cur.execute("SELECT * FROM region c"), columns=cols_region)
    city = df_cities['name'].str.lower().to_list().count(msg.text.lower())
    region = df_region['name'].str.lower().to_list().count(msg.text.lower())
    skip_step = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    skip_step.add('Пропустить')
    # если название города найдено в словаре
    if city > 0 or region:
        await state.update_data(city=msg.text.lower())
        count_cities = df_cities['name'].to_list().count(msg.text.lower())
        # бывают случаи, когда существует несколько городов с одинаковым называнием,
        # поэтому мы переходим к шагу выбора региона
        if count_cities > 1 or msg.text.lower() == 'пропустить':
            ls_regions_id = df_cities[df_cities['name'].str.lower() == msg.text.lower()]['parent_id'].to_list()
            list_regions = df_region[df_region['id'].isin(ls_regions_id)]['name'].to_list()
            region_buttons = types.ReplyKeyboardMarkup(resize_keyboard=True)
            region_buttons.add(*list_regions)
            await msg.reply('Мы нашли данный населенный пункт в нескольких регионах. Выберите Ваш регион',
                            reply_markup=region_buttons)
            await state.update_data(city=msg.text)
            await OrderParams.next()
        # найдено одно совпадение с названием города
        else:
            if region:
                city_id = df_region[df_region['name'].str.lower() == msg.text.lower()]['id'].to_list()[0]
            else:
                df_cities = DataFrame(cur.execute("SELECT * FROM cities"), columns=cols_cities)
                city_id = df_cities[df_cities['name'].str.lower() == msg.text.lower()]['id'].to_list()[0]
            await msg.answer('Напишите какую роль Вы хотите выполнять, например: '
                             '"тестироващик ПО" или "аналитик данных", более подробно ищите в /help (не работает во '
                             'время запущенного процесса подбора вакансий',
                             reply_markup=skip_step)
            await state.update_data(area=city_id)
            await OrderParams.waiting_search_word_keys.set()
    # если совпадений не найдено, то мы продолжаем ожидать от пользователя название города из справочника
    else:
        if msg.text.lower() == 'пропустить':
            await msg.answer('Осталось немного. Напишите, какую роль Вы хотите выполнять, например: '
                             '"тестироващик ПО" или "аналитик данных", более подробно ищите в /help (не работает во '
                             'время запущенного процесса подбора вакансий)',
                             reply_markup=skip_step)
            await OrderParams.waiting_search_word_keys.set()
        else:
            await msg.reply('Увы, но мне не удалось найти город с таким названием. Вероятно Вы допустили опечатку, '
                            'попробуйте ввести название населенного пункта снова')
    cur.close()


async def choose_region(msg: types.Message, state: FSMContext):
    to_log(log_user=msg.from_user.username,
           log_user_id=msg.from_user.id,
           log_text=f"Пользователь выбирал регион поиска вакансий: {msg.text}")
    # так как мы получили название города, и знаем что их больше одного,
    # то мы формируем список кнопок для выбора региона
    # для этого мы создаем запрос на получение списка регионов по названию города
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
        # обновляем справочник состояний
        await state.update_data(region=msg.text.lower())
        await state.update_data(area=city_id)
        await OrderParams.next()
    else:
        await msg.reply('Вероятно Вы указали регион не из списка предложенных вариантов. Выберите регион нажав на '
                        'кнопку ниже')
    await msg.answer('Напишите, какую роль Вы хотите выполнять, например: '
                     '"тестироващик ПО" или "аналитик данных", более подробно ищите в /help (не работает во '
                     'время заупщенного процесса подбора вакансий)',
                     reply_markup=types.ReplyKeyboardRemove())


async def search_vacs_word_keys(msg: types.Message, state: FSMContext):
    to_log(log_user=msg.from_user.username,
           log_user_id=msg.from_user.id,
           log_text=f"Пользователь выбирал ключевые слова поиска вакансий: {msg.text}")
    await state.update_data()
    if msg.text.lower() != 'пропустить':
        if (msg.text.lower()).find('или') != -1:
            res = ' OR '.join([phrase for phrase in (msg.text.lower()).split('или')])
        else:
            res = ' OR '.join(word for word in (msg.text.lower()).replace('"', '').split(' '))
        await state.update_data(text=''.join(res))
    await msg.answer('Теперь опишите функциональные обязанности, которые Вы хотите выполнять, более подробную '
                     'информацию по заполнению данного поля ищите в /help (не работает во '
                     'время заупщенного процесса подбора вакансий)',
                     reply_markup=types.ReplyKeyboardRemove())
    await OrderParams.next()


async def analyze_description(msg: types.Message, state: FSMContext):
    to_log(log_user=msg.from_user.username,
           log_user_id=msg.from_user.id,
           log_text=f"Пользователь ввел описание искомой вакансии: {msg.text}")
    await msg.answer('Механизм подбора вакансий запущен. К сожалению, необходимо немного подождать, '
                     'пока я подберу для Вас подходящие вакансии. Я обязательно Вам напишу, '
                     'как только закончу подбор вакансий. Ориентировочное время подбора вакансий: 5-10 минут')
    df_vacs, quantity = get_vacs(params=(await state.get_data()), user_id=msg.from_user.id)
    if quantity != 0:
        if len(df_vacs) >= 1000:
            await msg.answer(f'По названию запрашиваемых вакансий всего найдено {quantity} вакансий, но мы будем искать'
                             f' среди 1000 самых свежих опубликованных')
        else:
            await msg.answer(f'Всего найдено: {quantity} вакансий(я) по заправшиваемым параметрам.')
        if int(getenv('model')) == 0:
            ls_result = nlp_predict(user_text=msg.text, vacancies=df_vacs)
            if int(getenv('is_razmetka')):
                ls_result = list(set(ls_result))
                random.shuffle(ls_result, random=random.seed(42))
        else:
            ls_result = nlp_predict(user_text=msg.text, vacancies=df_vacs)
        await msg.reply('Поздравляю! Подбор вакансии по заданному описанию с применением одного из направлений '
                        'машинного обучения, а именно NLP, завершен!')
        for i, url in enumerate(ls_result):
            await msg.answer(str(i + 1) + ') ' + url)
        await state.finish()
        await msg.answer('Для нового поиска нажмите /start')
    else:
        name_vac = (await state.get_data()).get('text')
        await msg.answer(f"По вашему запросу названия [{name_vac}] вакансий не найдено")
        await state.finish()
        await msg.answer('Для нового поиска нажмите /start')
