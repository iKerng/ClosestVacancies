from os import environ, getenv, path, getcwd

from aiogram import Dispatcher, types
from aiogram.dispatcher.filters import IDFilter, Text, BoundFilter
from aiogram.utils.exceptions import MessageNotModified

from calculate.workflow import cmd_start, cmd_cancel, cmd_start_new, cmd_help
from calculate.workflow import choose_schedule, OrderParams, choose_city
from calculate.workflow import choose_region, analyze_description,  search_vacs_word_keys
from calculate.preload_dicts import reload_dict

import sqlite3 as sql

from calculate.log_writer import to_log

db_path = path.abspath(getcwd()) + '/data/vacancies.db'


# админская настройка для выбора используемой модели
async def switch_nlp_model(msg: types.Message):
    nlp_model = {'0': 'all', '1': 'TF-IDF', '2': 'Doc2Vec', '3': 'S-BERT'}
    ls_words = msg.text.lower().split(' ')
    if ls_words.count('all'):
        await msg.reply(f"Используемые модели: {str(list(nlp_model.values()[1:]))[1:-1]}")
        environ['model'] = '0'
        to_log(log_user=msg.from_user.username,
               log_user_id=msg.from_user.id,
               log_text='Включаем все виды используемых моделей NLP')
    elif ls_words.count('tf-idf'):
        await msg.reply('Используется модель: TF-IDF')
        environ['model'] = '1'
        to_log(log_user=msg.from_user.username,
               log_user_id=msg.from_user.id,
               log_text='Изменена модель NLP на TF-IDF')
    elif ls_words.count('word2vec'):
        await msg.reply('Используется модель: Doc2Vec')
        environ['model'] = '2'
        to_log(log_user=msg.from_user.username,
               log_user_id=msg.from_user.id,
               log_text='Изменена модель NLP на Doc2Vec')
    elif ls_words.count('s-bert'):
        await msg.reply('Используется модель: S-BERT')
        environ['model'] = '3'
        to_log(log_text='Изменена модель NLP на Sentence-BERT')
    elif msg.text.lower() == '/model':
        to_log(log_user=msg.from_user.username,
               log_user_id=msg.from_user.id,
               log_text='Указана модель не из списка возможных (1: TF-IDF, 2: Doc2Vec, 3: S-BERT)')
        await msg.answer(f"Используется модель: { nlp_model.get(getenv('model')) }")
        model_button = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        model_button.add(*['/model TF-IDF', '/model Doc2Vec', '/model S-BERT', '/model all'])
        await msg.reply(f'Выбери режим разметки:', reply_markup=model_button)
    else:
        await msg.reply('Данная модель мне неизвестна.')


# функция разблокирования пользователя
async def unblock_user(msg: types.Message):
    user_id = msg.text.split()[1]
    connect = sql.connect(db_path)
    connect.execute(f"UPDATE access set access = 1 where  user_id = {user_id}")
    connect.commit()
    connect.close()
    to_log(log_user=msg.from_user.username, log_user_id=msg.from_user.id,
           log_text=f'Пользователь с ID [{user_id}] разблокирован')
    await msg.bot.send_message(int(user_id), 'Поздравляю, доступ к услугам бота предоставлен!\r\nДля запуска сервиса '
                                             'воспользуйтесь командой /start')

# админская команда на обновление словарей
async def reload_dicts(msg: types.Message):
    reload_dict('all')
    to_log(log_user=msg.from_user.username, log_user_id=msg.from_user.id,
           log_text=f'Обновили все словари')
    await msg.reply('Выбранные словари обновлены.')


# админская команда для переключения в режим разметки
async def switch_mode_razmetki(msg: types.Message):
    # расшифровываем код переменной из виртуального окружения
    status_razmetki = 'ON' if int(getenv('is_razmetka')) else 'OFF'
    # парсим сообщения, чтобы понять, передается ли внутри устанавливаемое значение
    # или просто хотим узнать текущее значение
    list_words = msg.text.lower().split(' ')
    if list_words.count('on'):
        await msg.reply(f'Режим разметки установлен в состояние: ON')
        environ['is_razmetka'] = '1'
        to_log(log_user=msg.from_user.username, log_user_id=msg.from_user.id,
               log_text=f'Режим разметки данных включен')
    elif list_words.count('off'):
        await msg.reply(f'Режим разметки установлен в состояние: OFF')
        environ['is_razmetka'] = '0'
        to_log(log_user=msg.from_user.username, log_user_id=msg.from_user.id,
               log_text=f'Режим разметки данных выключен')
    elif msg.text == '/razmetka':
        await msg.answer(f"Режим разметки: {status_razmetki}")
        razmetka_button = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        razmetka_button.add(*['/razmetka OFF', '/razmetka ON'])
        await msg.reply(f'Выбери режим разметки:', reply_markup=razmetka_button)
        to_log(log_user=msg.from_user.username, log_user_id=msg.from_user.id,
               log_text=f"Запрос на получение текущего значения режима разметки. Текущее значение " +
                        f"{getenv('is_razmetka')}")
    else:
        await msg.reply('Данная команда мне неизвестна.')
    # to_log(msg.ge)


class IsAccess(BoundFilter):
    key = 'in_whitelist'

    def __init__(self, in_whitelist):
        self.in_whitelist = in_whitelist

    async def check(self, msg: types.Message) -> bool:
        new_user_id = msg.from_user.id
        connect = sql.connect(db_path)
        if connect.execute(f"SELECT count(*) FROM access WHERE user_id = '{new_user_id}' and access = 1")\
                .fetchall()[0][0]:
            connect.close()
            return [new_user_id]
        # elif connect.execute(f"SELECT count(*) FROM access WHERE user_id = '{new_user_id}' and access = 0")\
        #         .fetchall()[0][0]:
        #     to_log(log_user=msg.from_user.username,
        #            log_user_id=msg.from_user.id,
        #            log_text="Заблокированный пользователь пытается воспользоваться сервисом")
        else:
            connect.close()


def register_handlers_common(dp: Dispatcher, user_id):
    dp.register_message_handler(cmd_start, IsAccess(in_whitelist=user_id), commands="start", state="*")
    dp.register_message_handler(cmd_start_new, commands="start", state="*")

    @dp.callback_query_handler(text="add_user")
    async def add_user(call: types.CallbackQuery):
        new_user_id = call.message.reply_markup.inline_keyboard[0][0].text.split(' ')[-1]
        connect = sql.connect(db_path)
        if not connect.execute(f"SELECT count(*) FROM access WHERE user_id = '{new_user_id}'").fetchall()[0][0]:
            connect.execute(f"INSERT INTO access (user_id, access) VALUES ({new_user_id}, 1)")
            connect.commit()
            connect.close()
        else:
            connect.execute(f"UPDATE access set access = 1 where user_id = {new_user_id}")
            connect.commit()
            connect.close()
        await call.message.answer(f'Пользователю с ID [{new_user_id}] предоставлен доступ к сервису')
        await call.bot.send_message(int(new_user_id), text='Вам предоставлен доступ к сервису. Для запуска '
                                                           'необходимо снова произвести запуск с помощью '
                                                           'команды /start')
        to_log(log_text=f"Пользователю с ID [{new_user_id}] предоставлен доступ к сервису")


    @dp.callback_query_handler(text="block_user")
    async def block_user(call: types.CallbackQuery):
        new_user_id = call.message.reply_markup.inline_keyboard[0][0].text.split(' ')[-1]
        to_log(log_text=f"Пользователю с ID [{new_user_id}] заблокирован доступ к сервису")
        connect = sql.connect(db_path)
        if connect.execute(f"SELECT count(*) FROM access WHERE user_id = '{new_user_id}'").fetchall()[0][0]:
            connect.execute(f"UPDATE access set access = 0 where  user_id = {new_user_id}")
            connect.commit()
            connect.close()
        else:
            connect.execute(f"INSERT INTO access (user_id) VALUES ({new_user_id})")
            connect.commit()
            connect.close()
        await call.message.answer(f'Пользователю с ID [{new_user_id}] заблокирован доступ к сервису')
        # нужно решить, отдавать ли обратную связь по блокировке, так как могут обидеться и ддосить...
        # но заготовка пусть будет
        # await call.bot.send_message(int(new_user_id), 'К сожалению, доступ к сервису заблокирован')

    @dp.callback_query_handler(text="role_desc")
    async def role_desc(call: types.CallbackQuery):
        change_text = 'На данный момент в фильтре "наименование вакансии" реализован поиск через:\r\n' \
                      '1) операцию "или" между словами, например, "python разработчик": будут отбираться вакансии,' \
                      'у которых в заголовке вакансии встречается слово "python" или слово "разработчик"\r\n' \
                      '2) операцию "или" между словосочетаниями, например если ввести такое сообщение: "data analytic" ' \
                      'или "аналитик данных" или "data analyst", то будут отбираться вакансии, у которых в заголовке ' \
                      'будут встречаться одно из перечисленных в кавычках словосочетаний.'
        desc_role = 'Описание работы фильтра "Наименование роли"'
        desc_func = 'Рекомендации по вводу текста функциональных обязанностей'
        desc_main = 'Основное описание'
        inl_kb = [types.InlineKeyboardButton(desc_role, callback_data='role_desc'),
                  types.InlineKeyboardButton(desc_func, callback_data='func_desc'),
                  types.InlineKeyboardButton(desc_main, callback_data='return_desc')]
        keyboard = types.InlineKeyboardMarkup(row_width=1)
        keyboard.add(*inl_kb)
        try:
            await call.message.edit_text(change_text, reply_markup=keyboard)
        except MessageNotModified:
            return None

    @dp.callback_query_handler(text="func_desc")
    async def func_desc(call: types.CallbackQuery):
        change_text = 'В единственном обязательном пункте описания функциональных обязанностей ' \
                      'необходимо описывать то, что Вы ищите, то, чем Вы  хотите' \
                      'заниматься, какие функциональные обязанности Вы хотите выполнять, а так же те навыки, которымы' \
                      ' Вы уже овладели. Текст должен быть не коротким из серии "хочу быть python разработчиком" (для' \
                      'примера), а что-то вроде "ищу работу на должность full stack Python разработчика, прошел ' \
                      'обучение в такой-то школе, работал с такими-то библиотеками, реализовал следующие проекты' \
                      'умею работать с git, linux, windows, знаю SQL на таком-то уровне". В общем все то, что Вы ' \
                      'хотите и умеете применять.\r\n' \
                      'Альтернативный способ использования сервиса - это поиск вакансий релевантных Вашему запросу с ' \
                      'целью определения какие навыки требуются для соответствующей роли, чтобы подтянуть свои ' \
                      'навыки.\r\n' \
                      'Или же для какой-либо другой цели.'
        desc_role = 'Описание работы фильтра "Наименование роли"'
        desc_func = 'Рекомендации по вводу текста функциональных обязанностей'
        desc_main = 'Основное описание'
        inl_kb = [types.InlineKeyboardButton(desc_role, callback_data='role_desc'),
                  types.InlineKeyboardButton(desc_func, callback_data='func_desc'),
                  types.InlineKeyboardButton(desc_main, callback_data='return_desc')]
        keyboard = types.InlineKeyboardMarkup(row_width=1)
        keyboard.add(*inl_kb)
        try:
            await call.message.edit_text(change_text, reply_markup=keyboard)
        except MessageNotModified:
            return None

    @dp.callback_query_handler(text="return_desc")
    async def return_desc(call: types.CallbackQuery):
        change_text = 'Данный бот производит поиск наиболее подходящих вакансий на основе поданного текста на этапе' \
                      'описания функциональных обязанностей. Бот использует методы машинного обучения по ' \
                      'сопоставлению двух текстов (текста от пользователя и текста описания вакансий, выгруженных с ' \
                      'сайта HH.ru на основании заполненных фильтров (все фильтры необязательны и их указание можно ' \
                      'пропустить тем не менее рекомендую их заполнять, так как будет более релевантный результат)).' \
                      '\r\n' \
                      '1) Сначала нужно указать тип режима работы (если режим удаленный, то следующая стадия ' \
                      'будет 3)\r\n' \
                      '2) Необходимо указаать город\r\n' \
                      '2.1) Если существует несколько городов в разных областях с одинаковым городом, ' \
                      'то Вам будет предложен выбор региона, в котором был найден город\r\n' \
                      '3) Необходимо указать "Наименование роли" (по данному тексту будет производиться отбор ' \
                      'вакансий). Более подробное описание доступно по кнопке\r\n' \
                      '4) Необходимо описать функциональные обязанности, которые Вы хотите выполнять (более подробное' \
                      ' описание доступно по кнопке).\r\n\r\n' \
                      'Так же у бота реализованы команды /cancel или просто написать "отмена" или "отменить" в чат ' \
                      'для отмены текущего поиска.'
        desc_role = 'Описание работы фильтра "Наименование роли"'
        desc_func = 'Рекомендации по вводу текста функциональных обязанностей'
        desc_main = 'Основное описание'
        inl_kb = [types.InlineKeyboardButton(desc_role, callback_data='role_desc'),
                  types.InlineKeyboardButton(desc_func, callback_data='func_desc'),
                  types.InlineKeyboardButton(desc_main, callback_data='return_desc')]
        keyboard = types.InlineKeyboardMarkup(row_width=1)
        keyboard.add(*inl_kb)
        try:
            await call.message.edit_text(change_text, reply_markup=keyboard)
        except MessageNotModified:
            return None

    dp.register_message_handler(cmd_start, IsAccess(in_whitelist=user_id),
                                Text(equals=['старт', 'начать', 'запуск', 'запустить'], ignore_case=True), state="*")
    dp.register_message_handler(cmd_cancel, IsAccess(in_whitelist=user_id), commands="cancel", state="*")
    dp.register_message_handler(cmd_cancel, IsAccess(in_whitelist=user_id),
                                Text(equals=['отмена', 'отменить'], ignore_case=True), state="*")
    dp.register_message_handler(cmd_help, IsAccess(in_whitelist=user_id), commands='help', state="*")
    dp.register_message_handler(cmd_help, IsAccess(in_whitelist=user_id),
                                Text(equals=['помоги', 'помощь', 'help']), state="*")


def register_handlers_admin(dp: Dispatcher, user_id: int):
    dp.register_message_handler(switch_mode_razmetki, IDFilter(user_id=user_id), commands="razmetka", state='*')
    dp.register_message_handler(switch_nlp_model, IDFilter(user_id=user_id), commands="model", state='*')
    dp.register_message_handler(reload_dicts, IDFilter(user_id=user_id), commands="reload_dicts", state='*')
    dp.register_message_handler(unblock_user, IDFilter(user_id=user_id), commands="unblock", state='*')


def register_continue(dp: Dispatcher, user_id: int):
    # функция перехода по состояниям
    dp.register_message_handler(choose_schedule, IsAccess(in_whitelist=user_id), state=OrderParams.waiting_schedule)
    dp.register_message_handler(choose_city, IsAccess(in_whitelist=user_id), state=OrderParams.waiting_city)
    dp.register_message_handler(choose_region, IsAccess(in_whitelist=user_id), state=OrderParams.waiting_region)
    dp.register_message_handler(analyze_description, IsAccess(in_whitelist=user_id),
                                state=OrderParams.waiting_description)
    dp.register_message_handler(search_vacs_word_keys, IsAccess(in_whitelist=user_id),
                                state=OrderParams.waiting_search_word_keys)
