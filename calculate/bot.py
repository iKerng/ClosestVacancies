from os import environ, getenv, path, getcwd

from aiogram import Dispatcher, types
from aiogram.dispatcher.filters import IDFilter, Text, BoundFilter
from aiogram.utils.exceptions import MessageNotModified

from calculate.workflow import cmd_start, cmd_cancel, cmd_start_new, cmd_help
from calculate.workflow import choose_schedule, OrderParams, choose_city
from calculate.workflow import choose_region, analyze_description,  search_vacs_word_keys
from calculate.preload_dicts import reload_dict


def add_to_list(id: int, list: str):
    cur_str_ids = getenv(list)
    if len(cur_str_ids) > 0:
        new_value = cur_str_ids + ',' + str(id)
    else:
        new_value = str(id)
    environ[list] = new_value

    path_file = path.abspath(getcwd()) + '/data/set_env_' + list + '.py'
    file = open(path_file, 'r', encoding='utf8')
    cur_text = file.read()
    file.close()
    file = open(path_file, 'w', encoding='utf8')
    new_text = cur_text.split(" '")[0] + " '" + new_value + "'"
    file.write(new_text)
    file.close()


def remove_from_list(id: int, list: str):
    cur_str_ids = getenv(list)
    if len(cur_str_ids) > 0:
        list_ids = cur_str_ids.split(',')
        if list_ids.count(str(id)): list_ids.remove(str(id))
        new_str_ids = ','.join(list_ids)
        environ[list] = new_str_ids

        path_file = path.abspath(getcwd()) + '/data/set_env_' + list + '.py'
        file = open(path_file, 'r', encoding='utf8')
        cur_text = file.read()
        file.close()
        file = open(path_file, 'w', encoding='utf8')
        new_text = cur_text.split(" '")[0] + " '" + new_str_ids + "'"
        file.write(new_text)
        file.close()


# админская настройка для выбора используемой модели
async def switch_nlp_model(msg: types.Message):
    nlp_model = {'0': 'all', '1': 'TF-IDF', '2': 'Doc2Vec', '3': 'S-BERT'}
    ls_words = msg.text.lower().split(' ')
    if ls_words.count('all'):
        await msg.reply(f"Используемые модели: {str(list(nlp_model.values()[1:]))[1:-1]}")
        environ['model'] = '0'
    elif ls_words.count('tf-idf'):
        await msg.reply('Используется модель: TF-IDF')
        environ['model'] = '1'
    elif ls_words.count('word2vec'):
        await msg.reply('Используется модель: Doc2Vec')
        environ['model'] = '2'
    elif ls_words.count('s-bert'):
        await msg.reply('Используется модель: S-BERT')
        environ['model'] = '3'
    elif msg.text.lower() == '/model':
        await msg.answer(f"Используется модель: { nlp_model.get(getenv('model')) }")
        model_button = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        model_button.add(*['/model TF-IDF', '/model Doc2Vec', '/model S-BERT', '/model all'])
        await msg.reply(f'Выбери режим разметки:', reply_markup=model_button)
    else:
        await msg.reply('Данная модель мне неизвестна.')


# функция разблокирования пользователя
async def unblock_user(msg: types.Message):
    user_id = msg.text.split()[1]
    remove_from_list(user_id, 'blacklist')
    add_to_list(user_id, 'whitelist')
    await msg.bot.send_message(int(user_id), 'Поздравляю, доступ к услугам бота предоставлен!\r\nДля запуска сервиса '
                                             'воспользуйтесь командой /start')

# админская команда на обновление словарей
async def reload_dicts(msg: types.Message):
    reload_dict('all')
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
    elif list_words.count('off'):
        await msg.reply(f'Режим разметки установлен в состояние: OFF')
        environ['is_razmetka'] = '0'
    elif msg.text == '/razmetka':
        await msg.answer(f"Режим разметки: {status_razmetki}")
        razmetka_button = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        razmetka_button.add(*['/razmetka OFF', '/razmetka ON'])
        await msg.reply(f'Выбери режим разметки:', reply_markup=razmetka_button)
    else:
        await msg.reply('Данная команда мне неизвестна.')


class IsAccess(BoundFilter):
    key = 'in_whitelist'

    def __init__(self, in_whitelist):
        self.in_whitelist = in_whitelist

    async def check(self, msg: types.Message):
        new_user_id = msg.from_user.id
        if [int(ids) for ids in getenv('whitelist').split(',')].count(new_user_id):
            return new_user_id


def register_handlers_common(dp: Dispatcher, user_id):
    print(f'Список идентификаторов пользователей, которым разрешено пользоваться сервисом {user_id}')
    dp.register_message_handler(cmd_start, IsAccess(in_whitelist=user_id), commands="start", state="*")
    dp.register_message_handler(cmd_start_new, commands="start", state="*")

    @dp.callback_query_handler(text="add_user")
    async def add_user(call: types.CallbackQuery):
        new_user_id = call.message.reply_markup.inline_keyboard[0][0].text.split(' ')[-1]
        if not (getenv('whitelist')).split(',').count(new_user_id):
            remove_from_list(int(new_user_id), 'blacklist')
            add_to_list(int(new_user_id), 'whitelist')
            await call.message.answer(f'Пользователю с ID [{new_user_id}] предоставлен доступ к сервису')
            await call.bot.send_message(int(new_user_id), text='Вам предоставлен доступ к сервису')

    @dp.callback_query_handler(text="block_user")
    async def block_user(call: types.CallbackQuery):
        new_user_id = call.message.reply_markup.inline_keyboard[0][0].text.split(' ')[-1]
        add_to_list(int(new_user_id), 'blacklist')
        remove_from_list(int(new_user_id), 'whitelist')
        await call.message.answer(f'Пользователю с ID [{new_user_id}] заблокирован доступ к сервису')
        # нужно решить, отдавать ли обратную связь по блокировке, так как могут обидеться и ддосить...
        # но заготовка пусть будет
        # await call.bot.send_message(int(new_user_id), 'К сожалению, доступ к сервису заблокирован')

    @dp.callback_query_handler(text="role_desc")
    async def role_desc(call: types.CallbackQuery):
        change_text = 'На данный момент в фильтре "наименование вакансии" реализован поиск через:\r\n' \
                      '1) операцию или между словами, например, "python разработчик": будут отбираться вакансии,' \
                      'у которых в заголовке вакансии встречается слово "python" или слово "разработчик"\r\n' \
                      '2) операцию или между словосочетаниями, например если ввести такое сообщение: "data analytic" ' \
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
                      'умею работать с git, linux, windows, знаю SQL на таком-то уровне". В общем все то, что вы ' \
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
