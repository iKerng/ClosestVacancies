import asyncio
from os import path, getcwd, getenv, environ

from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher.filters import IDFilter
from aiogram.contrib.fsm_storage.memory import MemoryStorage

import sqlite3 as sql
from pandas import DataFrame

from preload import reload_dict, get_regions, reload_dict
from get_params import register_continue

# функция подключения к БД для sql-запросов
def sql_def():
    db_path = path.abspath(getcwd()) + '\\data\\vacancies.db'
    connect = sql.connect(db_path)
    return connect.cursor()

class OrderParams(StatesGroup):
    waiting_schedule = State()

async def cmd_start(msg: types.Message, state: FSMContext):
    # Если приходит новая команда /start, то мы прекращаем все предыдущие состояния
    await state.finish()
    await OrderParams.waiting_schedule.set()
    cur = sql_def()
    # запрос к БД на получение списка колонок
    cols = DataFrame(cur.execute("SELECT name FROM pragma_table_info('schedule')"))[0].to_list()
    # запрос к БД на получение значений словаря режима работы
    df_schedule = DataFrame(cur.execute('''SELECT * FROM SCHEDULE'''), columns=cols)
    cur.close()
    work_type_office = df_schedule['name'].to_list()
    work_type_button = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    work_type_button.add(*work_type_office)
    await msg.reply(
        f"{msg.from_user.first_name}, Вас приветствует бот по поиску вакансий с использованием Natural " +\
        f"Language Processing" +
        "\r\n\r\nДля поиска вакансий необходимо предварительно получить некоторые данные." +
        "\r\n\r\nВыберите формат работы", reply_markup=work_type_button)

# async switch_razmetka()

async def cmd_cancel(message: types.Message, state: FSMContext):
    await state.finish()
    await message.answer("Действие отменено", reply_markup=types.ReplyKeyboardRemove())


# админская настройка для выбора используемой модели
async def switch_nlp_model(msg: types.Message):
    nlp_model = {'0': 'all', '1': 'TF-IDF', '2': 'Word2Vec', '3': 'S-BERT'}
    ls_words = msg.text.lower().split(' ')
    if ls_words.count('all'):
        await msg.reply(f"Используемые модели: {str(list(nlp_model.values()[1:]))[1:-1]}")
        environ['model'] = '0'
    elif ls_words.count('tf-idf'):
        await msg.reply('Используется модель: TF-IDF')
        environ['model'] = '1'
    elif ls_words.count('word2vec'):
        await msg.reply('Используется модель: Word2Vec')
        environ['model'] = '2'
    elif ls_words.count('s-bert'):
        await msg.reply('Используется модель: S-BERT')
        environ['model'] = '3'
    elif msg.text.lower() == '/model':
        await msg.answer(f"Используется модель: { nlp_model.get(getenv('model')) }")
        model_button = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        model_button.add(*['/model TF-IDF', '/model Word2Vec', '/model S-BERT', '/model all'])
        await msg.reply(f'Выбери режим разметки:', reply_markup=model_button)
    else:
        await msg.reply('Данная модель мне неизвестна.')

# админская команда на обновление словарей
async def reload_dicts(msg: types.Message):
    reload_dict('all')



async def switch_mode_razmetki(msg: types.Message):
    # расшифровываем код переменной из виртуального окружения
    status_razmetki = 'ON' if getenv('is_razmetka') else 'OFF'
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


def register_handlers_common(dp: Dispatcher, user_id: int):
    dp.register_message_handler(cmd_start, commands="start", state="*")
    dp.register_message_handler(cmd_cancel, commands="cancel", state="*")
    # todo добавить обработку команды по слову "отмена"
    # dp.register_message_handler(cmd_cancel, Text(equals="отмена", ignore_case=True), state="*")
    dp.register_message_handler(switch_mode_razmetki, IDFilter(user_id=user_id), commands="razmetka", state='*')
    dp.register_message_handler(switch_nlp_model, IDFilter(user_id=user_id), commands="model", state='*')
    dp.register_message_handler(reload_dicts, IDFilter(user_id=user_id), commands="reload_dicts", state='*')




async def main():
    # берем токен телеграм-бота из виртуального окружения
    bot_token = getenv('bot_token')
    if not bot_token:
        exit('Error: Bot token not found in enviroment variables')

    bot = Bot(token=bot_token)
    dp = Dispatcher(bot, storage=MemoryStorage())

    bot_admin = getenv('bot_admin')

    # проверяем не является ли сообщение командой пользователя или командой пользователя-админа
    register_handlers_common(dp, bot_admin)
    register_continue(dp)

    # запускаем бота
    await dp.start_polling()


if __name__ == '__main__':
    asyncio.run(main())