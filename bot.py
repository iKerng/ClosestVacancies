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


def sql_def():
    # super.__init__()
    db_path = path.abspath(getcwd()) + '\\data\\vacancies.db'
    connect = sql.connect(db_path)
    return connect.cursor()

class OrderParams(StatesGroup):
    waiting_schedule = State()
    waiting_region = State()

async def cmd_start(msg: types.Message, state: FSMContext):
    print(msg)
    await state.finish()
    await OrderParams.waiting_schedule.set()
    cur = sql_def()
    cols = DataFrame(cur.execute("SELECT name FROM pragma_table_info('schedule')"))[0].to_list()
    df_schedule = DataFrame(cur.execute('''SELECT * FROM SCHEDULE'''), columns=cols)
    cur.close()
    work_type_office = df_schedule['name'].to_list()
    work_type_button = types.ReplyKeyboardMarkup(resize_keyboard=True)
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


# todo нужно дописать админскую настройку для выбора режима обработки текста

async def secret_command(message: types.Message):
    print(f"Разметка до: [{getenv('is_razmetka')}]")
    status_razmetki = 'включен' if getenv('is_razmetka') else 'выключен'
    list_words = message.text.lower().split(' ')
    input_command = list_words[1] if len(list_words) > 1 else None
    commands = ['включить', 'включи', 'выключить', 'выключи']
    if commands[0:2].count(input_command):
        print(getenv('is_razmetka'))
        await message.reply(f'Режим разметки установлен в состояние: включен')
        environ['is_razmetka'] = '1'
        print(getenv('is_razmetka'))
    elif commands[2:4].count(input_command):
        print(getenv('is_razmetka'))
        await message.reply(f'Режим разметки установлен в состояние: выключен')
        environ['is_razmetka'] = '0'
        print(getenv('is_razmetka'))
    elif message.text == '/razmetka':
        await message.answer(f"Приветствую тебя, админ! На текущий момент параметр разметки {status_razmetki}")
        # await message.answer(f'Текущее состояние разметки: {status_razmetki}')
        razmetka_button = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        razmetka_button.add(*['/razmetka выключить', '/razmetka включить'])
        await message.reply(f'Выбери режим разметки:', reply_markup=razmetka_button)
    else:
        await message.reply('Данная команда мне неизвестна.')
    print(f"Разметка после: [{getenv('is_razmetka')}]")


def register_handlers_common(dp: Dispatcher, user_id: int):
    dp.register_message_handler(cmd_start, commands="start", state="*")
    dp.register_message_handler(cmd_cancel, commands="cancel", state="*")
    # dp.register_message_handler(cmd_cancel, Text(equals="отмена", ignore_case=True), state="*")
    dp.register_message_handler(secret_command, IDFilter(user_id=user_id), commands="razmetka", state='*')

# def register_continue(dp: Dispatcher):



async def main():
    reload_dict('all')

    bot_token = getenv('bot_token')
    if not bot_token:
        exit('Error: Bot token not found in enviroment variables')

    bot = Bot(token=bot_token)
    dp = Dispatcher(bot, storage=MemoryStorage())

    search_params = dict()
    bot_admin = getenv('bot_admin')
    register_handlers_common(dp, bot_admin)
    register_continue(dp)

    await dp.start_polling()


if __name__ == '__main__':
    asyncio.run(main())
