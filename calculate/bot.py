from os import environ, getenv

from aiogram import Dispatcher, types
from aiogram.dispatcher.filters import IDFilter, Text

from calculate.workflow import cmd_start, cmd_cancel
from calculate.workflow import choose_schedule, OrderParams, choose_city
from calculate.workflow import choose_region, analyze_description,  search_vacs_word_keys
from calculate.preload_dicts import reload_dict


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


def register_handlers_common(dp: Dispatcher, user_id: int):
    dp.register_message_handler(cmd_start, commands="start", state="*")
    dp.register_message_handler(cmd_start, Text(equals=['старт', 'начать', 'запуск', 'запустить'], ignore_case=True),
                                state="*")
    dp.register_message_handler(cmd_cancel, commands="cancel", state="*")
    dp.register_message_handler(cmd_cancel, Text(equals=['отмена', 'отменить'], ignore_case=True), state="*")
    dp.register_message_handler(switch_mode_razmetki, IDFilter(user_id=user_id), commands="razmetka", state='*')
    dp.register_message_handler(switch_nlp_model, IDFilter(user_id=user_id), commands="model", state='*')
    dp.register_message_handler(reload_dicts, IDFilter(user_id=user_id), commands="reload_dicts", state='*')


def register_continue(dp: Dispatcher):
    # функция перехода по состояниям
    dp.register_message_handler(choose_schedule, state=OrderParams.waiting_schedule)
    dp.register_message_handler(choose_city, state=OrderParams.waiting_city)
    dp.register_message_handler(choose_region, state=OrderParams.waiting_region)
    dp.register_message_handler(analyze_description, state=OrderParams.waiting_description)
    dp.register_message_handler(search_vacs_word_keys, state=OrderParams.waiting_search_word_keys)
