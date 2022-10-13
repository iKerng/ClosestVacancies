import asyncio
from os import getenv

from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from calculate.bot import register_handlers_common, register_continue, register_handlers_admin

from data.set_env_default_params import set_env
from data.set_env_blacklist import set_env_blacklist
from data.set_env_whitelist import set_env_whitelist
from calculate.log_writer import to_log

async def run_bot():

    # берем токен телеграм-бота из виртуального окружения
    bot_admin = getenv('bot_admin')
    whitelist = [int(idents) for idents in getenv('whitelist').split(',')]

    dp = Dispatcher(bot, storage=MemoryStorage(), loop=loop)
    register_handlers_common(dp, whitelist)
    register_handlers_admin(dp, int(bot_admin))
    register_continue(dp, whitelist)

    # запускаем бота
    # razmetka_status = 'включен' if getenv('is_razmetka') else 'выключен'
    to_log(log_text='Запускаем приложение бота со следующими параметрами')
    to_log(log_text=f"Режим разметки: {getenv('is_razmetka')}")
    to_log(log_text=f"Разметка выполнена: {getenv('razmetka_done')}")
    to_log(log_text=f"Используется модель (1: TF-IDF; 2: Doc2Vec; 3: S-BERT): {getenv('model')}")
    await dp.start_polling()


if __name__ == '__main__':
    set_env()
    set_env_whitelist()
    set_env_blacklist()

    bot_token = getenv('bot_token')
    if not bot_token:
        exit('Error: токен бота не найден')

    bot = Bot(token=bot_token)
    loop = asyncio.get_event_loop()
    loop.create_task(run_bot())
    loop.run_forever()

    asyncio.create_task(run_bot())
