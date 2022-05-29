import asyncio
from os import getenv, environ

from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from calculate.bot import register_handlers_common, register_continue

from calculate.set_env_default_params import set_env


async def run_bot():

    # берем токен телеграм-бота из виртуального окружения
    bot_admin = getenv('bot_admin')

    dp = Dispatcher(bot, storage=MemoryStorage(), loop=loop)

    # проверяем не является ли сообщение командой пользователя или командой пользователя-админа
    register_handlers_common(dp, int(bot_admin))
    register_continue(dp)

    # запускаем бота
    await dp.start_polling()


if __name__ == '__main__':
    set_env()

    bot_token = getenv('bot_token')
    if not bot_token:
        exit('Error: токен бота не найден')

    bot = Bot(token=bot_token)
    loop = asyncio.get_event_loop()
    loop.create_task(run_bot())
    loop.run_forever()

    asyncio.create_task(run_bot())
