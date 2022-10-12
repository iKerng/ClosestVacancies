import asyncio
from os import getenv

from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from calculate.bot import register_handlers_common, register_continue, register_handlers_admin

from data.set_env_default_params import set_env
from data.set_env_blacklist import set_env_blacklist
from data.set_env_whitelist import set_env_whitelist


async def run_bot():

    # берем токен телеграм-бота из виртуального окружения
    bot_admin = getenv('bot_admin')
    whitelist = [int(idents) for idents in getenv('whitelist').split(',')]

    dp = Dispatcher(bot, storage=MemoryStorage(), loop=loop)
    register_handlers_common(dp, whitelist)
    register_handlers_admin(dp, int(bot_admin))
    register_continue(dp, whitelist)

    # запускаем бота
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
