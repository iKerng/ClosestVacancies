import asyncio
from os import getenv

from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from calculate.bot import register_handlers_common, register_continue


async def main():
    # берем токен телеграм-бота из виртуального окружения
    bot_token = getenv('bot_token')
    if not bot_token:
        exit('Error: Bot token not found in enviroment variables')

    bot = Bot(token=bot_token)
    dp = Dispatcher(bot, storage=MemoryStorage())

    bot_admin = getenv('bot_admin')

    # проверяем не является ли сообщение командой пользователя или командой пользователя-админа
    register_handlers_common(dp, int(bot_admin))
    register_continue(dp)

    # запускаем бота
    await dp.start_polling()


if __name__ == '__main__':
    asyncio.run(main())
