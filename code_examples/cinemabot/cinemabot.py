import datetime as dt
import os

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
import aiohttp

from DataBaseHandler import DataBaseHandler
from http_tools import find_imdb_id, get_imdb_info_by_id, find_watch_references, check_img_size_and_read_if_needed
from msg_tools import (
    START_GREETING,
    HISTORY_PREFIX,
    STATS_PREFIX,
    NO_HISTORY_MESSAGE,
    NO_FILM_WAS_FOUND_MESSAGE,
    ASK_PASS_ARG_TO_FIND,
    get_film_answer
)


DATABASE_HANDLER = DataBaseHandler('database.db')
IMDB_API_TOKEN = os.environ['IMDB_API_TOKEN']


bot = Bot(token=os.environ['BOT_TOKEN'])
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def send_start_and_help_message(message: types.Message):
    await message.answer(START_GREETING, parse_mode='HTML')


@dp.message_handler(commands=['history'])
async def get_history(message: types.Message):
    ans = ''
    for film_name, req_dt in DATABASE_HANDLER.get_user_history(message.from_id):
        ans += '\n' + dt.datetime.fromisoformat(req_dt).strftime('%-d %b %Y %H:%M:%S') + ' - ' + film_name

    if ans:
        await message.answer(HISTORY_PREFIX + ans)
    else:
        await message.answer(NO_HISTORY_MESSAGE)


@dp.message_handler(commands=['stats'])
async def get_stats(message: types.Message):
    ans = ''
    for film_name, count in DATABASE_HANDLER.get_user_stats(message.from_id):
        ans += '\n' + str(count) + ' - ' + film_name

    if ans:
        await message.answer(STATS_PREFIX + ans)
    else:
        await message.answer(NO_HISTORY_MESSAGE)


@dp.message_handler(commands=['find'])
async def find_film(message: types.Message):
    if not message.get_args():
        await message.reply(ASK_PASS_ARG_TO_FIND)
        return

    async with aiohttp.ClientSession() as session:
        imdb_id = await find_imdb_id(session, message.get_args())
        if imdb_id is None:
            await message.answer(NO_FILM_WAS_FOUND_MESSAGE)
            return

        film_info = await get_imdb_info_by_id(session, IMDB_API_TOKEN, imdb_id)
        if film_info['name'] is None:
            await message.answer(NO_FILM_WAS_FOUND_MESSAGE)
            return

        film_info['watch_refs'] = '\n'.join(await find_watch_references(session, film_info['name']))

    film_from_db = DATABASE_HANDLER.get_film_by_film_name(film_info['name'])
    if film_from_db is None:
        film_id = DATABASE_HANDLER.add_film(**film_info)
    else:
        film_id = film_from_db[0]
    DATABASE_HANDLER.add_user_request(message.from_id, film_id)

    img_content = await check_img_size_and_read_if_needed(film_info['image_ref'])
    if img_content is None:
        await message.answer_photo(film_info['image_ref'],
                                   caption=get_film_answer(film_info),
                                   parse_mode='HTML')
    else:
        await message.answer_photo(types.InputFile(img_content),
                                   caption=get_film_answer(film_info),
                                   parse_mode='HTML')


@dp.message_handler()
async def side_requests(message: types.Message):
    await message.answer(START_GREETING, parse_mode='HTML')


async def on_startup(dp: Dispatcher):
    await bot.set_my_commands([
        types.BotCommand('start', 'Start bot'),
        types.BotCommand('help', 'Help'),
        types.BotCommand('history', 'Requests history'),
        types.BotCommand('stats', 'Requests summary'),
        types.BotCommand('find', 'Find info about film')
    ])
    await dp.bot.set_chat_menu_button(menu_button=types.MenuButtonCommands())


async def on_shutdown(dp: Dispatcher):
    DATABASE_HANDLER.teardown()


if __name__ == '__main__':
    executor.start_polling(dp, on_startup=on_startup, on_shutdown=on_shutdown, skip_updates=True)
