import json
import typing as tp
from io import BytesIO

import aiohttp
from bs4 import BeautifulSoup
from PIL import Image


GOOGLE_SEARCH_URL = 'http://www.google.com/search?q='
IMDB_API_PREFIX = 'https://imdb-api.com/API/Title/'
IMDB_URL_PREFIX = 'https://www.imdb.com/title/'

WATCH_WEBSITES_MARKERS = [
    'kinopoisk', 'more.tv', 'ivi', 'megogo', 'kinogo', 'okko', 'lordfilm', 'film.ru', 'netflix'
]

HEADERS = {'User-Agent': ''}

MAX_CONTENT_LENGTH = 1000000
RESIZED_HEIGHT = 1000.0


async def find_imdb_id(session: aiohttp.ClientSession, query: str) -> str | None:
    async with session.get(GOOGLE_SEARCH_URL + 'imdb+' + query.replace(' ', '+'), headers=HEADERS) as resp:
        soup = BeautifulSoup(await resp.text(), 'html.parser')

    refs = soup.find_all("a")

    for elem in refs:
        link = elem["href"]
        if link.startswith('/url'):
            url = link[7:].split('&')[0]
            if url.startswith(IMDB_URL_PREFIX):
                id_candidate = url[len(IMDB_URL_PREFIX):].split('/')[0]
                if id_candidate.startswith('tt'):
                    return id_candidate
    return None


async def get_imdb_info_by_id(session: aiohttp.ClientSession, imdb_api_token: str, imdb_id: str) -> dict[str, tp.Any]:
    async with session.get(IMDB_API_PREFIX + imdb_api_token + '/' + imdb_id) as resp:
        info = json.loads(await resp.text())

    year = info.get('year')
    rating = info.get('imDbRating')
    return {
        'name': info.get('fullTitle') or info.get('title'),
        'description': info.get('plot'),

        'year': year if year is None else int(year),
        'countries': info.get('countries'),
        'genres': info.get('genres'),
        'stars': info.get('stars'),
        'rating': rating if rating is None else float(rating),
        'duration': info.get('runtimeStr'),

        'image_ref': info.get('image')
    }


async def find_watch_references(session: aiohttp.ClientSession, film_name: str) -> list[str]:
    async with session.get(GOOGLE_SEARCH_URL + 'смотреть+онлайн+' + film_name.replace(' ', '+'),
                           headers=HEADERS) as resp:
        soup = BeautifulSoup(await resp.text(), 'html.parser')

    refs = soup.find_all("a")
    res = []
    for elem in refs:
        link = elem["href"]
        if link.startswith('/url'):
            url = link[7:].split('&')[0]
            for marker in WATCH_WEBSITES_MARKERS:
                if url not in res and marker in url:
                    res.append(url)

    if not res:
        print('No watch ref was found :(')
    return res


async def check_img_size_and_read_if_needed(url: str) -> BytesIO | None:
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=HEADERS) as resp:
            content_length = resp.headers.get('Content-Length')
            if content_length is None or int(content_length) > MAX_CONTENT_LENGTH:
                content = await resp.read()
            else:
                return None

    img = Image.open(BytesIO(content))
    scale = RESIZED_HEIGHT / img.size[1]
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
    img = img.resize(new_size)

    bio = BytesIO()
    img.save(bio, 'JPEG')
    bio.seek(0)
    return bio
