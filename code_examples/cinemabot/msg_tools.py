import typing as tp


START_GREETING = '''
Hi!
I'm CinemaBot.

I am supposed to help you to find information about different films in the whole Internet!
This info includes short film description, year of production, genres, main stars, etc.
I also try to find links where you can possibly watch these films!

Here is a pool of commands supported by me :)

/start or /help - see this message again
/find <i>film name</i> - will find info about passed film
/history - show history of your requests
/stats - show statistic of how many times each film was suggested for you by CinemaBot
'''

HISTORY_PREFIX = 'The history of your film search requests is listed below:\n'
STATS_PREFIX = 'Here is how many times each film was suggested for you by CinemaBot:\n'

NO_HISTORY_MESSAGE = 'You didn\'t commit any requests yet!'
NO_FILM_WAS_FOUND_MESSAGE = 'Sorry, but we cannot find movie or series with such name :('
ASK_PASS_ARG_TO_FIND = 'Please, enter the film name after /find command to search it :)'


def get_film_answer(film_info: dict[str, tp.Any]) -> str:
    res = f'<b>{film_info["name"]}</b>\n\n'
    if film_info['description'] is not None:
        res += film_info['description'] + '\n\n'

    if film_info['year'] is not None:
        res += 'Year: <b>' + str(film_info['year']) + '</b>\n'
    if film_info['countries'] is not None:
        res += 'Country: <b>' + film_info['countries'] + '</b>\n'
    if film_info['genres'] is not None:
        res += 'Genres: <b>' + film_info['genres'] + '</b>\n'
    if film_info['stars'] is not None:
        res += 'Stars: <b>' + film_info['stars'] + '</b>\n'
    if film_info['rating'] is not None:
        res += 'Imdb rating: <b>' + str(film_info['rating']) + '</b>\n'
    if film_info['duration'] is not None:
        res += 'Duration: <b>' + film_info['duration'] + '</b>\n'

    if film_info['watch_refs']:
        res += '\nYou can try to watch it using following links:\n\n' + film_info['watch_refs']
    else:
        res += '\nUnfortunately, we cannot find links to watch it :('

    return res
