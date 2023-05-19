import datetime as dt
import os
import typing as tp

from pypika import Query, Table, Column, Order
import pypika.functions as func
import sqlite3


VARCHAR_MAX_LENGTH = 300
DESCRIPTION_MAXLENGTH = 1000
REFERENCE_MAXLENGTH = 500

USER_REQUESTS_SCHEMA = [
    ('user_id', 'INT', False),
    ('film_id', 'INT', False),
    ('request_datetime', 'DATETIME', False)
]

FILMS_SCHEMA = [
    ('film_id', 'INT', False),
    ('name', f'VARCHAR({VARCHAR_MAX_LENGTH})', False),
    ('description', f'VARCHAR({DESCRIPTION_MAXLENGTH})', True),

    ('year', 'INT', True),
    ('countries', f'VARCHAR({VARCHAR_MAX_LENGTH})', True),
    ('genres', f'VARCHAR({VARCHAR_MAX_LENGTH})', True),
    ('stars', f'VARCHAR({VARCHAR_MAX_LENGTH})', True),
    ('rating', 'REAL', True),
    ('duration', f'VARCHAR({VARCHAR_MAX_LENGTH})', True),

    ('image_ref', f'VARCHAR({REFERENCE_MAXLENGTH})', True),
    ('watch_refs', f'VARCHAR({REFERENCE_MAXLENGTH})', False)
]


class DataBaseHandler:
    def __init__(self, sqlite_database_name: str) -> None:
        """
        Initialize all the context of database handler here
        :param sqlite_database_name: path to the sqlite3 database file
        """
        self.sqlite_database_name = sqlite_database_name
        if os.path.exists(sqlite_database_name):
            self.handler_creates_db = False
        else:
            self.handler_creates_db = True

        self.connection = sqlite3.connect(sqlite_database_name)
        self.cursor = self.connection.cursor()

        if self.handler_creates_db:
            q_create_user_requests_table = Query.create_table('user_requests') \
                .columns(*[Column(col_name, col_type, nullable=nullable)
                           for col_name, col_type, nullable in USER_REQUESTS_SCHEMA])
            self.cursor.execute(str(q_create_user_requests_table))
            self.connection.commit()

            q_create_films_table = Query.create_table('films') \
                .columns(*[Column(col_name, col_type, nullable=nullable)
                           for col_name, col_type, nullable in FILMS_SCHEMA]) \
                .unique('name') \
                .primary_key('film_id')
            self.cursor.execute(str(q_create_films_table))
            self.connection.commit()

        self.user_requests = Table('user_requests')
        self.films = Table('films')

        q = Query.from_(self.films) \
            .select(func.Count('*'))
        self.cursor.execute(str(q))

        self.film_id = self.cursor.fetchone()[0]

    def add_film(self, **kwargs) -> int:
        """
        :param kwargs: new row of table
        :return: id of added film
        """
        self.film_id += 1

        q = Query.into(self.films) \
            .insert(self.film_id, *[kwargs[col_name] for col_name, _, _ in FILMS_SCHEMA[1:]])

        self.cursor.execute(str(q))
        self.connection.commit()

        return self.film_id

    def get_film_by_film_name(self, name: str) -> tuple[tp.Any, ...] | None:
        """
        :param name: film name to search in database
        :return: row with information about this film if film was found, otherwise None
        """
        q = Query.from_(self.films) \
            .where(self.films.name == name) \
            .select(self.films.film_id)

        self.cursor.execute(str(q))
        return self.cursor.fetchone()

    def add_user_request(self, user_id: int, film_id: int) -> None:
        """
        :param user_id: id of user making request
        :param film_id: id of requested film
        """
        q = Query.into(self.user_requests) \
            .insert(user_id, film_id, dt.datetime.now())
        self.cursor.execute(str(q))
        self.connection.commit()

    def get_user_history(self, user_id: int) -> tp.Sequence[tuple[str, str]]:
        """
        :param user_id: id of user making request
        :return: sequence of pairs (request datetime, name of requested film) sorted desc by datetime
        """
        q = Query.from_(self.user_requests) \
            .where(self.user_requests.user_id == user_id) \
            .join(self.films).using('film_id') \
            .select(self.films.name, self.user_requests.request_datetime) \
            .orderby(self.user_requests.request_datetime, order=Order.desc)

        self.cursor.execute(str(q))
        return self.cursor.fetchall()

    def get_user_stats(self, user_id: int) -> tp.Sequence[tuple[str, int]]:
        """
        :param user_id: id of user making request
        :return: sequence of pairs (name of requested film, requests number) sorted desc by requests number
        """
        grouped_stat = Query.from_(self.user_requests) \
            .where(self.user_requests.user_id == user_id) \
            .groupby(self.user_requests.film_id) \
            .select(self.user_requests.film_id, func.Count('*').as_('count'))

        q = Query.from_(grouped_stat) \
            .join(self.films).using('film_id') \
            .select(self.films.name, grouped_stat.count) \
            .orderby(grouped_stat.count, order=Order.desc)

        self.cursor.execute(str(q))
        return self.cursor.fetchall()

    def teardown(self) -> None:
        """
        Cleanup everything after working with database
        """
        self.cursor.close()
        self.connection.close()
