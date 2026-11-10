import os
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

from app import db

ADMIN_URL = os.environ.get("ADMIN_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
TEST_DB = "jobs_test"


def _with_db(url: str, name: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, "/" + name, p.query, p.fragment))


@pytest.fixture(scope="session")
def db_url() -> str:
    with psycopg.connect(ADMIN_URL, autocommit=True) as c:
        c.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
        c.execute(f'CREATE DATABASE "{TEST_DB}"')
    url = _with_db(ADMIN_URL, TEST_DB)
    db.init_schema(url)
    return url


@pytest.fixture()
def conn(db_url):
    c = db.connect(db_url)
    c.execute("TRUNCATE jobs, executions, effects, stale_writes, job_transitions RESTART IDENTITY CASCADE")
    yield c
    c.close()


@pytest.fixture()
def new_conn(db_url):
    opened = []

    def make():
        c = db.connect(db_url)
        opened.append(c)
        return c
    yield make
    for c in opened:
        c.close()
