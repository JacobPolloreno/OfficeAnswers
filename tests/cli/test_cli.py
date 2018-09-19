import click
import pytest

from click.testing import CliRunner


@pytest.fixture(scope="module")
def click_runner():
    return CliRunner()


def test_dummy(click_runner):
    runner = click_runner
    assert runner
