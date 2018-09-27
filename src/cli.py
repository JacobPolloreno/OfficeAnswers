"""Office Answers CLI

"""

import click
import logging
import os

from officeanswers.preprocess import prepare_and_preprocess
from officeanswers.model import train as model_train
from officeanswers.model import predict as model_predict
from officeanswers.util import Config

from matchzoo import engine


logger = logging.getLogger(__name__)


class ExpandedPath(click.Path):
    def convert(self, value, *args, **kwargs):
        value = os.path.expanduser(value)
        return super(ExpandedPath, self).convert(value, *args, **kwargs)


@click.group()
@click.argument('model_config_file',
                type=ExpandedPath(exists=True))
@click.option('--debug', is_flag=True)
@click.pass_context
def cli(ctx, debug, model_config_file):
    """CLI for the office assistant model.

    Example Usage:\n
        python src/main.py config/model.config train
    """
    ctx.obj['DEBUG'] = debug
    ctx.obj['CONFIG'] = Config()
    ctx.obj['CONFIG'].from_json_file(model_config_file)

    logger.info("Loaded configuration file...")


@cli.command()
@click.pass_context
@click.option('--save', '-s', is_flag=True, help="Save corpus and relations")
def prepare(ctx, save: bool=False):
    prepare_and_preprocess(ctx.obj['CONFIG'], save)


@cli.command()
@click.pass_context
def train(ctx):
    model_train(ctx.obj['CONFIG'])


@cli.command()
@click.pass_context
@click.option('--num_largest', '-n', type=int, default=5)
def predict(ctx, num_largest):
    logger.info('Loading model...')

    config = ctx.obj['CONFIG']
    pr_dir = config.paths['processed_dir']
    model_type = config.model['type']
    net_name = config.net_name

    if model_type.lower() == 'dssm':
        model = engine.load_model(pr_dir, net_name)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    query = click.prompt("What do you want to search?", type=str)
    while query and query != 'exit':
        results = model_predict(ctx.obj['CONFIG'],
                                model,
                                query,
                                num_largest)
        for res in results:
            print(res)
        query = click.prompt("What do you want to search?", type=str)


@cli.command()
@click.pass_context
def test(ctx):
    logger.info('Testing model...')


if __name__ == "__main__":
    cli(obj={})
