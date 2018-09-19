"""Office Assistant"""

import click
import json
import os
import sys


class ExpandedPath(click.Path):
    def convert(self, value, *args, **kwargs):
        value = os.path.expanduser(value)
        return super(ExpandedPath, self).convert(value, *args, **kwargs)


@click.group()
@click.argument('model_config_file',
                type=ExpandedPath(exists=True),
                default=os.path.expanduser('./configs/model.config'))
@click.option('--debug', is_flag=True)
@click.pass_context
def cli(ctx, debug, model_config_file):
    """CLI for the office assistant model.

    Example Usage:\n
        python src/main.py config/model.config train
    """
    ctx.obj['DEBUG'] = debug
    with open(model_config_file, 'r') as f:
        try:
            ctx.obj['CONFIG'] = json.load(f)
        except json.decoder.JSONDecodeError:
            error_msg = f"Error reading `{model_config_file} file`\n" + \
                    "Expecting json file. See `data/sample_model.config'"
            click.echo(
                click.style(error_msg, fg="red"),
                err=True)
            sys.exit(-1)
    click.echo(click.style("Loaded configuration file...", fg="green"))


@cli.command()
@click.pass_context
def train(ctx):
    click.echo('Training model')


@cli.command()
@click.pass_context
def predict(ctx):
    click.echo('Running predictions')


@cli.command()
@click.pass_context
def test(ctx):
    click.echo('Testing model')


if __name__ == "__main__":
    cli(obj={})
