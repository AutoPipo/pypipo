# -*- coding: utf-8 -*-

import click
import __version__
from convert import convert_image


class Config(object):
    def __init__(self):
        self.config = {}

    def set_config(self, key, value):
        self.config[key] = value


pass_config = click.make_pass_decorator(Config)


@click.group(name="pypipo")
@click.version_option(version=__version__)
# @click.option("-f", "--filepath", help="File path that want to process.")
# @click.argument("filepath", type=click.Path(exists=True))
@click.pass_context
def cli(ctx, *args, **kwargs):
    """Pypipo : Automatically convert to PIPO Painting canvas."""
    ctx.obj = Config()
    for key, value in kwargs.items():
        ctx.obj.set_config(key, value)

'''
Usage: python -m  [OPTIONS] COMMAND [ARGS]...

  Pypipo : Automatically convert to PIPO Painting canvas.

Options:
  --version            Show the version and exit.
  -f, --filepath TEXT  File path that want to process.
  --help               Show this message and exit.

Commands:
  process  explain about this function
'''

@cli.command("process")
@click.option(
    "-k",
    "--number",
    default="16",
    help="The number of clusted color of image.",
)
@click.argument("filepath", type=click.Path(exists=True))
@pass_config
def run(c, *args, **kwargs):
    """explain about this function"""
    conf = c.config
    print(f"conf is {conf}")
    # number = conf.pop("number")
    filepath = kwargs.pop("filepath")
    # kwargs.update(conf)
    print(filepath,  kwargs)
    output = convert_image(filepath,  **kwargs)

    # click.echo("Found {} tables".format(tables.n))

    return 

