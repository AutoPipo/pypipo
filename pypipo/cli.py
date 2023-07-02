# -*- coding: utf-8 -*-

import click
import __version__
from convert import pipo_convert


class Config(object):
    def __init__(self):
        self.config = {}

    def set_config(self, key, value):
        self.config[key] = value


pass_config = click.make_pass_decorator(Config)


@click.group(name="pypipo")
@click.version_option(version=__version__)
# @click.argument("filepath", type=click.Path(exists=True))
@click.pass_context
def cli(ctx, *args, **kwargs):
    """Pypipo : Automatically convert to PIPO Painting canvas."""
    ctx.obj = Config()
    for key, value in kwargs.items():
        ctx.obj.set_config(key, value)


@cli.command("convert")
@click.option(
    "-n",
    "--number",
    default=16,
    help="Number of color clustered",
)
@click.option(
    "-a",
    "--attempts",
    default=1,
    help="How many iterate try to k-means clustering",
)
@click.option(
    "-u",
    "--is_upscale",
    default=False,
    help="Expand size of image",
)
@click.option(
    "-t",
    "--target_size",
    default=3,
    help="Size that want to expand image.",
)
@click.option(
    "-d",
    "--div",
    default=8,
    help="Reducing numbers of color on image",
)
@click.option(
    "-S",
    "--sigma",
    default=20,
    help="bilateralFilter Parameter",
)
@click.option(
    "-c",
    "--color_label",
    default=True,
    help="Show color label at left-top of output image",
)
@click.argument("filepath", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(writable=True))
@pass_config
def convert(c, *args, **kwargs):
    filepath = kwargs.pop("filepath")
    output_path = kwargs.pop("output_path")
    color_label = kwargs.pop("color_label")
    output = pipo_convert(filepath, output_path, color_label, **kwargs)

    click.echo("Image Converting Finished!")

    return output

