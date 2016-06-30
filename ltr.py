#!/usr/bin/env python3
import logging
import click

from data_processing.processed_data_interface import clear_processed_data_cache
from data_processing import build_machine_learning_dataset
from models.save_file_helper import delete_model_debugging_files

_LOG_LEVELS = [
    'CRITICAL'
    'ERROR',
    'WARNING',
    'INFO',
    'DEBUG'
]

@click.group()
@click.option('--ll', type=click.Choice(_LOG_LEVELS), help="The log level", default='INFO')
@click.option('--cache', default=True)
@click.pass_context
def cli(ctx, ll, cache):
    logger = logging.getLogger()
    logger.setLevel(ll)
    ctx.obj['use_cache'] = cache

@cli.command(help="Remove all cached data")
@click.pass_context
def clean(ctx):
    _clean()

@cli.command(help="Build machine learning feature set")
@click.pass_context
def pd(ctx):
    # When building a new dataset, we should clear all cache since the models and analysis are no longer valid
    _clean()
    click.echo("Building new dataset")
    build_machine_learning_dataset()
    click.echo("New dataset built")

@cli.command(help="Build the decision engine")
@click.pass_context
def bde(ctx):
    print('')

@cli.command(help="Run decision engine analysis")
@click.pass_context
def dea(ctx):
    print('wassup')

def _clean():
    clear_processed_data_cache()
    click.echo("Cached preprocessed data removed")
    delete_model_debugging_files()
    click.echo("Model debugging files removed")

if __name__ == '__main__':
    cli(obj={})
