#!/usr/bin/env python3
import logging
import click

from data_processing.processed_data_interface import clear_processed_data_cache
from data_processing.ml_data_prepairer import get_ml_data
from models.save_file_helper import delete_model_debugging_files
from models.build_decision_engine import get_decision_engine, delete_cached_model
from models.analysis.decision_engine_analyzer import DecisionEngineAnalyzer, delete_previous_analysis_reports, ANALYSIS_RESULTS_DIR

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
    clear_processed_data_cache()
    click.echo("Cached preprocessed data removed")
    delete_model_debugging_files()
    click.echo("Model debugging files removed")
    delete_cached_model()
    click.echo("Decision engine cache deleted")
    delete_previous_analysis_reports()
    click.echo("Analysis reports deleted")

@cli.command(help="Build machine learning feature set")
@click.pass_context
def pd(ctx):
    # When building a new dataset, we should clear all cache since the models and analysis are no longer valid
    clean(ctx)
    get_ml_data()
    click.echo("New dataset built")

@cli.command(help="Build the decision engine")
@click.pass_context
def bde(ctx):
    delete_cached_model()
    click.echo("Decision engine cache deleted")
    delete_previous_analysis_reports()
    click.echo("Analysis reports deleted")
    get_decision_engine(get_ml_data())
    click.echo("Decision engine created")

@cli.command(help="Run decision engine analysis")
@click.pass_context
def dea(ctx):
    delete_previous_analysis_reports()
    click.echo("Analysis reports deleted")
    ml_data = get_ml_data()
    decision_engine = get_decision_engine(ml_data)
    analyzer = DecisionEngineAnalyzer(decision_engine, ml_data)
    analyzer.create_analysis_reports()
    click.echo("Reports created in directory %s" % ANALYSIS_RESULTS_DIR)


if __name__ == '__main__':
    cli(obj={})
