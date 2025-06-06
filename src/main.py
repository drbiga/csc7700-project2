import os
from pathlib import Path

import warnings

from util import add_title_to_pagerank

from tfidf_computation import run_tfidf, compute_score, get_spark

from pagerank import (
    get_top_n_ranked_nodes,
    compute_pageranks,
    sum_all_pageranks,
    compute_pagerank2,
)

from dataset import (
    create_sample_parquet,
    create_sample_json,
    parse_db,
    plot_sampled_citation_graph,
)

from experiments import (
    generate_query_database,
    alpha,
    query_performance,
    average_required_size,
)

from search_engine import SearchEngine, load_parquet_to_elasticsearch

from ascii import printSampleComplete

from pyspark.sql import SparkSession


ORIGINAL_DB_PATH = "data/dblp.v12.json"
PARSED_DB_PATH = "data/parsed.json"

SAMPLE_TFIDF_DIR = "sample-spark-tfidf"
SAMPLE_TFIDF_VECTORS_OUTPUT_PATH = f"{SAMPLE_TFIDF_DIR}/vectors.parquet"
SAMPLE_TFIDF_PIPELINE_OUTPUT_PATH = f"{SAMPLE_TFIDF_DIR}/model.parquet"
SAMPLE_PAGERANK_OUTPUT_PATH = "sample-spark-pageranks"

ENTIRE_DATABASE_TFIDF_DIR = "entire-database-spark-tfidf"
ENTIRE_DATABASE_TFIDF_VECTORS_OUTPUT_PATH = (
    f"{ENTIRE_DATABASE_TFIDF_DIR}/vectors.parquet"
)
ENTIRE_DATABASE_TFIDF_PIPELINE_OUTPUT_PATH = (
    f"{ENTIRE_DATABASE_TFIDF_DIR}/pipeline.model"
)
ENTIRE_DATABASE_PAGERANK_OUTPUT_PATH = "entire-database-spark-pageranks"
ENTIRE_DATABASE_EXPERIMENTS_DIR = "entire-database-spark-experiments"
ENTIRE_DATABASE_PAGERANK_WITH_TITLES_OUTPUT_PATH = (
    f"{ENTIRE_DATABASE_EXPERIMENTS_DIR}/pageranks_with_titles.parquet"
)
ENTIRE_DATABASE_EXPERIMENTS_ALPHA_DIR = f"{ENTIRE_DATABASE_EXPERIMENTS_DIR}/alpha"
ENTIRE_DATABASE_EXPERIMENTS_ALPHA_VALUES = (
    f"{ENTIRE_DATABASE_EXPERIMENTS_ALPHA_DIR}/times.csv"
)
ENTIRE_DATABASE_EXPERIMENTS_ALPHA_STATS_DIR = (
    f"{ENTIRE_DATABASE_EXPERIMENTS_ALPHA_DIR}/stats"
)
ENTIRE_DATABASE_EXPERIMENTS_ALPHA_STATS_GENERAL = (
    f"{ENTIRE_DATABASE_EXPERIMENTS_ALPHA_STATS_DIR}/general.csv"
)
ENTIRE_DATABASE_EXPERIMENTS_ALPHA_STATS_QUERY = (
    f"{ENTIRE_DATABASE_EXPERIMENTS_ALPHA_STATS_DIR}/query.csv"
)
ENTIRE_DATABASE_EXPERIMENTS_ALPHA_PLOT_DIR = (
    f"{ENTIRE_DATABASE_EXPERIMENTS_ALPHA_DIR}/plots"
)
ENTIRE_DATABASE_EXPERIMENTS_QUERY_PERFORMANCE_DIR = (
    f"{ENTIRE_DATABASE_EXPERIMENTS_DIR}/query_performance"
)
ENTIRE_DATABASE_EXPERIMENTS_QUERY_PERFORMANCE_TIMES = (
    f"{ENTIRE_DATABASE_EXPERIMENTS_QUERY_PERFORMANCE_DIR}/times.csv"
)
ENTIRE_DATABASE_EXPERIMENTS_QUERY_PERFORMANCE_STATS = (
    f"{ENTIRE_DATABASE_EXPERIMENTS_QUERY_PERFORMANCE_DIR}/stats.csv"
)
ENTIRE_DATABASE_EXPERIMENTS_QUERY_PERFORMANCE_PLOT = (
    f"{ENTIRE_DATABASE_EXPERIMENTS_QUERY_PERFORMANCE_DIR}/histogram.png"
)
ENTIRE_DATABASE_EXPERIMENTS_AVERAGE_SIZE_REQUIRED_DIR = (
    f"{ENTIRE_DATABASE_EXPERIMENTS_DIR}/average-size-required"
)
ENTIRE_DATABASE_EXPERIMENTS_AVERAGE_SIZE_REQUIRED_SIZES = (
    f"{ENTIRE_DATABASE_EXPERIMENTS_AVERAGE_SIZE_REQUIRED_DIR}/sizes.csv"
)
ENTIRE_DATABASE_EXPERIMENTS_AVERAGE_SIZE_REQUIRED_STATS = (
    f"{ENTIRE_DATABASE_EXPERIMENTS_AVERAGE_SIZE_REQUIRED_DIR}/stats.csv"
)
ENTIRE_DATABASE_EXPERIMENTS_AVERAGE_SIZE_REQUIRED_PLOT = (
    f"{ENTIRE_DATABASE_EXPERIMENTS_AVERAGE_SIZE_REQUIRED_DIR}/box.png"
)

QUERY_DATABASE_PATH = "query_database.parquet"


def test_with_sample(spark):
    # ---------------------------------------------
    original_db_path = ORIGINAL_DB_PATH
    tfidf_output = "sample-spark-tfidf/output"
    pagerank_output = SAMPLE_PAGERANK_OUTPUT_PATH
    # ---------------------------------------------
    dataset_path = "data/sample_1e5.json"
    if not os.path.exists(dataset_path):
        create_sample_json(original_db_path, dataset_path, int(1e2))
    # ---------------------------------------------
    if not os.path.exists(tfidf_output):
        run_tfidf(spark, dataset_path, output_tfidf_path=tfidf_output)
    if not os.path.exists(pagerank_output):
        compute_pageranks(spark, dataset_path, pagerank_output, iterations=100)
        # compute_pagerank2(spark, dataset_path)
    nodes = get_top_n_ranked_nodes(pagerank_output, 10)
    print(nodes)
    print(sum_all_pageranks(pagerank_output))


def run_for_entire_database(spark: SparkSession):
    if not os.path.exists(ENTIRE_DATABASE_PAGERANK_OUTPUT_PATH):
        parse_db(ORIGINAL_DB_PATH, PARSED_DB_PATH)

        if not os.path.exists(ENTIRE_DATABASE_TFIDF_DIR):
            os.mkdir(ENTIRE_DATABASE_TFIDF_DIR)
        run_tfidf(
            spark,
            PARSED_DB_PATH,
            output_tfidf_path=ENTIRE_DATABASE_TFIDF_VECTORS_OUTPUT_PATH,
            pipeline_model_path=ENTIRE_DATABASE_TFIDF_PIPELINE_OUTPUT_PATH,
        )
        compute_pagerank2(spark, PARSED_DB_PATH, ENTIRE_DATABASE_PAGERANK_OUTPUT_PATH)
        nodes = get_top_n_ranked_nodes(ENTIRE_DATABASE_PAGERANK_OUTPUT_PATH, 10)
        print(nodes)
        print(sum_all_pageranks(ENTIRE_DATABASE_PAGERANK_OUTPUT_PATH))


def main():
    warnings.simplefilter("ignore")
    spark = get_spark()
    search_engine = SearchEngine()
    test_with_sample(spark)
    printSampleComplete()
    run_for_entire_database(spark)
    add_title_to_pagerank(
        ENTIRE_DATABASE_PAGERANK_OUTPUT_PATH,
        PARSED_DB_PATH,
        ENTIRE_DATABASE_PAGERANK_WITH_TITLES_OUTPUT_PATH,
    )
    # Experiments
    if not os.path.exists(QUERY_DATABASE_PATH):
        generate_query_database(
            spark, ENTIRE_DATABASE_PAGERANK_WITH_TITLES_OUTPUT_PATH, QUERY_DATABASE_PATH
        )
    required_directories = [
        ENTIRE_DATABASE_TFIDF_DIR,
        ENTIRE_DATABASE_EXPERIMENTS_DIR,
        ENTIRE_DATABASE_EXPERIMENTS_ALPHA_DIR,
        ENTIRE_DATABASE_EXPERIMENTS_ALPHA_PLOT_DIR,
        ENTIRE_DATABASE_EXPERIMENTS_ALPHA_STATS_DIR,
        ENTIRE_DATABASE_EXPERIMENTS_AVERAGE_SIZE_REQUIRED_DIR,
        ENTIRE_DATABASE_EXPERIMENTS_QUERY_PERFORMANCE_DIR,
    ]

    for d in required_directories:
        Path(d).mkdir(parents=True, exist_ok=True)

    alpha(
        search_engine,
        query_db_path=QUERY_DATABASE_PATH,
        result_size=100,
        values_path=ENTIRE_DATABASE_EXPERIMENTS_ALPHA_VALUES,
        stats_general_path=ENTIRE_DATABASE_EXPERIMENTS_ALPHA_STATS_GENERAL,
        stats_per_query_path=ENTIRE_DATABASE_EXPERIMENTS_ALPHA_STATS_QUERY,
    )

    load_parquet_to_elasticsearch(ENTIRE_DATABASE_PAGERANK_WITH_TITLES_OUTPUT_PATH)

    query_performance(
        search_engine=search_engine,
        query_db_path=QUERY_DATABASE_PATH,
        times_output_path=ENTIRE_DATABASE_EXPERIMENTS_QUERY_PERFORMANCE_TIMES,
        stats_output_path=ENTIRE_DATABASE_EXPERIMENTS_QUERY_PERFORMANCE_STATS,
        plot_output_path=ENTIRE_DATABASE_EXPERIMENTS_QUERY_PERFORMANCE_PLOT,
    )
    average_required_size(
        search_engine=search_engine,
        query_db_path=QUERY_DATABASE_PATH,
        sizes_output_path=ENTIRE_DATABASE_EXPERIMENTS_AVERAGE_SIZE_REQUIRED_SIZES,
        stats_output_path=ENTIRE_DATABASE_EXPERIMENTS_AVERAGE_SIZE_REQUIRED_STATS,
        plot_output_path=ENTIRE_DATABASE_EXPERIMENTS_AVERAGE_SIZE_REQUIRED_PLOT,
    )

    spark.stop()

    plot_sampled_citation_graph(PARSED_DB_PATH, 20, "graph.png")


if __name__ == "__main__":
    main()
