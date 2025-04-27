import os

from util import add_title_to_pagerank

from tfidf_computation import run_tfidf, compute_score

from pagerank import (
    get_top_n_ranked_nodes,
    compute_pageranks,
    sum_all_pageranks,
    compute_pagerank2,
)
from scoring import score

from dataset import create_sample_parquet, create_sample_json, parse_db

from experiments import generate_query_database


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

QUERY_DATABASE_PATH = "query_database.parquet"


def test_with_sample():
    # ---------------------------------------------
    original_db_path = ORIGINAL_DB_PATH
    tfidf_output = "sample-spark-tfidf/output"
    pagerank_output = SAMPLE_PAGERANK_OUTPUT_PATH
    # ---------------------------------------------
    dataset_path = "data/sample_1e5.json"
    # create_sample_json(original_db_path, dataset_path, int(1e2))
    # ---------------------------------------------
    # run_tfidf(dataset_path, output_tfidf_path=tfidf_output)
    # compute_pageranks(dataset_path, pagerank_output, iterations=100)
    # compute_pagerank2(dataset_path)
    # nodes = get_top_n_ranked_nodes(pagerank_output, 10)
    # print(nodes)
    print(sum_all_pageranks(pagerank_output))


def run_for_entire_database():
    parse_db(ORIGINAL_DB_PATH, PARSED_DB_PATH)

    if not os.path.exists(ENTIRE_DATABASE_TFIDF_DIR):
        os.mkdir(ENTIRE_DATABASE_TFIDF_DIR)
    run_tfidf(
        PARSED_DB_PATH,
        output_tfidf_path=ENTIRE_DATABASE_TFIDF_VECTORS_OUTPUT_PATH,
        pipeline_model_path=ENTIRE_DATABASE_TFIDF_PIPELINE_OUTPUT_PATH,
    )
    compute_pagerank2(PARSED_DB_PATH, ENTIRE_DATABASE_PAGERANK_OUTPUT_PATH)
    nodes = get_top_n_ranked_nodes(ENTIRE_DATABASE_PAGERANK_OUTPUT_PATH, 10)
    print(nodes)
    print(sum_all_pageranks(ENTIRE_DATABASE_PAGERANK_OUTPUT_PATH))


def main():
    # test_with_sample()
    # run_for_entire_database()
    # add_title_to_pagerank(
    #     ENTIRE_DATABASE_PAGERANK_OUTPUT_PATH,
    #     PARSED_DB_PATH,
    #     ENTIRE_DATABASE_PAGERANK_WITH_TITLES_OUTPUT_PATH,
    # )
    generate_query_database(
        ENTIRE_DATABASE_PAGERANK_WITH_TITLES_OUTPUT_PATH, QUERY_DATABASE_PATH
    )


if __name__ == "__main__":
    main()
