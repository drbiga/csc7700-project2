import os

from tfidf_computation import run_tfidf, compute_score
from pagerank import (
    get_top_n_ranked_nodes,
    compute_pageranks,
    sum_all_pageranks,
    compute_pagerank2,
)
from scoring import score

from dataset import create_sample_parquet, create_sample_json, parse_db


def test_with_sample():
    # ---------------------------------------------
    original_db_path = "data/dblp.v12.json"
    tfidf_output = "sample-spark-tfidf/output"
    pagerank_output = "sample-spark-pagerank/output.parquet"
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
    original_db_path = "data/dblp.v12.json"
    parsed_database_path = "data/parsed.json"
    pagerank_output = "sample-spark-pagerank/output.parquet"

    # parse_db(original_db_path, parsed_database_path)
    tfidf_output_dir = "entire-database-spark-tfidf"
    if not os.path.exists(tfidf_output_dir):
        os.mkdir(tfidf_output_dir)
    # run_tfidf(
    #     parsed_database_path,
    #     output_tfidf_path=f"{tfidf_output_dir}/tfidf.parquet",
    #     pipeline_model_path=f"{tfidf_output_dir}/pipeline.parquet",
    # )
    # pagerank_output_dir = "entire-database-spark-pageranks"
    # pagerank_output = f"{pagerank_output_dir}/ranks.parquet"
    # if not os.path.exists(pagerank_output_dir):
    #     os.mkdir(pagerank_output_dir)
    # # compute_pageranks(parsed_database_path, output_parquet_path=pagerank_output)
    compute_pagerank2(parsed_database_path, pagerank_output)
    nodes = get_top_n_ranked_nodes(pagerank_output, 10)
    print(nodes)
    print(sum_all_pageranks(pagerank_output))


def main():
    # test_with_sample()
    run_for_entire_database()


if __name__ == "__main__":
    main()
