from tfidf_computation import run_tfidf, compute_score
from pagerank import compute_and_save_pagerank, get_top_n_ranked_nodes
from scoring import score

from dataset import create_sample_parquet, create_sample_json


def main():
    # run_tfidf("data/sample.json")
    # result = compute_score("network protocol")
    # print(result)
    # run_tfidf("data/dblp.v12.json")
    # ---------------------------------------------
    original_db_path = "data/dblp.v12.json"
    tfidf_output = "spark-tfidf/output"
    pagerank_output = "spark-pagerank/output.parquet"
    # ---------------------------------------------
    # dataset_path = "data/sample_1e3.parquet"
    # create_sample_parquet(original_db_path, dataset_path, int(1e3))
    # ---------------------------------------------
    dataset_path = "data/sample_1e1.json"
    # create_sample_json(original_db_path, dataset_path, int(1e1))
    # ---------------------------------------------
    run_tfidf(dataset_path, output_tfidf_path=tfidf_output)
    compute_and_save_pagerank(dataset_path, pagerank_output)
    # nodes = get_top_n_ranked_nodes(pagerank_output, 10)
    # print(nodes)


if __name__ == "__main__":
    main()
