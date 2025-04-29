import os

from datetime import datetime

import itertools

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import udf, col
from pyspark.sql.types import DoubleType

from search_engine import SearchEngine
from evaluation import difference_between_result_sets
from tfidf_computation import compute_score
from scoring import compute_score as score_matheus

from util import parse_text_for_matching, filter_word_based, count_common_words


def generate_query_database(
    spark: SparkSession, input_path: str, output_path: str
) -> None:
    """Generates the query database that will be used in the experiments using the input file.
    The generated file will be stored in `output_path`

    Parameters
    ----------
        input_path: str - A JSON file containing the page rank results and the titles for every paper.
        output_path: str - An output path that will contain a JSON file with a list of query and id pairs
    """
    N_WORDS_PER_QUERY = 5  # number of words you want to sample
    N_QUERIES = 1000

    df = spark.read.parquet(input_path)
    df = parse_text_for_matching(df, "title", "parsed_title", keep_array=True)
    df = df.withColumn("words_array", F.col("parsed_title"))
    df = df.withColumn("word_count", F.size(F.col("words_array"))).where(
        F.col("word_count") >= N_WORDS_PER_QUERY
    )
    count = df.count()
    highest_ranking_percentage = int(0.1 * count)
    df_sample = df.orderBy(F.desc("rank")).limit(highest_ranking_percentage)
    df_sample = df_sample.orderBy(F.rand()).limit(N_QUERIES)
    # Shuffle the array randomly (keep their shuffled order, no sort)
    df_sample = df_sample.withColumn("shuffled_words", F.shuffle(F.col("words_array")))
    # Take first N words
    df_sample = df_sample.withColumn(
        "sampled_words", F.slice(F.col("shuffled_words"), 1, N_WORDS_PER_QUERY)
    )
    # Join them back into a string
    df_sample = df_sample.withColumn(
        "sampled_text", F.array_join(F.col("sampled_words"), " ")
    ).cache()
    query_df = df_sample.select(
        F.col("id").alias("id"), F.col("sampled_text").alias("query")
    )
    query_df.show(truncate=False)
    query_df.write.mode("overwrite").parquet(output_path)


def alpha(
    search_engine: SearchEngine,
    query_db_path: str,
    result_size: int,
    values_path: str = "entire-database-spark-experiments/alpha/results.csv",
    stats_general_path: str = "entire-database-spark-experiments/alpha/stats/general.csv",
    stats_per_query_path: str = "entire-database-spark-experiments/alpha/stats/query.csv",
    plots_path: str = "entire-database-spark-experiments/alpha/plots",
) -> None:
    """Performs the evaluation of various different scenarios for alpha, which
    is the weight given to TFIDF relative to the final score. Alpha values are
    constrained between 0 and 1.

    The evaluation will use the query database created according to the README
    file in order to generate result sets.

    Result set sizes will be set to 100 (arbitrary).

    Steps:
    1. Generate a range of values (alpha) between 0 and 1
    2. Get all queries in the query database
    3. For every query
    3.1 For all alpha values in the range
    3.1.1 Evaluate the score for all documents
    3.1.2 Sample N=100 highest scoring ones
    3.1.3 Store the result set indexed by query and alpha
    3.2 Compute result set difference for all result set pairs of the current query
    3.3 Store the difference indexed by query and pairs of alphas
    4. Plot (scatter, line, whatever) the alphas vs the differences for each query - maybe one plot for each query or even one plot with several hues
    """
    df_queries = pd.read_parquet(query_db_path)
    query_count = df_queries.count()
    print("Query count:", query_count)

    # alpha_values = np.linspace(0, 1, 100)
    alpha_values = np.geomspace(1e-6, 1, 100)
    rows = []
    counter = 0
    # 2) for each query, drive all α locally
    for _, row in df_queries.iterrows():
        # compute top-N at each α
        counter += 1
        print("Iteration", counter)
        score_dicts = {}
        for a in alpha_values:
            scores = search_engine.get_doc_scores_for_query(
                row["query"], alpha=a, result_size=result_size
            )
            # build {id → score} lookup
            score_dicts[a] = {
                score["_source"]["id"]: score["_score"] for score in scores
            }

        # pairwise differences
        for a1, a2 in itertools.combinations(alpha_values, 2):
            diff = difference_between_result_sets(
                score_dicts[a1], score_dicts[a2], a1, a2
            )
            rows.append(
                {"query_id": row["id"], "alpha1": a1, "alpha2": a2, "difference": diff}
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(values_path, index=False)

    agg_functions = ["mean", "median", "std"]
    out_df.agg(
        {
            "difference": agg_functions,
        }
    ).to_csv(stats_general_path, index=False)
    df = out_df.groupby("query_id").agg({"difference": agg_functions}).reset_index()
    df.columns = df.columns.map("_".join)
    df.to_csv(stats_per_query_path, index=False)

    out_df["alpha1"] = np.log(out_df["alpha1"].apply(lambda a: round(a, 2)))
    out_df["alpha2"] = np.log(out_df["alpha2"].apply(lambda a: round(a, 2)))
    # Pivot the DataFrame
    heatmap_data = out_df.drop("query_id", axis=1).pivot_table(
        index="alpha1", columns="alpha2", values="difference", aggfunc="mean"
    )

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap="viridis")
    plt.title("Heatmap of y by x1 and x2")
    plt.xlabel("α₁")
    plt.ylabel("α₂")
    plt.savefig(f"{plots_path}/heatmap.png")
    plt.close()

    # 4) per-query scatter
    for qid in out_df["query_id"].unique():
        sub = out_df[out_df["query_id"] == qid]
        plt.figure()
        plt.scatter(
            sub["alpha1"],
            sub["alpha2"],
            s=sub["difference"] * 100,  # scale for visibility
        )
        plt.title(f"Alpha sweep diffs for query {qid}")
        plt.xlabel("α₁")
        plt.ylabel("α₂")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, f"alpha_sweep_{qid}.png"))
        plt.close()


def query_performance(
    search_engine: SearchEngine,
    query_db_path: str,
    times_output_path: str,
    stats_output_path: str,
    plot_output_path: str,
) -> None:
    """Performs the evaluation of query performance time.

    The evaluation will use the query database created according to the README
    file in order to generate query processing times.

    Steps:
    1. Get all queries in the query database
    2. For every query
    2.1 Record the time as start time
    2.2 Run the scoring function
    2.3 Record the time as end time
    2.4 Store the result in a list
    3. Compute average, median, and standard deviation of scoring time
    4. Plot histogram of scoring time
    """
    # df_queries = spark.read.parquet(query_db_path)
    df_queries = pd.read_parquet(query_db_path)
    query_count = df_queries.count()
    print("Query count:", query_count)
    counter = 0
    ids_list = []
    query_list = []
    query_times_seconds_list = []
    for _, row in df_queries[["id", "query"]].iterrows():
        counter += 1
        print("Running query", counter)
        ts_start = datetime.now()
        search_engine.get_doc_scores_for_query(row["query"], 0.5)
        ts_end = datetime.now()
        elapsed_time_seconds = (ts_end - ts_start).total_seconds()
        ids_list.append(row["id"])
        query_list.append(row["query"])
        query_times_seconds_list.append(elapsed_time_seconds)
    df = pd.DataFrame(
        {"id": ids_list, "query": query_list, "time": query_times_seconds_list}
    )
    df.to_csv(times_output_path, index=False)
    df["time"].agg(["mean", "median", "std"]).to_csv(stats_output_path, index=False)
    fig, ax = plt.subplots(1, 1)
    sns.histplot(df, x="time", ax=ax)
    fig.savefig(plot_output_path)
    plt.close(fig)


def average_required_size(
    search_engine: SearchEngine,
    query_db_path: str,
    sizes_output_path: str,
    stats_output_path: str,
    plot_output_path: str,
) -> None:
    """Performs the evaluation of average required result set size.

    The evaluation will use the query database created according to the README
    file in order to generate result sets.

    Steps:
    1. Get all queries in the query database
    2. For every query
    2.1 Run the scoring function
    2.2 For every result IN DESCENDING ORDER OF SCORES
    2.2.1 Test if the current document was the one from which the query was generated
    2.2.2 Store the index of the result in a list
    3. Compute average, median, and standard deviation of index sizes?
    4. Plot histogram of index sizes?
    """
    df_queries = pd.read_parquet(query_db_path)
    counter = 0
    alphas_list = []
    ids_list = []
    queries_list = []
    sizes_list = []
    for alpha in np.linspace(0, 1, 10):
        for _, row in df_queries[["id", "query"]].iterrows():
            counter += 1
            print("Running query", counter)

            rank = search_engine.get_rank(row["id"], row["query"], alpha=0.5)

            alphas_list.append(alpha)
            ids_list.append(row["id"])
            queries_list.append(row["query"])
            sizes_list.append(rank)
    df = pd.DataFrame(
        {
            "alpha": alphas_list,
            "id": ids_list,
            "query": queries_list,
            "size": sizes_list,
        }
    )
    df.to_csv(sizes_output_path, index=False)
    df = pd.read_csv(sizes_output_path)
    df["alpha"] = df["alpha"].apply(lambda a: round(a, 2))
    df_agg = df.groupby("alpha")[["size"]].agg(["mean", "median", "std"]).reset_index()
    df_agg.columns = df_agg.columns.map("_".join)
    df_agg.to_csv(stats_output_path, index=False)
    fig, ax = plt.subplots(1, 1)
    sns.boxplot(df, x="alpha", y="size", ax=ax, showfliers=False)
    plt.xticks(rotation=90)
    fig.savefig(plot_output_path)
    plt.close(fig)
    df["eq1"] = df["size"].apply(lambda s: s == 1)
    print(df.groupby("eq1").count())
