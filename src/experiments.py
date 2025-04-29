import os

from datetime import datetime

import itertools

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import udf, col
from pyspark.sql.types import DoubleType

from evaluation import difference_between_result_sets
from util import parse_text_for_matching
from tfidf_computation import compute_score
from scoring import compute_score as score_matheus


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
    spark: SparkSession,
    query_db_path: str,
    tfidf_path: str = "spark/tfidf",
    pagerank_path: str = "entire-database-spark-pageranks",
    pipeline_model_path: str = "spark/pipeline_model",
    result_size: int = 100,
    alpha_values: list = [i / 10.0 for i in range(11)],
    sample_frac: float = 0.1,
    sample_limit: int = 100,
    output_csv: str = "entire-database-spark-experiments/alpha_results.csv",
    output_dir: str = "entire-database-spark-experiments/alpha_plots",
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
    from pyspark.sql.functions import lit

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load and sample queries
    queries = (
        spark.read.parquet(query_db_path)
        .select("id", "query")
        .sample(False, sample_frac, seed=42)
        .limit(sample_limit)
        .collect()
    )
    queries = [(r["id"], r["query"]) for r in queries]

    # Pre-load data
    model = PipelineModel.load(pipeline_model_path)
    df_tfidf = spark.read.parquet(tfidf_path)
    df_pr = spark.read.parquet(pagerank_path).select("id", F.col("rank").alias("pr_score"))

    rows = []
    for qid, query in queries:
        # Compute query vector once
        query_df = spark.createDataFrame([(query,)], ["title"])
        q_vec = model.transform(query_df).select("tfidfFeatures").first()["tfidfFeatures"]
        q_norm = float(q_vec.norm(2))

        # Broadcast vector and norm
        sc = spark.sparkContext
        q_vec_b = sc.broadcast(q_vec)
        q_norm_b = sc.broadcast(q_norm)

        # Define UDF without embedding large closure
        def cosine_sim(v):
            qb = q_vec_b.value
            dot = float(v.dot(qb))
            denom = float(v.norm(2)) * q_norm_b.value
            return dot / denom if denom != 0.0 else 0.0
        cosine_udf = udf(cosine_sim, DoubleType())

        # Compute TF-IDF similarity once
        df_scores = (
            df_tfidf.withColumn("tfidf_score", cosine_udf(col("tfidfFeatures")))
            .join(df_pr, on="id", how="left")
            .na.fill({"pr_score": 0.0})
        )

        # Sweep alpha values
        for a in alpha_values:
            combined = df_scores.withColumn("score", lit(a) * col("tfidf_score") + lit(1.0 - a) * col("pr_score"))
            topn = (
                combined.select("id", "score")
                .orderBy(col("score").desc())
                .limit(result_size)
                .toPandas()
            )
            score_dict = dict(zip(topn['id'], topn['score']))

            # Compare against other α later via pairwise diffs
            # Store temporarily
            rows.extend([
                {'query_id': qid, 'alpha': a, 'scores': score_dict}
            ])

    # Now compute pairwise differences
    diff_rows = []
    # Organize by query
    from collections import defaultdict
    by_query = defaultdict(dict)
    for rec in rows:
        by_query[rec['query_id']][rec['alpha']] = rec['scores']

    for qid, scores_map in by_query.items():
        for a1, a2 in itertools.combinations(alpha_values, 2):
            diff = difference_between_result_sets(scores_map[a1], scores_map[a2], a1, a2)
            diff_rows.append({'query_id': qid, 'alpha1': a1, 'alpha2': a2, 'difference': diff})

    # Save CSV
    out_df = pd.DataFrame(diff_rows)
    out_df.to_csv(output_csv, index=False)

    # Generate scatter plots per query
    for qid in out_df['query_id'].unique():
        sub = out_df[out_df['query_id'] == qid]
        plt.figure()
        plt.scatter(sub['alpha1'], sub['alpha2'], s=sub['difference'] * 100)
        plt.title(f"Alpha sweep diffs for query {qid}")
        plt.xlabel("α₁")
        plt.ylabel("α₂")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"alpha_sweep_{qid}.png"))
        plt.close()



def query_performance(
    spark: SparkSession,
    query_db_path: str,
    tfidf_path: str,
    pipeline_model_path: str,
    pagerank_path: str,
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
    df_queries = spark.read.parquet(query_db_path)
    query_count = df_queries.count()
    print("Query count:", query_count)
    query_times_seconds_list = []
    model = PipelineModel.load(pipeline_model_path)
    df_tfidf_vectors = spark.read.parquet(tfidf_path)
    df_pageranks = spark.read.parquet(pagerank_path)
    counter = 0
    for query in df_queries.select("query").toLocalIterator():
        counter += 1
        if counter > min(5 / 0.5, query_count):
            break
        print("Running query", counter)
        ts_start = datetime.now()

        conditions = [F.col("title").contains(word) for word in query["query"].split()]
        # Combine them
        combined_condition = conditions[0]
        for cond in conditions[1:]:
            combined_condition = combined_condition | cond

        # Filter
        df_tfidf_vectors_filtered = df_tfidf_vectors.filter(combined_condition)
        score_matheus(
            spark,
            query["query"],
            df_tfidf_vectors_filtered,
            model,
            df_pageranks,
            alpha=0.5,  # the values for alpha and beta do not matter too much here,
            beta=0.5,  # as we are just collecting execution time data
        )
        ts_end = datetime.now()
        elapsed_time_seconds = (ts_end - ts_start).total_seconds()
        query_times_seconds_list.append(elapsed_time_seconds)
    df = pd.DataFrame({"time": query_times_seconds_list})
    os.makedirs(os.path.dirname(times_output_path), exist_ok=True)
    df.to_csv(times_output_path)
    df.agg(["mean", "median", "std"]).to_csv(stats_output_path)
    fig, ax = plt.subplots(1, 1)
    sns.histplot(df, x="time", ax=ax)
    fig.savefig(plot_output_path)
    plt.close(fig)


def average_required_size(
    spark: SparkSession,
    query_db_path: str,
    tfidf_vectors_path: str,
    pipeline_model_path: str,
    pageranks_path,
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
    df_queries = spark.read.parquet(query_db_path)
    query_count = df_queries.count()
    query_times_seconds_list = []
    model = PipelineModel.load(pipeline_model_path)
    df_tfidf_vectors = spark.read.parquet(tfidf_vectors_path)
    model = PipelineModel.load(pipeline_model_path)
    df_pageranks = spark.read.parquet(pageranks_path)
    counter = 0
    for query in df_queries.select("query").toLocalIterator():
        counter += 1
        if counter > min(5 / 0.5, query_count):
            break
        print("Running query", counter)
        ts_start = datetime.now()
        score_matheus(
            spark,
            query["query"],
            df_tfidf_vectors,
            model,
            df_pageranks,
            alpha=0.5,  # the values for alpha and beta do not matter too much here,
            beta=0.5,  # as we are just collecting execution time data
        )
        ts_end = datetime.now()
        elapsed_time_seconds = (ts_end - ts_start).total_seconds()
        query_times_seconds_list.append(elapsed_time_seconds)


# =============================================
# There is something else we could try. Maybe not only doing the "average required size"
# but also doing something that resembles precision@K. Perhaps setting "good results" to
# be the titles that contain at least one word from the query. Then, we could look for
# some common words or something in the database to search for, I'm not sure. There will
# be some sort of arbitrary decision here.
