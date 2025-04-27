import os
import shutil

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, lit, collect_list, desc, size
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from typing import Optional


def compute_pagerank2(input_path: str, output_path: str, tolerance: float = 1e-4):
    # Initialize Spark
    spark = (
        SparkSession.builder.appName("PageRank Improved")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )

    # Load JSON data
    df = spark.read.option("multiLine", True).json(input_path)

    # Create edges DataFrame (src, dst)
    edges = df.select(
        col("id").alias("src"), explode(col("references")).alias("dst")
    ).cache()
    edges.count()

    # List of all unique nodes
    nodes = (
        df.select(col("id"))
        .union(edges.select(col("dst").alias("id")))
        .distinct()
        .cache()
    )

    num_nodes = nodes.count()

    # Initialize ranks: all nodes start with equal rank summing to 1
    ranks = nodes.withColumn("rank", lit(1.0 / num_nodes)).cache()
    ranks.count()

    # Count outgoing links
    out_degree = edges.groupBy("src").count().cache()
    out_degree.count()

    # Iterative PageRank parameters
    num_iterations = 50
    damping_factor = 0.85

    tmpdir = "spark-tmp"
    dead_ends = nodes.join(out_degree, nodes.id == out_degree.src, "left_anti")

    ranks.write.mode("overwrite").parquet(tmpdir)
    for i in range(num_iterations):
        # Join edges with ranks
        print("Iteration", i + 1)
        ranks = spark.read.parquet(tmpdir)
        joined = edges.join(ranks, edges.src == ranks.id, "left").select(
            edges.src, edges.dst, ranks.rank
        )

        # Join with out_degree to calculate contribution
        joined = joined.join(out_degree, "src", "left")

        # Each node contributes rank / out_degree to each of its outgoing neighbors
        contributions = joined.withColumn(
            "contribution", col("rank") / col("count")
        ).select("dst", "contribution")

        # Sum of contributions received by each node
        contribs_per_node = contributions.groupBy("dst").agg(
            F.sum("contribution").alias("sum_contribution")
        )

        # Handle dead ends: total rank lost due to nodes with no outgoing edges
        dead_end_rank = (
            ranks.join(dead_ends, "id", "inner")
            .agg(F.sum("rank").alias("dead_end_rank"))
            .collect()[0]["dead_end_rank"]
        )

        if dead_end_rank is None:
            dead_end_rank = 0.0

        # Distribute dead_end_rank evenly across all nodes
        dead_end_contrib = dead_end_rank / num_nodes

        ranks = (
            nodes.join(contribs_per_node, nodes.id == contribs_per_node.dst, "left")
            .withColumn(
                "sum_contribution", F.coalesce(col("sum_contribution"), F.lit(0.0))
            )
            .select(
                nodes.id,
                (
                    (1.0 - damping_factor) / num_nodes
                    + damping_factor * (col("sum_contribution") + dead_end_contrib)
                ).alias("rank"),
            )
        )

        previous_ranks = spark.read.parquet(tmpdir)
        total_rank = ranks.agg(F.sum("rank").alias("total")).collect()[0]["total"]
        err = (
            ranks.join(previous_ranks, "id")
            .select(F.abs(ranks.rank - previous_ranks.rank).alias("diff"))
            .agg(F.sum("diff"))
            .collect()[0][0]
        )
        print("\tError:", err)
        print("\tTotal rank:", total_rank)
        ranks.write.mode("overwrite").parquet(tmpdir)
        if err < tolerance:
            print("\tALGORITHM CONVERGED IN", i + 1, "ITERATIONS")
            break

    ranks = spark.read.parquet(tmpdir)
    total_rank = ranks.agg(F.sum("rank").alias("total")).collect()[0]["total"]
    ranks = ranks.withColumn("rank", col("rank") / total_rank)
    ranks.write.mode("overwrite").parquet(output_path)
    spark.stop()


# Do NOT use
# This does not work
# Use V2 above
def compute_pageranks(
    input_json_path: str,
    output_parquet_path: str,
    iterations: int = 10,
    tolerance: float = 1e-4,
    damping_factor: float = 0.85,
):
    """
    Computes PageRank scores for documents defined in a JSON file.

    Parameters:
    - input_json_path (str): Path to the input JSON file containing documents.
    - output_parquet_path (str): Path where the output Parquet file will be saved.
    - iterations (int): Number of iterations for the PageRank algorithm. Default is 10.
    - damping_factor (float): Damping factor for the PageRank algorithm. Default is 0.85.
    """
    # Initialize Spark session
    spark = (
        SparkSession.builder.appName("ComputePageRanks")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.autoBroadcastJoinThreshold", -1)
        .getOrCreate()
    )

    # Define the schema for the input JSON
    schema = StructType(
        [
            StructField("id", StringType(), nullable=False),
            StructField("title", StringType(), nullable=True),
            StructField("references", ArrayType(StringType()), nullable=True),
        ]
    )

    # Read the JSON file into a DataFrame
    docs_df = spark.read.schema(schema).json(input_json_path)
    # Create edges: from 'id' to each of its 'references'
    edges_df = docs_df.select(
        col("id").alias("src"), explode(col("references")).alias("dst")
    ).cache()
    edges_df.count()

    out_degrees_df = (
        edges_df.groupBy("src")
        .agg(F.countDistinct("dst").alias("out_degree"))
        .select(col("src").alias("id"), col("out_degree"))
    )
    in_degrees_df = (
        edges_df.groupBy("dst")
        .agg(F.countDistinct("src").alias("in_degree"))
        .select(col("dst").alias("id"), col("in_degree"))
    )
    ranks_df = (
        out_degrees_df.join(in_degrees_df, on="id", how="outer")
        .fillna({"out_degree": 0, "in_degree": 0})
        .withColumn("rank", lit(1.0))
    )

    ranks_df.show(5)

    # Get the list of all node IDs
    all_nodes = ranks_df.select("id").cache()
    num_nodes = all_nodes.count()

    ranks_df = ranks_df.withColumn("rank", col("rank") / num_nodes)

    tmpdir = "spark-tmp"
    ranks_df.write.mode("overwrite").parquet(tmpdir)

    # Iteratively compute PageRank
    num_iters = 0
    err = 2 * tolerance  # just a dummy value to pass the first check and begin the loop
    while not (num_iters > iterations or err < tolerance):
        num_iters += 1

        ranks_df = spark.read.parquet(tmpdir)
        joined_df = ranks_df.select(
            col("id").alias("id_j"),
            col("rank").alias("rank_j"),
            col("out_degree").alias("out_degree_j"),
            col("in_degree").alias("in_degree_j"),
        ).join(edges_df, col("id_j") == edges_df.dst, how="left")
        joined_df = joined_df.join(
            ranks_df.select(
                col("id").alias("id_i"),
                col("rank").alias("rank_i"),
                col("out_degree").alias("out_degree_i"),
                col("in_degree").alias("in_degree_i"),
            ),
            joined_df.src == col("id_i"),
            how="left",
        )
        # At this point, we should have a table with all nodes (j) and all neighbor nodes (i) that point to those nodes
        # So, we can compute the damped rank
        ranks2_df = joined_df.groupBy("id_j", "out_degree_j", "in_degree_j").agg(
            F.sum(col("rank_i") / col("out_degree_i")).alias("rank_j")
        )
        ranks2_df = ranks2_df.withColumn(
            "rank_j",
            F.when(col("in_degree_j") == 0, 0).otherwise(
                damping_factor * col("rank_j")
            ),
        )
        ranks_df = ranks2_df.select(
            col("id_j").alias("id"),
            col("rank_j").alias("rank"),
            col("in_degree_j").alias("in_degree"),
            col("out_degree_j").alias("out_degree"),
        )
        sum_current_ranks = (
            ranks_df.select(F.sum(col("rank")).alias("rank")).collect()[0].rank
        )
        ranks_df = ranks_df.withColumn(
            "rank", col("rank") + lit((1 - sum_current_ranks) / num_nodes)
        )

        previous_ranks = spark.read.parquet(tmpdir)
        ranks_df.show(5)
        previous_ranks.show(5)
        sum_previous_ranks = (
            previous_ranks.select(F.sum("rank").alias("total_rank"))
            .collect()[0]
            .total_rank
        )
        ranks_df.write.mode("overwrite").parquet(tmpdir)
        err = abs(sum_current_ranks - sum_previous_ranks)
        print("\tCurrent pagerank error:", err)

    if err < tolerance:
        print("PageRank algorithm CONVERGED")
    else:
        print(f"PageRank algorithm finished running and DID NOT CONVERGE")

    ranks_df = spark.read.parquet(tmpdir)
    ranks_df.write.mode("overwrite").parquet(output_parquet_path)
    shutil.rmtree(tmpdir)
    # Stop the Spark session
    spark.stop()


def get_top_n_ranked_nodes(pagerank_parquet_path, n):
    # Written by ChatGPT
    spark = (
        SparkSession.builder.config("spark.sql.caseSensitive", "true")
        .appName("TopPageRankRetriever")
        .getOrCreate()
    )

    pr_df = spark.read.parquet(pagerank_parquet_path)

    top_n = (
        pr_df.orderBy(desc("rank"))
        .limit(n)
        .select("id", "rank")
        .rdd.map(lambda row: (row["id"], row["rank"]))
        .collect()
    )

    spark.stop()
    return top_n


def sum_all_pageranks(pagerank_parquet_path: str) -> float:
    spark = (
        SparkSession.builder.config("spark.sql.caseSensitive", "true")
        .appName("TopPageRankRetriever")
        .getOrCreate()
    )
    df = spark.read.parquet(pagerank_parquet_path)
    df.show(5)
    result = df.select(F.sum(col("rank"))).collect()

    result = result if result is not None else 0
    return result
