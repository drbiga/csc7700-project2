import ijson
import json
import random
from decimal import Decimal

from pyspark.sql import SparkSession, functions as F, DataFrame
from pyspark.ml.feature import Tokenizer, StopWordsRemover


def convert_decimals(obj):
    if isinstance(obj, list):
        return [convert_decimals(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_decimals(value) for key, value in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj


def add_title_to_pagerank(
    input_pagerank_path: str, parsed_json_db_path: str, output_path: str
) -> None:
    """Connector between the output of the pagerank computation and the experiments.
    The pagerank algorithm does not include the titles, only ids and ranks, in the final file
    """

    spark = (
        SparkSession.builder.appName("App")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )
    df_pr = spark.read.parquet(input_pagerank_path)
    df_db = spark.read.option("multiLine", True).json(parsed_json_db_path)
    df = df_pr.join(df_db.drop("references"), on="id")
    df = df.orderBy(
        F.desc("rank"),
    )
    df.show(5)
    df.write.mode("overwrite").parquet(output_path)


def parse_text_for_matching(
    df: DataFrame, text_column_name: str, out_column_name: str, keep_array: bool = True
) -> DataFrame:
    df = df.withColumn(
        "filtered",
        F.regexp_replace(F.col(text_column_name), r"[^a-zA-Z0-9\- ]", ""),
    )
    tokenizer = Tokenizer(inputCol="filtered", outputCol="tokens")
    remover = StopWordsRemover(inputCol="tokens", outputCol=out_column_name)
    df = tokenizer.transform(df)
    df = remover.transform(df)
    if not keep_array:
        df = df.withColumn(out_column_name, F.array_join(F.col(out_column_name), " "))
    return df.drop("filtered", "tokens")


def filter_word_based(
    df: DataFrame, text_column_name: str, text_filter: str
) -> DataFrame:

    conditions = [
        F.col(text_column_name).contains(word) for word in text_filter.split()
    ]
    # Combine them
    combined_condition = conditions[0]
    for cond in conditions[1:]:
        combined_condition = combined_condition | cond

    # Filter
    df_filtered = df.filter(combined_condition)
    return df_filtered


def count_common_words(
    df: DataFrame,
    text_column_name: str,
    text_to_match: str,
    word_count_column_name: str,
) -> DataFrame:
    words_to_match = text_to_match.lower().split()

    # Broadcast the query words to all workers
    words_broadcast = F.array(*[F.lit(word) for word in words_to_match])

    # Step 1: Tokenize the text column (simple split by space)
    df = df.withColumn("text_words", F.split(F.lower(F.col(text_column_name)), r"\s+"))

    # Step 2: Find intersection between text_words and query_words
    df = df.withColumn(
        "common_words", F.array_intersect(F.col("text_words"), words_broadcast)
    )

    # Step 3: Count common words
    df = df.withColumn(word_count_column_name, F.size(F.col("common_words")))

    return df.drop("text_words", "common_words")
