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
