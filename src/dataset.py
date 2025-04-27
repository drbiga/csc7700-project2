import os
from itertools import cycle
import random
import json
import ijson

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from pyspark.sql.functions import explode, col, desc

from util import convert_decimals


def create_sample_parquet(
    input_path: str, output_path: str, sample_size: int = int(1e3)
) -> None:
    spark = (
        SparkSession.builder.appName("PageRankComputation")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.caseSensitive", "true")
        .getOrCreate()
    )
    df: DataFrame = spark.read.option("multiLine", True).json(input_path)
    df.limit(sample_size).write.parquet(output_path)


def create_sample_json(input_path: str, output_path: str, sample_size: int = int(1e3)):
    if os.path.exists(output_path):
        return
    with open(input_path, "r") as f:
        print("Starting json extraction")
        objects = ijson.items(f, "item")
        print("\tCreated objects variable")
        sample = []

        # probability_of_sampling = 1e-2
        for obj in objects:
            data = {"id": obj["id"], "title": obj["title"]}
            if "references" in obj:
                data["references"] = obj["references"]
            # if random.random() < probability_of_sampling:
            sample.append(convert_decimals(data))
            if len(sample) == sample_size:
                break
    with open(output_path, "w") as out:
        json.dump(sample, out)


def parse_db(input_path: str, output_path: str) -> None:
    if os.path.exists(output_path):
        return
    with open(input_path, "r") as f:
        print("Starting json extraction")
        objects = ijson.items(f, "item")
        print("\tCreated objects variable")
        sample = []

        for obj in objects:
            data = {"id": obj["id"], "title": obj["title"]}
            if "references" in obj:
                data["references"] = obj["references"]
            # if random.random() < probability_of_sampling:
            sample.append(convert_decimals(data))
    with open(output_path, "w") as out:
        json.dump(sample, out)


def compute_average_in_degree(input_json_path: str) -> float:
    """
    Computes the average number of incoming edges (in-degree) per node from a JSON file.

    Parameters:
    - input_json_path (str): Path to the input JSON file containing documents.

    Returns:
    - float: The average in-degree across all nodes.
    """
    # Initialize Spark session
    spark = SparkSession.builder.appName("ComputeAverageInDegree").getOrCreate()

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
    ).dropna()

    # Compute in-degree for each node
    in_degrees_df = edges_df.groupBy("dst").count()

    # Calculate total in-degree and number of unique nodes with in-degree
    total_in_degree = in_degrees_df.agg({"count": "sum"}).collect()[0][0]
    num_nodes_with_in_degree = in_degrees_df.count()

    # Compute average in-degree
    average_in_degree = (
        total_in_degree / num_nodes_with_in_degree
        if num_nodes_with_in_degree > 0
        else 0.0
    )

    # Stop the Spark session
    spark.stop()

    return average_in_degree


from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from typing import List, Tuple


def count_nodes_by_in_degree(input_json_path: str) -> List[Tuple[int, int]]:
    """
    Counts the number of nodes grouped by their in-degree from a JSON file.

    Parameters:
    - input_json_path (str): Path to the input JSON file containing documents.

    Returns:
    - List[Tuple[int, int]]: A list of tuples where each tuple contains:
        - in-degree (int): The number of incoming edges to a node.
        - count (int): The number of nodes with that in-degree.
    """
    # Initialize Spark session
    spark = SparkSession.builder.appName("CountNodesByInDegree").getOrCreate()

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
    ).dropna()

    # Compute in-degree for each node
    in_degrees_df = (
        edges_df.groupBy("dst").count().withColumnRenamed("count", "in_degree")
    )

    # Count the number of nodes for each in-degree
    in_degree_counts_df = (
        in_degrees_df.groupBy("in_degree")
        .count()
        .withColumnRenamed("count", "node_count")
    )

    # Collect the results as a list of tuples
    result = [
        (row["in_degree"], row["node_count"]) for row in in_degree_counts_df.collect()
    ]

    # Stop the Spark session
    spark.stop()

    return result


def main():
    dataset_path = "data/sample_1e3.json"
    print(compute_average_in_degree(dataset_path))
    print(count_nodes_by_in_degree(dataset_path))


if __name__ == "__main__":
    main()
