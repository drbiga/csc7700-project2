import random
import json
import ijson

from pyspark.sql import SparkSession, DataFrame
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
    with open(input_path, "r") as f:
        print("Starting json extraction")
        objects = ijson.items(f, "item")
        print("\tCreated objects variable")
        sample = []

        probability_of_sampling = 1e-2
        for obj in objects:
            data = {"id": obj["id"], "title": obj["title"]}
            if "references" in obj:
                data["references"] = obj["references"]
            if random.random() < probability_of_sampling:
                sample.append(convert_decimals(data))
                if len(sample) == sample_size:
                    break
    with open(output_path, "w") as out:
        json.dump(sample, out)
