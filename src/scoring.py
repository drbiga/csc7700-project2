import pandas as pd

from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import PipelineModel


def compute_score(
    spark: SparkSession,
    query: str,
    df_tfidf_vectors: DataFrame,
    model: PipelineModel,
    df_pageranks: DataFrame,
    id_field: str = "id",
    text_field: str = "title",
    top_n: int = 10,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> pd.DataFrame:
    """
    Computes cosine similarity scores between `query` and all documents in the TF-IDF store.
    Returns a DataFrame of (id, score) sorted by descending score, limited to `top_n`.

    Args:
        spark: SparkSession used to read models and data.
        query: The search string to score against the corpus.
        tfidf_df_path: Path where precomputed TF-IDF DataFrame is stored.
        pipeline_model_path: Path where the fitted pipeline is saved.
        id_field: Identifier column name.
        text_field: Original text column name (unused here).
        top_n: Number of top results to return.
    """
    # 1) Transform the query into TF-IDF using the saved pipeline
    # spark = get_spark()

    # Build TF-IDF features for the query
    query_df = spark.createDataFrame([(query,)], [text_field])
    q_vec = model.transform(query_df).select("tfidfFeatures").first()["tfidfFeatures"]

    # 2) Define a UDF for cosine similarity against the query vector
    def cosine_sim(v):
        dot = float(v.dot(q_vec))
        norm_d = float(v.norm(2))
        norm_q = float(q_vec.norm(2))
        return dot / (norm_d * norm_q) if norm_d and norm_q else 0.0

    cosine_udf = F.udf(cosine_sim, DoubleType())

    # Apply TF-IDF scoring
    tfidf_scored = df_tfidf_vectors.withColumn(
        "tfidf_score", cosine_udf(F.col("tfidfFeatures"))
    )

    # Load PageRank scores
    pr_df = df_pageranks.select(F.col(id_field), F.col("rank").alias("pr_score"))

    # Join and combine
    combined = (
        tfidf_scored.join(pr_df, on=id_field, how="left")
        .na.fill({"pr_score": 0.0})
        .withColumn("score", alpha * F.col("tfidf_score") + beta * F.col("pr_score"))
    )

    # Select and order
    out = (
        combined.select(F.col(id_field), F.col(text_field), F.col("score"))
        .orderBy(F.col("score").desc())
        .limit(top_n)
        .toPandas()
    )

    return out
