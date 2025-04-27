from pyspark.sql import SparkSession, functions as F

from util import parse_text_for_matching


def generate_query_database(input_path: str, output_path: str) -> None:
    """Generates the query database that will be used in the experiments using the input file.
    The generated file will be stored in `output_path`

    Parameters
    ----------
        input_path: str - A JSON file containing the page rank results and the titles for every paper.
        output_path: str - An output path that will contain a JSON file with a list of query and id pairs
    """
    spark = (
        SparkSession.builder.appName("Query Database")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )
    N = 5  # number of words you want to sample

    df = spark.read.parquet(input_path)
    df = parse_text_for_matching(df, "title", "parsed_title", keep_array=True)
    df = df.withColumn("words_array", F.col("parsed_title"))
    df = df.withColumn("word_count", F.size(F.col("words_array"))).where(
        F.col("word_count") >= N
    )
    count = df.count()
    sample_size = int(0.1 * count)
    df_sample = df.orderBy(F.desc("rank")).limit(sample_size)
    # Shuffle the array randomly (keep their shuffled order, no sort)
    df_sample = df_sample.withColumn("shuffled_words", F.shuffle(F.col("words_array")))
    # Take first N words
    df_sample = df_sample.withColumn(
        "sampled_words", F.slice(F.col("shuffled_words"), 1, N)
    )
    # Join them back into a string
    df_sample = df_sample.withColumn(
        "sampled_text", F.array_join(F.col("sampled_words"), " ")
    ).cache()
    df_sample.select("id", "sampled_text").withColumnRenamed(
        "sampled_text", "query"
    ).show(truncate=False)
    df_sample.write.mode("overwrite").parquet(output_path)


def alpha() -> None:
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


def query_performance() -> None:
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


def average_required_size() -> None:
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


# =============================================
# There is something else we could try. Maybe not only doing the "average required size"
# but also doing something that resembles precision@K. Perhaps setting "good results" to
# be the titles that contain at least one word from the query. Then, we could look for
# some common words or something in the database to search for, I'm not sure. There will
# be some sort of arbitrary decision here.
