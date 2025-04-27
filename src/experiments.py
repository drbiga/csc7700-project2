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
