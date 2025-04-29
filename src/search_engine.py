import pandas as pd
from elasticsearch import Elasticsearch, helpers

ELASTICSEARCH_INDEX_NAME = "papers"


def load_parquet_to_elasticsearch(parquet_path):
    """
    Load a Parquet file into a local ElasticSearch index.

    Args:
        parquet_path (str): Path to the .parquet file.
        index_name (str): Target index name in ElasticSearch.
        es_host (str): URL of ElasticSearch instance (default localhost).
    """
    # Connect to ElasticSearch
    es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", "3AQp0+4gk+pQubOM_8AG"),
        ca_certs="./http_ca.crt",
    )

    # Read Parquet file into DataFrame
    df = pd.read_parquet(parquet_path)

    # Convert DataFrame into dictionary records
    records = df.to_dict(orient="records")

    # Prepare actions for bulk API
    actions = [
        {"_index": ELASTICSEARCH_INDEX_NAME, "_source": record} for record in records
    ]

    # Bulk insert into ElasticSearch
    helpers.bulk(es, actions)

    print(
        f"Successfully inserted {len(actions)} records into index '{ELASTICSEARCH_INDEX_NAME}'."
    )


class SearchEngine:
    def __init__(self):
        self.username = "elastic"
        self.password = "password"
        self.index_name = "papers"
        self.es = Elasticsearch(
            "https://localhost:9200",
            basic_auth=("elastic", "3AQp0+4gk+pQubOM_8AG"),
            ca_certs="./http_ca.crt",
        )

    def get_doc_scores_for_query(
        self, query: str, alpha: float, result_size: int = 200
    ) -> float:
        # Define query (same structure as JSON)
        query = {
            "size": result_size,
            "query": {
                "script_score": {
                    "query": {"match": {"title": query}},
                    "script": {
                        "source": f"{alpha} * _score + {1 - alpha} * doc['rank'].value"
                    },
                }
            },
        }
        # Search
        response = self.es.search(index=ELASTICSEARCH_INDEX_NAME, body=query)

        return response["hits"]["hits"]

    def get_rank(
        self,
        _id: str,
        query: str,
        alpha: float,
        batch_size: int = 1000,
        scroll_time: str = "2m",
    ) -> int:
        """
        Returns the rank (1-based index) of a document with the given ID for a search query.
        Uses the scroll API to go beyond Elasticsearch's size limit.

        Parameters:
            id (str): Document ID to search for.
            query (str): The search query string.
            index (str): The name of the Elasticsearch index to search.
            batch_size (int): Number of documents to fetch per scroll request.
            scroll_time (str): Scroll context lifetime (e.g., '2m').

        Returns:
            int: Rank of the document (1-based), or -1 if not found.
        """
        from elasticsearch import Elasticsearch

        es = self.es  # Elasticsearch client

        # Initial search with scroll
        response = es.search(
            index=ELASTICSEARCH_INDEX_NAME,
            body={
                "query": {
                    "script_score": {
                        "query": {"match": {"title": query}},
                        "script": {
                            "source": f"{alpha} * _score + {1 - alpha} * doc['rank'].value"
                        },
                    }
                },
            },
            scroll=scroll_time,
            size=batch_size,
        )

        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
        rank = 1

        while hits:
            for hit in hits:
                if hit["_source"]["id"] == _id:
                    # Clear scroll context before returning
                    es.clear_scroll(scroll_id=scroll_id)
                    return rank
                rank += 1

            # Fetch next batch
            response = es.scroll(scroll_id=scroll_id, scroll=scroll_time)
            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]

        # Not found
        es.clear_scroll(scroll_id=scroll_id)
        raise Exception("Doc not found")
