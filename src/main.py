from tfidf_computation import run_tfidf, compute_score
from scoring import score


def main():
    # run_tfidf("data/sample.json")
    result = compute_score("network protocol")
    print(result)
    # run_tfidf("data/dblp.v12.json")


if __name__ == "__main__":
    main()
