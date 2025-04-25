def score(query: str, paper_title: str, alpha: float, beta: float) -> float:
    page_rank_value = page_rank(paper_title)
    tfidf_value = tfidf(query, paper_title)
    return alpha*tfidf_value + beta*page_rank_value

def tfidf():
    pass

def page_rank(paper_title: str) -> float:
    pass


