def difference_between_result_sets(
    result_set_1: dict[str, float], 
    result_set_2: dict[str, float], 
    alpha: float, 
    beta: float
) -> float:
    normalizing_factor = sum(result_set_1.values()) + sum(result_set_2.values())
    paper_titles = set(result_set_1.keys())
    paper_titles = paper_titles.union(result_set_2.keys())
    difference = 0.0
    for title in paper_titles:
        if title in result_set_1 and title in result_set_2:
            difference += abs(result_set_1[title] - result_set_2[title])
        elif title in result_set_1:
            difference += result_set_1[title]
        elif title in result_set_2:
            difference += result_set_2[title]
        else:
            raise ValueError("Title does not belong to either result sets. This should not have happened.")

    if normalizing_factor == 0:
        return 0.0
    return difference / normalizing_factor