


def K_nearest_neighbors(X, y, K, lam):
    """

    """

    #TODO

    scores = model_selection.cross_val_score(lasso, X, y, scoring="neg_mean_squared_error", cv=K)
    return -1 * scores.mean()
