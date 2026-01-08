from typing import Union, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors

def top_knn_test(
        Y,
        Y_pred,
        Y_pred_ids,
        k: Union[int, Sequence[int]],
        metric: str = 'euclidean'
):
    neighbors = NearestNeighbors(metric=metric)

    if not isinstance(k, (list, tuple, np.ndarray)):
        k = [k]

    neighbors.fit(Y)

    nearest_ids = neighbors.kneighbors(Y_pred, n_neighbors=np.max(k), return_distance=False)
    Y_pred_ids = Y_pred_ids[:, None]
    accuracy = [
        np.any(nearest_ids[:, :int(some_k)] == Y_pred_ids, axis=1).mean()
        for some_k in k
    ]
    return accuracy