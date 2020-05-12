'''
Name: app/algo/metrics.py
Author: David Stern - Team IC.
Purpose: Provides similarity metrics.

Easily extended with more or different styles of metrics.
Demonstrates each method of implementation, with explanations.
'''

from sklearn.metrics.pairwise import (cosine_similarity, euclidean_distances,
                                      manhattan_distances)
import numpy as np
from scipy.spatial.distance import euclidean, canberra
from typing import Callable
from settings import LIBRARY_METRICS

# Defines a type alias
Array = np.ndarray


class SimilarityMetric:
    '''
    Parent class to be inherited by each Similarity Metric.

    Takes advantage of the similarities between the calculation of each metric,
    to save on repeated code.
    '''
    def similarity(self, word_matrices: Array, unknown_vector: Array) -> float:
        '''Defines a general distance function'''

        return 1 - self.avg_distance(word_matrices, unknown_vector)

    def set_scaling(self, func: Callable[[float], float] = None):
        '''Setter for the scaling function. This is applied to the result of
           the normalised distance'''
        self._scaling = func

    def set_dist(self, func: Callable[[Array, Array], float]) -> None:
        '''Setter for the distance function'''
        self._dist = func

    def set_name(self, name: str) -> None:
        self._name = name

    def avg_distance(self, word_matrices: Array, unknown_vector: Array) -> float:
        '''
        Calculates the average distance between an unknown document vector,
        calculated by a provided distance method, and each of the word matrix
        vectors for the valid documents, with some scaling applied to make the
        result between 0 and 1.

        However, note that euclidean distance, for example, already shares a
        direct relationship w/ cosine similarity, and so a euclidean similarity
        is simply a transformed cosine distance:
            https://stats.stackexchange.com/a/158309
        '''
        # Use a library function on the sum of all doc vectors instead of
        # individually
        if self._name in LIBRARY_METRICS:
            word_matrices = np.sum(word_matrices, axis=0).reshape(1, -1)
            distance = self._dist(word_matrices[0].reshape(1, -1),
                                  unknown_vector)

        # Get the avg by summing the distances b/w the new vector & each text by
        # the student
        else:
            distance = 0
            for i, text_array in enumerate(word_matrices):
                    distance += self._dist(text_array, unknown_vector)

        # @TODO: Allow settings.py to disable the normalisation below
        # Apply scaling, if defined
        if self._scaling is not None:
            return self._scaling(distance / len(word_matrices))
        return distance / len(word_matrices)


class EuclideanMetric(SimilarityMetric):

    def __init__(self):
        '''
        Sets the scaling and distance functions that define the Euclidean
        Similarity Metric.

        (1 / (.0000000001 + d)) is from https://stats.stackexchange.com/a/158285
        100 is manually chosen, by @Zoran, to help scale.
        tanh squishes values from [0, infty] to [0, 1].
        '''
        self.set_name("Euclidean")
        self.set_scaling(lambda dist: np.tanh(100 * (1 / (.0000000001 + dist))))
        self.set_dist(lambda wm, uv: np.sqrt(np.sum((wm - uv) ** 2, axis=1))[0])


def keselj(wm, uv):
    '''
    Calculate Keselj distance while ignoring division by 0 errors, which are a
    part of the Keselj distance function if you are calculating distances for
    each text.
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nansum(((2 * (wm - uv))/(wm + uv))**2, axis=1)[0]

class KeseljMetric(SimilarityMetric):

    def __init__(self):
        '''
        Sets the scaling and distance functions that define the Keselj
        Similarity Metric.

        (1 / (.0000000001 + d)) is from https://stats.stackexchange.com/a/158285
        2000 is manually chosen, by @Zoran, to help scale.
        '''
        self.set_name("Keselj")
        self.set_scaling(lambda dist: np.tanh(400 *
                                              (1 / (.0000000001 + dist))))
        self.set_dist(keselj)


class ManhattanMetric(SimilarityMetric):
    def __init__(self):
        '''
        Sets the scaling and distance functions that define the Manhattan
        Similarity Metric.

        (1 / (.0000000001 + d)) is from https://stats.stackexchange.com/a/158285
        2000 is manually chosen, by @Zoran, to help scale.
        '''
        self.set_name("Manhattan")
        self.set_scaling(lambda dst: np.tanh(400 * (1 / (.0000000001 + dst))))
        self.set_dist(lambda wm, uv: manhattan_distances(wm, uv)[0][0])


class CanberraMetric(SimilarityMetric):
    def __init__(self):
        '''
        Sets the scaling and distance functions that define the Canberra
        Similarity Metric.

        (1 / (.0000000001 + d)) is from https://stats.stackexchange.com/a/158285
        2000 is manually chosen, by @Zoran, to help scale.
        '''
        self.set_name("Canberra")
        self.set_scaling(lambda dist: np.tanh(100 * (1 / (.0000000001 + dist))))
        self.set_dist(canberra)


class CosineMetric(SimilarityMetric):

    def similarity(self, wm: Array, uv: Array) -> float:
        return cosine_similarity(np.sum(wm, axis=0).reshape(1, -1), uv)[0][0]

    def __init__(self):
        self.set_name("Cosine")
        self.set_scaling()


# Exporting the metrics
metrics = {
    # 'cosine_similarity': lambda a, b: cosine_similarity(np.sum(a, axis=0).reshape(1, -1), b)[0][0],
    'cosine_similarity': lambda a, b: CosineMetric().similarity(a, b),
    'canberra_similarity': lambda a, b: CanberraMetric().similarity(a, b),
    'euclidean_similarity': lambda a, b: EuclideanMetric().similarity(a, b),
    'keselj_similarity': lambda a, b: KeseljMetric().similarity(a, b),
    'manhattan_similarity': lambda a, b: ManhattanMetric().similarity(a, b),
}
