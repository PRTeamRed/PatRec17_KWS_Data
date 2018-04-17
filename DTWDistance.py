#!/usr/bin/env python
# DTW
# Written by C.B. Doorenbos (2018)


def DTWDistance (feature_vector_1, feature_vector_2, distancemetric = 'euclidean'):
    """
    Computes dynamic time warping distance
    Arguments:
        feature_vector_1 and 2:
            The vectors for which the DTW distance must be computed. Expected
            format: numpy array where dimension 0 corresponds to the features,
            dimension 1 corresponds to the time domain.
        distancemetric:
            Same options as scipy.spatial.distance.cdist():
            e.g. euclidean (default), cityblock, cosine...

    returns: distance, moves
        distance:
            the computed DTW distance
        moves:
            a string with the moves walked by the algorithm:
            0 means one step further in feature_vector_1,
            2 means one step further in feature_vector_2,
            1 means one step further in both feature vectors.


    Implementation:
        The function computes the DTW distance by a greedy search of the
        distance matrix. It starts in the corner (0,0), then increases i, j or
        both at each step, until it reaches the corner (N, M). At every step,
        there are three possibilities, the algorithm picks the move which lands
        on element wth the lowest value of the three in the distance matrix.
    """
    import scipy as sc;
    # Compute distance matrix for all sets of feature vectors
    import scipy.spatial.distance;
    distance_matrix = scipy.spatial.distance.cdist(
            feature_vector_1.transpose(),
            feature_vector_2.transpose(),
            distancemetric);

    # Initialise variables
    moves = "";
    distance = i = j = 0;

    # Traverse distance matrix
    while (i < distance_matrix.shape[0] - 1) & (j < distance_matrix.shape[1] - 1):
        possible_new_positions = sc.array([distance_matrix[i+1, j],
                                           distance_matrix[i+1,j+1],
                                           distance_matrix[i, j+1]])
        move = sc.argmin(possible_new_positions);
        distance += possible_new_positions[move];
        if move == 0:
            i += 1;
            moves += "0"
        elif move == 2:
            j += 1;
            moves += "2"
        else:
            i += 1;
            j += 1;
            moves += "1"

    # When reaching one side of the matrix, finish by moving completely to the corner
    if (i == distance_matrix.shape[0] - 1):
        moves += (distance_matrix.shape[1] - 1 - j) * "2";
        distance += distance_matrix[i, j:].sum();
    elif (j == distance_matrix.shape[1] - 1):
        moves += (distance_matrix.shape[0] - 1 - i) * "0";
        distance += distance_matrix[i:, j].sum();

    # Return distance and moves string
    return distance, moves;


