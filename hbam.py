import networkx as nx
import numpy as np

SIGNATURE_SIZE = 8
EMBEDDING_SIZE = 300


# TODO: add code to compute the algorithmic complexity of a string
# TODO: add code to shuffle arrays and compare algorithmic complexities


def add_node(g: nx.Graph) -> nx.Graph:
    g = g.copy()
    target = np.random.choice(g.nodes)
    g.add_edge(max(g.nodes) + 1, target)

    return g


def del_node(g: nx.Graph) -> nx.Graph:
    g = g.copy()
    target = np.random.choice(g.nodes)
    g.remove_node(target)

    return g


def add_edge(g: nx.Graph) -> nx.Graph:
    g = g.copy()
    source = np.random.choice(g.nodes)
    target = np.random.choice([n for n in g.nodes if (source, n) not in g.edges])
    g.add_edge(source, target)

    return g


def del_edge(g: nx.Graph) -> nx.Graph:
    g = g.copy()
    edges = list(g.edges)
    edge = edges[np.random.choice(len(edges))]
    g.remove_edge(*edge)

    return g


def change_edge(g: nx.Graph) -> nx.Graph:
    g = g.copy()
    edges = list(g.edges)
    source, target = edges[np.random.choice(len(edges))]
    new_target = np.random.choice(list(g.nodes - [target]))
    g.add_edge(source, new_target)

    return g


ACTIONS = {
    'add_node': add_node,
    'del_node': del_node,
    'add_edge': add_edge,
    'del_edge': del_edge,
    'change_edge': change_edge
}


def cosine(x: np.array, y: np.array, w: np.array = None) -> float:
    """
    Computes the weighted cosine distance between two vectors

    :param x: input array
    :param y: input array
    :param w: array of weights
    :return: weighted cosine distance
    """
    assert x.shape == y.shape, "Input arrays must have identical shapes"
    assert w is None or w.shape == x.shape, "Weight array must have the same shape as the input array"

    w = np.ones(x.shape) if w is None else w

    similarity = (x * y * w).sum() / (np.sqrt((w * x ** 2).sum()) * np.sqrt((w * y ** 2).sum()))

    return similarity


def modify(g: nx.Graph, action: str = None) -> nx.Graph:
    """
    Performs a random modification of the input graph

    :param g: input graph to which modification is applied
    :param action: the name of the modification to be performed, if None, then a random modification is performed
    :return: modified graph
    """
    assert action in ACTIONS or action is None, "Wrong value of the action parameter"

    if not action:
        result = np.random.choice(list(ACTIONS.values()))(g)
    else:
        result = ACTIONS[action](g)

    return result


def get_weights(n: int, type: str = 'linear') -> np.array:
    """
    Returns the weight vector computed by one of three possible weighting functions: linear, exponential, or adaptive

    :param n: length of the weight vector
    :param type: weighting function
    :return: array with weights
    """
    assert type in ['linear', 'exponential', 'adaptive'] or type is None, "Wrong type of weighting function"

    if type == 'linear' or type is None:
        w = np.array([(n - i) / n for i in range(n)])
    elif type == 'exponential':
        w = np.array([1 / (1 + i) for i in range(n)])
    elif type == 'adaptive':
        # TODO: implement the adaptive weight
        w = np.array([(n - i) / n for i in range(n)])

    return w


def embed(M: np.array, signature_size: int = SIGNATURE_SIZE) -> np.array:
    """
    Embeds an adjacency matrix as a hierarchical bitmap
    :param M:
    :param signature_size:
    :return:
    """
    n_rows, n_cols = M.shape
    M = M.reshape(n_rows * n_cols, )
    _M = unbinarize(M, signature_size=signature_size)

    embedding = seq2hbseq(_M, signature_size=signature_size)

    return embedding


def permute(g: nx.Graph) -> nx.Graph:
    """
    Creates a random permutation of the graph

    :param g: input graph
    :return: isomorphic graph with relabeled vertices
    """
    permutation = np.random.permutation(g.nodes)
    mapping = {k: v for (k, v) in zip(g.nodes, permutation)}

    h = nx.relabel_nodes(g, mapping=mapping)

    return h


def compression(M: np.array, signature_size: int = SIGNATURE_SIZE) -> float:
    """
    Computes the ratio of the size of embedded matrix to the original size of the matrix

    :param M: input array
    :param signature_size: size of a single signature
    :return: encoding compression
    """
    embedding = embed(M, signature_size=signature_size)

    original_length = M.size
    embedding_length = len(embedding)

    return embedding_length / original_length


def unbinarize(a: np.array, signature_size: int = SIGNATURE_SIZE) -> np.array:
    """
    Converts a binary array into an array of integers

    :param a: binary array
    :param signature_size: size of a single signature
    :return: output array
    """

    # length of the input array must be the multiple of the signature size
    if len(a) % signature_size:
        a = np.append(a, np.zeros(signature_size - len(a) % signature_size))
    a = a.reshape(len(a) // signature_size, signature_size).astype(int)
    result = np.apply_along_axis(bin2int, axis=1, arr=a, signature_size=signature_size)

    return result


def binarize(a: np.array) -> np.array:
    """
    Converts an array of integers into a binary array

    :param a: input array
    :return: binary array
    """

    return a.astype(bool).astype(int)


def bin2int(a: np.array, signature_size: int = SIGNATURE_SIZE) -> int:
    """
    Encodes a single signature represented as a binary array into an integer

    :param a: input array
    :param signature_size: size of a single signature
    :return: integer representation of an array
    """

    assert signature_size <= SIGNATURE_SIZE, f"Size of binary signature cannot be larger than {SIGNATURE_SIZE}"
    assert len(a) <= signature_size, f"Input array size cannot be larger than {SIGNATURE_SIZE}"
    assert np.unique(a).tolist() in [[0], [1], [0, 1]], f"Input array must be binary"

    str_array = ''.join(map(str, a))
    int_value_of_array = int(str_array, base=2)

    return int_value_of_array


def seq2hbseq(a: np.array, signature_size: int = SIGNATURE_SIZE) -> np.array:
    """
    Converts a sequence of integers into a hierarchical bitmap sequence

    :param a: input array
    :param signature_size: size of a single signature
    :return:  array of ints forming the condensed hierarchical bitmap sequence
    """

    # TODO: add tests for this method

    # length of the input array must be the multiple of the signature size
    a = np.append(a, np.zeros(signature_size - len(a) % signature_size))

    result = np.empty(0)

    current_level = a

    while len(current_level) >= signature_size:

        result = np.insert(result, 0, current_level)

        if len(current_level) % signature_size:
            current_level = np.append(current_level, np.zeros(signature_size - len(current_level) % signature_size))
        current_level = current_level.reshape(len(current_level) // signature_size, signature_size)

        if len(current_level) >= signature_size:
            next_level = np.apply_along_axis(binarize, axis=1, arr=current_level)
        else:
            next_level = binarize(current_level)

        if len(next_level) % signature_size and len(next_level) > 1:
            next_level = np.append(next_level,
                                   np.zeros((signature_size - len(next_level) % signature_size, signature_size)))
            next_level = next_level.reshape(len(next_level) // signature_size, signature_size).astype(int)

        next_level = np.apply_along_axis(bin2int, axis=1, arr=next_level, signature_size=signature_size)
        current_level = next_level

    result = np.insert(result, 0, current_level)
    result = result[result > 0].astype(int)

    result_length = len(result)

    if result_length < EMBEDDING_SIZE:
        return np.pad(result, pad_width=(0, EMBEDDING_SIZE - result_length), mode='constant', constant_values=0)
    else:
        return result[:EMBEDDING_SIZE]
