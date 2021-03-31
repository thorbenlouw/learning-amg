import graph_nets as gn
import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix, coo_matrix

from data import As_poisson_grid
from graph_net_model import EncodeProcessDecodeNonRecurrent
from utils import get_accelerator_device
from numba import jit

from typing import Tuple


def get_model(model_name, model_config, run_config, octave, train=False, train_config=None):
    dummy_input = As_poisson_grid(1, 7 ** 2)[0]
    checkpoint_dir = './training_dir/' + model_name
    graph_model, optimizer, global_step = load_model(checkpoint_dir, dummy_input, model_config,
                                                     run_config,
                                                     octave, get_optimizer=train,
                                                     train_config=train_config)
    if train:
        return graph_model, optimizer, global_step
    else:
        return graph_model


def load_model(checkpoint_dir, dummy_input, model_config, run_config, octave, get_optimizer=True,
               train_config=None):
    model = create_model(model_config)

    # we have to use the model at least once to get the list of variables
    model(csrs_to_graphs_tuple([dummy_input], octave, coarse_nodes_list=np.array([[0, 1]]),
                               baseline_P_list=[tf.convert_to_tensor(dummy_input.toarray()[:, [0, 1]])],
                               node_indicators=run_config.node_indicators,
                               edge_indicators=run_config.edge_indicators))

    variables = model.variables
    variables_dict = {variable.name: variable for variable in variables}
    if get_optimizer:
        global_step = tf.Variable(1, name="global_step")
        decay_steps = 100
        decay_rate = 1.0
        learning_rate = tf.train.exponential_decay(train_config.learning_rate, global_step, decay_steps, decay_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        checkpoint = tf.train.Checkpoint(**variables_dict, optimizer=optimizer, global_step=global_step)
    else:
        optimizer = None
        global_step = None
        checkpoint = tf.train.Checkpoint(**variables_dict)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is None:
        raise RuntimeError(f'training_dir {checkpoint_dir} does not exist')
    checkpoint.restore(latest_checkpoint)
    return model, optimizer, global_step


def make_dummy_graph_tuple(run_config) -> gn.graphs.GraphsTuple:
    dummy_input = As_poisson_grid(1, 7 ** 2)[0]  # magic 49x49 Poisson grid?
    return csrs_to_graphs_tuple([dummy_input], None, coarse_nodes_list=np.array([[0, 1]]),
                                baseline_P_list=[tf.convert_to_tensor(dummy_input.toarray()[:, [0, 1]])],
                                node_indicators=run_config.node_indicators,
                                edge_indicators=False)


def checkpoint_to_savedmodel(checkpoint_dir: str, savedmodel_dir: str, model_config, run_config):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is None:
        raise RuntimeError(f'checkpoint dir {checkpoint_dir} does not exist')

    with get_accelerator_device():
        model = EncodeProcessDecodeNonRecurrent(
            num_cores=model_config.mp_rounds,
            edge_output_size=1,
            node_output_size=1,
            global_block=model_config.global_block,
            latent_size=model_config.latent_size,
            num_layers=model_config.mlp_layers,
            concat_encoder=model_config.concat_encoder)  # This is happily a tf.Module

        # we have to use the model at least once to get the list of variables
        dummy_input: gn.graphs.GraphsTuple = make_dummy_graph_tuple(run_config)
        model(dummy_input)  # One dummy invocation to load variables
        variables = model.variables
        variables_dict = {variable.name: variable for variable in variables}
        checkpoint = tf.train.Checkpoint(**variables_dict)
        checkpoint.restore(latest_checkpoint).expect_partial()
        tf.saved_model.save(model, savedmodel_dir)  # signatures and options???
        print("OK")


def create_model(model_config):
    with get_accelerator_device():
        return EncodeProcessDecodeNonRecurrent(num_cores=model_config.mp_rounds, edge_output_size=1,
                                               node_output_size=1, global_block=model_config.global_block,
                                               latent_size=model_config.latent_size,
                                               num_layers=model_config.mlp_layers,
                                               concat_encoder=model_config.concat_encoder)


def csrs_to_graphs_tuple(csrs, octave, node_feature_size=128, coarse_nodes_list=None, baseline_P_list=None,
                         node_indicators=True, edge_indicators=True):
    dtype = tf.float64

    # build up the arguments for the GraphsTuple constructor
    n_node = tf.convert_to_tensor([csr.shape[0] for csr in csrs])
    n_edge = tf.convert_to_tensor([csr.nnz for csr in csrs])

    if not edge_indicators:
        numpy_edges = np.concatenate([csr.data for csr in csrs])
        edges = tf.expand_dims(tf.convert_to_tensor(numpy_edges, dtype=dtype), axis=1)
    else:
        edge_encodings_list = []
        for csr, coarse_nodes, baseline_P in zip(csrs, coarse_nodes_list, baseline_P_list):
            if tf.is_tensor(baseline_P):
                baseline_P = csr_matrix(baseline_P.numpy())

            baseline_P_rows, baseline_P_cols = P_square_sparsity_pattern(baseline_P, baseline_P.shape[0],
                                                                         coarse_nodes, octave)
            coo = csr.tocoo()

            # construct numpy structured arrays, where each element is a tuple (row,col), so that we can later use
            # the numpy set function in1d()
            baseline_P_indices = np.core.records.fromarrays([baseline_P_rows, baseline_P_cols], dtype='i,i')
            coo_indices = np.core.records.fromarrays([coo.row, coo.col], dtype='i,i')

            same_indices = np.in1d(coo_indices, baseline_P_indices, assume_unique=True)
            baseline_edges = same_indices.astype(np.float64)
            non_baseline_edges = (~same_indices).astype(np.float64)

            edge_encodings = np.stack([coo.data, baseline_edges, non_baseline_edges]).T
            edge_encodings_list.append(edge_encodings)
        numpy_edges = np.concatenate(edge_encodings_list)
        edges = tf.convert_to_tensor(numpy_edges, dtype=dtype)

    # COO format for sparse matrices contains a list of row indices and a list of column indices
    coos = [csr.tocoo() for csr in csrs]
    senders_numpy = np.concatenate([coo.row for coo in coos])
    senders = tf.convert_to_tensor(senders_numpy)
    receivers_numpy = np.concatenate([coo.col for coo in coos])
    receivers = tf.convert_to_tensor(receivers_numpy)

    # see the source of _concatenate_data_dicts for explanation
    offsets = gn.utils_tf._compute_stacked_offsets(n_node, n_edge)
    senders += offsets
    receivers += offsets

    if not node_indicators:
        nodes = None
    else:
        node_encodings_list = []
        for csr, coarse_nodes in zip(csrs, coarse_nodes_list):
            coarse_indices = np.in1d(range(csr.shape[0]), coarse_nodes, assume_unique=True)

            coarse_node_encodings = coarse_indices.astype(np.float64)
            fine_node_encodings = (~coarse_indices).astype(np.float64)
            node_encodings = np.stack([coarse_node_encodings, fine_node_encodings]).T

            node_encodings_list.append(node_encodings)

        numpy_nodes = np.concatenate(node_encodings_list)
        nodes = tf.convert_to_tensor(numpy_nodes, dtype=dtype)

    graphs_tuple = gn.graphs.GraphsTuple(
        nodes=nodes,
        edges=edges,
        globals=None,
        receivers=receivers,
        senders=senders,
        n_node=n_node,
        n_edge=n_edge
    )
    if not node_indicators:
        graphs_tuple = gn.utils_tf.set_zero_node_features(graphs_tuple, 1, dtype=dtype)

    graphs_tuple = gn.utils_tf.set_zero_global_features(graphs_tuple, node_feature_size, dtype=dtype)

    return graphs_tuple


# TODO really need to test that this does the same thing!

# @jit(nopython=True, parallel=True)
def do_fast_dense_for_small_matrix(P: np.ndarray, size: int, coarse_nodes: np.array) -> Tuple[np.array, np.array]:
    size_diff = ((0, size - P.shape[0]), (0, size - P.shape[1]))
    sparsity = np.pad((P != 0.).todense(), size_diff).astype(float) if P.shape != (size, size) else (
            P != 0).to_dense().astype(float)
    mask = np.zeros(shape=(size, size), dtype=np.bool)
    mask[:, coarse_nodes] = 1
    sparsity = (sparsity * mask).astype(np.int8)
    as_coo = coo_matrix(sparsity)
    return as_coo.row, as_coo.col


# @jit(nopython=True, parallel=True)
def python_square_P(P: np.ndarray, coarse_nodes: np.array) -> Tuple[np.array, np.array]:
    # % Find out how many rows and cols our matrix will have
    # %% num_rows is total_size
    # %% num_cols is the number of columns in coarse_nodes
    # % Create a sparse matrix P of size (num_rows,num_cols).
    # %% row indices are in P_rows
    # %% col indicides are in P_cols
    # %% values are in P_values
    # % Create a sparse matrix P_square of num_rows x num_rows
    # % Set P_square's coarse nodes locations to P's vals
    # %% Return the locations of non-zero elems in P_square (sparsity pattern)
    P_coo = P.tocoo()
    rows = []
    cols = []
    for r, c, v in zip(P_coo.row, P_coo.col, P_coo.data):
        if c in coarse_nodes:
            rows.append(r)
            cols.append(c)
    return rows, cols


def P_square_sparsity_pattern(P, size, coarse_nodes, octave):
    # if size < 100:
    #     P_dense = np.zeros(P.shape)
    #     P.toarray(out=P_dense)
    #     P_dense = P_dense.copy()
    #     P_dense.resize((size, size), refcheck=False)
    #     mask = np.zeros((size, size), dtype=np.bool)
    #     mask[:, coarse_nodes] = True
    #     P_sparse = P_dense[mask].tocoo()
    #     return P_sparse.row, P_sparse.col

    if size < 6000:
        return do_fast_dense_for_small_matrix(P, size, coarse_nodes)

    # If coarse nodes are in a sizexsize shape, which are non-zero (as per P)?
    # Needs to scale to very large, so keep sparse

    return python_square_P()
    # P_coo = P.tocoo()
    # P_rows = octave.double(P_coo.row + 1)
    # P_cols = octave.double(P_coo.col + 1)
    # P_values = octave.double(P_coo.data)
    # coarse_nodes = octave.double(coarse_nodes + 1)
    # rows, cols = octave.square_P(P_rows, P_cols, P_values, size, coarse_nodes, nout=2)
    # rows = rows.reshape(rows.size, order='F') - 1  # -1 becauase Matlab 1 indexed
    # cols = cols.reshape(cols.size, order='F') - 1  # It's already a 1D byt is (X,1) and we want it dimensionless (X,)
    # rows, cols = rows.T, cols.T  # WHY?! the [0]?! Mistak?! REMOVE?? Why would we just want the 1x1?
    # return rows, cols


def graphs_tuple_to_sparse_tensor(graphs_tuple):
    senders = graphs_tuple.senders
    receivers = graphs_tuple.receivers
    indices = tf.cast(tf.stack([senders, receivers], axis=1), tf.int64)

    # first element in the edge feature is the value, the other elements are metadata
    values = tf.squeeze(graphs_tuple.edges[:, 0])

    shape = tf.concat([graphs_tuple.n_node, graphs_tuple.n_node], axis=0)
    shape = tf.cast(shape, tf.int64)

    matrix = tf.sparse.SparseTensor(indices, values, shape)
    # reordering is required because the pyAMG coarsening step does not preserve indices order
    matrix = tf.sparse.reorder(matrix)

    return matrix


def to_prolongation_matrix_csr(matrix, coarse_nodes, baseline_P, nodes, normalize_rows=True,
                               normalize_rows_by_node=False):
    """
    sparse version of the above function, for when the dense matrix is too large to fit in GPU memory
    used only for inference, so no need for backpropagation, inputs are csr matrices
    """
    # prolongation from coarse point to itself should be identity. This corresponds to 1's on the diagonal
    matrix.setdiag(np.ones(matrix.shape[0]))

    # select only columns corresponding to coarse nodes
    matrix = matrix[:, coarse_nodes]

    # set sparsity pattern (interpolatory sets) to be of baseline prolongation
    baseline_P_mask = (baseline_P != 0).astype(np.float64)
    matrix = matrix.multiply(baseline_P_mask)
    matrix.eliminate_zeros()

    if normalize_rows:
        if normalize_rows_by_node:
            baseline_row_sum = nodes
        else:
            baseline_row_sum = baseline_P.sum(axis=1)
            baseline_row_sum = np.array(baseline_row_sum)[:, 0]

        matrix_row_sum = np.array(matrix.sum(axis=1))[:, 0]
        # https://stackoverflow.com/a/12238133
        matrix_copy = matrix.copy()
        matrix_copy.data /= matrix_row_sum.repeat(np.diff(matrix_copy.indptr))
        matrix_copy.data *= baseline_row_sum.repeat(np.diff(matrix_copy.indptr))
        matrix = matrix_copy
    return matrix


def to_prolongation_matrix_tensor(matrix, coarse_nodes, baseline_P, nodes,
                                  normalize_rows=True,
                                  normalize_rows_by_node=False):
    dtype = tf.float64
    matrix = tf.cast(matrix, dtype)
    matrix = tf.sparse.to_dense(matrix)

    # prolongation from coarse point to itself should be identity. This corresponds to 1's on the diagonal
    matrix = tf.linalg.set_diag(matrix, tf.ones(matrix.shape[0], dtype=dtype))

    # select only columns corresponding to coarse nodes
    matrix = tf.gather(matrix, coarse_nodes, axis=1)

    # set sparsity pattern (interpolatory sets) to be of baseline prolongation
    baseline_zero_mask = tf.cast(tf.not_equal(baseline_P, tf.zeros_like(baseline_P)), dtype)
    matrix = matrix * baseline_zero_mask

    if normalize_rows:
        if normalize_rows_by_node:
            baseline_row_sum = nodes
        else:
            baseline_row_sum = tf.reduce_sum(baseline_P, axis=1)
        baseline_row_sum = tf.cast(baseline_row_sum, dtype)

        matrix_row_sum = tf.reduce_sum(matrix, axis=1)
        matrix_row_sum = tf.cast(matrix_row_sum, dtype)

        # there might be a few rows that are all 0's - corresponding to fine points that are not connected to any
        # coarse point. We use "divide_no_nan" to let these rows remain 0's
        matrix = tf.math.divide_no_nan(matrix, tf.reshape(matrix_row_sum, (-1, 1)))

        matrix = matrix * tf.reshape(baseline_row_sum, (-1, 1))
    return matrix


def graphs_tuple_to_sparse_matrices(graphs_tuple, return_nodes=False):
    num_graphs = int(graphs_tuple.n_node.shape[0])
    graphs = [gn.utils_tf.get_graph(graphs_tuple, i)
              for i in range(num_graphs)]

    matrices = [graphs_tuple_to_sparse_tensor(graph) for graph in graphs]

    if return_nodes:
        nodes_list = [tf.squeeze(graph.nodes) for graph in graphs]
        return matrices, nodes_list
    else:
        return matrices
