import fire

import tensorflow as tf
import configs
import os
from model import checkpoint_to_savedmodel, get_model, graphs_tuple_to_sparse_matrices
import numpy as np
from utils import init_octave
import json
import shutil
import sonnet as snt
import graph_nets as gn


def convert(model_name='47887', config='GRAPH_LAPLACIAN_EVAL', checkpoint_dir=None, savedmodel_dir=None):
    seed = 1
    resolved_config = getattr(configs, config)
    if checkpoint_dir is None or not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError("Need to specify checkpoint_dir")
    if savedmodel_dir is None:
        raise NotADirectoryError("Need to specify savedmodel_dir")

    if savedmodel_dir is not None and os.path.isdir(savedmodel_dir):
        shutil.rmtree(savedmodel_dir, ignore_errors=True)

    os.makedirs(savedmodel_dir)

    if model_name is None:
        raise RuntimeError("model name required")
    model_name = str(model_name)

    # fix random seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    octave = init_octave(seed)

    config_file = f"results/{model_name}/config.json"
    with open(config_file) as f:
        data = json.load(f)
        model_config = configs.ModelConfig(**data['model_config'])
        run_config = configs.RunConfig(**data['run_config'])

    model = get_model(model_name, model_config, run_config, octave)

    @tf.function(input_signature=[tf.TensorSpec([None], tf.float64, "globals"),
                                  tf.TensorSpec([None, None], tf.float64, "nodes"),
                                  tf.TensorSpec([None, None], tf.float64, "edges"),
                                  tf.TensorSpec([None], tf.int32, "senders"),
                                  tf.TensorSpec([None], tf.int32, "receivers")])
    def inference(_globals, _nodes, _edges, _senders, _receivers):
        graph_tuples = gn.utils_tf.data_dicts_to_graphs_tuple([{
            "globals": _globals,
            "nodes": _nodes,
            "edges": _edges,
            "senders": _senders,
            "receivers": _receivers,
        }])
        return model(graph_tuples)

    to_save = snt.Module()
    to_save.predict_prolongation = inference
    to_save.all_variables = list(model.variables)
    signatures = {
        'serving_default': to_save.predict_prolongation.get_concrete_function(),
    }
    options = tf.saved_model.SaveOptions(function_aliases={
        'predict_prolongation': to_save.predict_prolongation,
    })
    # tf.saved_model.save(to_save, savedmodel_dir, signatures, options)  # signatures and options???
    tf.saved_model.save(to_save, savedmodel_dir)

    print("Saved OK. Testing a load and inference....")

    loaded = tf.saved_model.load(savedmodel_dir)

    # Use the inference method. Note this doesn't run the Python code from `to_save`
    # but instead uses the TensorFlow Graph that is part of the saved model.
    loaded.predict_prolongation(
        np.random.rand(100).astype(np.float64),
        np.random.rand(30, 2).astype(np.float64),
        np.random.rand(4, 3).astype(np.float64),
        np.random.randint(30, size=4, dtype=np.int32),
        np.random.randint(30, size=4, dtype=np.int32), )

    # The all_variables property can be used to retrieve the restored variables.
    assert len(loaded.all_variables) > 0
    print("Model correctly saved to {}".format(savedmodel_dir))


# docker run -p 8500:8500 -p 8501:8501 --mount  -e MODEL_NAME=prolongate -t
# ❯ docker run -p 8500:8500 -p 8501:8501 \
# --mount type=bind,source=savedmodels/47887,target=/models/prolongate/1 \
# -e MODEL_NAME=prolongate -t tensorflow/serving

# curl --output - http://localhost:8501/v1/models/prolongate/metadata -H 'Content-Type: application/json'

# OUPUT IS SUPER UGLY. Can we tag outputs properly?
# {
# "model_spec":{
#  "name": "prolongate",
#  "signature_name": "",
#  "version": "1"
# }
# ,
# "metadata": {"signature_def": {
#  "signature_def": {
#   "serving_default": {
#    "inputs": {
#     "globals": {
#      "dtype": "DT_DOUBLE",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "-1",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "serving_default_globals:0"
#     },
#     "edges": {
#      "dtype": "DT_DOUBLE",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "-1",
#         "name": ""
#        },
#        {
#         "size": "-1",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "serving_default_edges:0"
#     },
#     "senders": {
#      "dtype": "DT_INT32",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "-1",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "serving_default_senders:0"
#     },
#     "receivers": {
#      "dtype": "DT_INT32",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "-1",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "serving_default_receivers:0"
#     },
#     "nodes": {
#      "dtype": "DT_DOUBLE",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "-1",
#         "name": ""
#        },
#        {
#         "size": "-1",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "serving_default_nodes:0"
#     }
#    },
#    "outputs": {
#     "output_0": {
#      "dtype": "DT_DOUBLE",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "-1",
#         "name": ""
#        },
#        {
#         "size": "1",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "StatefulPartitionedCall:0"
#     },
#     "output_1": {
#      "dtype": "DT_DOUBLE",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "-1",
#         "name": ""
#        },
#        {
#         "size": "1",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "StatefulPartitionedCall:1"
#     },
#     "output_2": {
#      "dtype": "DT_INT32",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "-1",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "StatefulPartitionedCall:2"
#     },
#     "output_3": {
#      "dtype": "DT_INT32",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "-1",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "StatefulPartitionedCall:3"
#     },
#     "output_4": {
#      "dtype": "DT_DOUBLE",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "1",
#         "name": ""
#        },
#        {
#         "size": "64",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "StatefulPartitionedCall:4"
#     },
#     "output_5": {
#      "dtype": "DT_INT32",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "1",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "StatefulPartitionedCall:5"
#     },
#     "output_6": {
#      "dtype": "DT_INT32",
#      "tensor_shape": {
#       "dim": [
#        {
#         "size": "1",
#         "name": ""
#        }
#       ],
#       "unknown_rank": false
#      },
#      "name": "StatefulPartitionedCall:6"
#     }
#    },
#    "method_name": "tensorflow/serving/predict"
#   },
#   "__saved_model_init_op": {
#    "inputs": {},
#    "outputs": {
#     "__saved_model_init_op": {
#      "dtype": "DT_INVALID",
#      "tensor_shape": {
#       "dim": [],
#       "unknown_rank": true
#      },
#      "name": "NoOp"
#     }
#    },
#    "method_name": ""
#   }
#  }
# }
# }
# }


def print_graphs_tuple(graphs_tuple):
    print("Shapes of GraphsTuple's fields:")
    for field in gn.graphs.ALL_FIELDS:
        print("{} : {}".format(field,
                               None if hasattr(graphs_tuple, field) is None else getattr(graphs_tuple, field).shape))
    print("\nData contained in GraphsTuple's fields:")
    # print(f"globals:\n{graphs_tuple.globals}")
    # print(f"nodes:\n{graphs_tuple.nodes}")
    # print(f"edges:\n{graphs_tuple.edges}")
    # print(f"senders:\n{graphs_tuple.senders}")
    # print(f"receivers:\n{graphs_tuple.receivers}")
    # print(f"n_node:\n{graphs_tuple.n_node}")
    # print(f"n_edge:\n{graphs_tuple.n_edge}")


def test_inference(savedmodel_dir=None):
    loaded = tf.saved_model.load(savedmodel_dir)

    # Use the inference method. Note this doesn't run the Python code from `to_save`
    # but instead uses the TensorFlow Graph that is part of the saved model.
    num_nodes = 4
    num_edges = 6
    num_node_features = 2
    num_edge_features = 3


    def rnd_is_a_C_node():
        if np.random.rand() <= 0.5: # is a C-Node
            return np.array([1., 0.]).astype(np.float64)
        else:# is not a C-Node
            return np.array([0,1]).astype(np.float64)

    def rnd_is_part_of_sparsity_pattern(A_ij_val):
        if np.random.rand() <= 0.5:  # is part of sparsity pattern
            return np.array([A_ij_val, 1, 0]).astype(np.float64)
        else:
            return np.array([A_ij_val, 0, 1]).astype(np.float64)

    def rnd_A_ij():
        return np.random.rand() * 100


    _globals = []
    _nodes = np.random.rand(num_nodes, num_node_features).astype(np.float64)
    _edges = np.random.rand(num_edges, num_edge_features).astype(np.float64)
    edge_defs = dict()
    while len(edge_defs) < num_edges:
        a = np.int32((np.random.rand() * num_nodes))
        b = np.int32((np.random.rand() * num_nodes))
        while b == a:
            b = np.int32((np.random.rand() * num_nodes))
        if (a, b) not in edge_defs.keys():
            edge_defs[(a, b)] = None
    _senders = np.array([a for (a, b) in edge_defs.keys()])
    _receivers = np.array([b for (a, b) in edge_defs.keys()])

    print("globals:\n", _globals)
    print("nodes:\n", _nodes)
    print("edges:\n", _edges)
    print("senders:\n", _senders)
    print("receivers:\n", _receivers)

    result = loaded.predict_prolongation(
        _globals,
        _nodes,
        _edges,
        _senders,
        _receivers)

    print_graphs_tuple(result)

    for i, x in enumerate(graphs_tuple_to_sparse_matrices(result)):
        print(gn.graphs.ALL_FIELDS[i])
        print(tf.sparse.to_dense(x).numpy())


if __name__ == '__main__':
    fire.Fire(test_inference)
