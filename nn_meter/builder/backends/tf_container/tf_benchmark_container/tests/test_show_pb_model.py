import tensorflow as tf

def list_pb_tensors(model_path):
    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    for node in graph_def.node:
        print(f"Operation name: {node.name} - Type: {node.op}")


model_path = "/data/pb_models/resnet18_0.pb"
list_pb_tensors(model_path)
