import tensorflow as tf


def load_and_print_pb_model(pb_model_path):
    # 加载模型
    with tf.io.gfile.GFile(pb_model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # 导入图
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    # 打印节点信息
    for node in graph.as_graph_def().node:
        print(f"Name: {node.name}, Op: {node.op}, Inputs: {node.input}")


# 使用函数
pb_model_path = "/home/wds/zhitai/graduate/projects/nn-Meter/data/pb_models/yolo-v3-tiny-tf_fixed.pb"
load_and_print_pb_model(pb_model_path)
