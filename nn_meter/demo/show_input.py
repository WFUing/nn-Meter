import tensorflow as tf


def inspect_input_shape(pb_model_path):
    with tf.io.gfile.GFile(pb_model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # 查找 Placeholder 节点
    for node in graph_def.node:
        if node.op == "Placeholder":
            print(f"Input Node Name: {node.name}")
            if "shape" in node.attr:
                shape = node.attr["shape"].shape
                print("Input Shape:")
                print([dim.size for dim in shape.dim])
            else:
                print("No shape information available for this node.")


# 使用函数
pb_model_path = "/home/wds/zhitai/graduate/projects/nn-Meter/data/pb_models/yolo-v3-tiny-tf_fixed.pb"
inspect_input_shape(pb_model_path)
