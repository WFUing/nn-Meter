import tensorflow as tf


def fix_yolo_input_shape(pb_model_path, output_path, static_shape):
    with tf.io.gfile.GFile(pb_model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    for node in graph_def.node:
        if node.op == "Placeholder":
            print(f"Modifying Placeholder: {node.name}")
            shape = node.attr["shape"].shape
            for i, dim in enumerate(static_shape):
                shape.dim[i].size = dim
            print(f"Updated shape to: {static_shape}")

    with tf.io.gfile.GFile(output_path, "wb") as f:
        f.write(graph_def.SerializeToString())


fix_yolo_input_shape("/home/wds/zhitai/graduate/projects/nn-Meter/data/pb_models/yolo-v3-tiny-tf.pb", "/home/wds/zhitai/graduate/projects/nn-Meter/data/pb_models/yolo-v3-tiny-tf_fixed.pb", [1, 416, 416, 3])
