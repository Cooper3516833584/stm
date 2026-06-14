"""Generate precise NPU diagnostic test models targeting MaxPool chain patterns."""
import onnx, numpy as np, os
from onnx import helper, TensorProto

out_dir = os.path.dirname(os.path.abspath(__file__))

# Common params: SPPF block from YOLO model.9
# Conv(64->32, k=1) -> MaxPool(k=5,s=1,p=2) ×3 chained -> Concat
C, H, W = 64, 80, 80
mid_c = 32

def make_sppf_model(name, num_pools, chained=True, kernel=5, input_size=80,
                     add_resize=False, add_extra_conv=False):
    Hw = Ww = input_size
    pad = kernel // 2
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, Hw, Ww])
    out_c = mid_c * (num_pools + 1)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, out_c, Hw, Ww])
    W1 = helper.make_tensor('W1', TensorProto.FLOAT, [mid_c, C, 1, 1],
        np.random.randn(mid_c, C, 1, 1).astype(np.float32).tobytes(), raw=True)
    B1 = helper.make_tensor('B1', TensorProto.FLOAT, [mid_c],
        np.zeros(mid_c, dtype=np.float32).tobytes(), raw=True)

    nodes = []
    inits = [W1, B1]

    cv1 = helper.make_node('Conv', ['X', 'W1', 'B1'], ['cv1_out'], kernel_shape=[1, 1])
    nodes.append(cv1)

    pool_outputs = []
    prev_out = 'cv1_out'

    for i in range(num_pools):
        pool_name = f"mp_{i}"
        out_name = f"mp_{i}_out"
        pool_input = prev_out if chained else 'cv1_out'
        mp = helper.make_node(
            'MaxPool', [pool_input], [out_name],
            kernel_shape=[kernel, kernel],
            strides=[1, 1],
            pads=[pad, pad, pad, pad],
            dilations=[1, 1]
        )
        nodes.append(mp)
        pool_outputs.append(out_name)
        prev_out = out_name

    concat_inputs = ['cv1_out'] + pool_outputs
    cat = helper.make_node('Concat', concat_inputs, ['cat_out'], axis=1)
    nodes.append(cat)

    last_out = 'cat_out'

    if add_resize:
        scale_init = helper.make_tensor(
            'rz_scale', TensorProto.FLOAT, [4],
            np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32).tobytes(), raw=True
        )
        inits.append(scale_init)
        rz = helper.make_node(
            'Resize', ['cat_out', '', 'rz_scale'], ['rz_out'],
            mode='nearest', coordinate_transformation_mode='asymmetric'
        )
        nodes.append(rz)
        last_out = 'rz_out'

    if add_extra_conv:
        W2 = helper.make_tensor('W2', TensorProto.FLOAT, [out_c, out_c, 1, 1],
            np.random.randn(out_c, out_c, 1, 1).astype(np.float32).tobytes(), raw=True)
        B2 = helper.make_tensor('B2', TensorProto.FLOAT, [out_c],
            np.zeros(out_c, dtype=np.float32).tobytes(), raw=True)
        inits.extend([W2, B2])
        extra_c = helper.make_node('Conv', [last_out, 'W2', 'B2'], ['Y'], kernel_shape=[1, 1])
        nodes.append(extra_c)
    else:
        # Rename last_out to Y
        alias = helper.make_node('Identity', [last_out], ['Y'])
        nodes.append(alias)

    graph = helper.make_graph(nodes, name, [X], [Y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)], ir_version=8)
    path = os.path.join(out_dir, f"test_npu_{name}.onnx")
    onnx.save(model, path)
    return path


# ===== Generate all test models =====

tests = [
    # (name, num_pools, chained, kernel, input_size, add_resize, add_extra_conv)
    ("sppf_1pool_k5",       1, True,  5, 80, False, False),
    ("sppf_2pool_k5_chained", 2, True,  5, 80, False, False),
    ("sppf_3pool_k5_chained", 3, True,  5, 80, False, False),
    ("sppf_3pool_k5_parallel", 3, False, 5, 80, False, False),
    ("sppf_3pool_k3_chained", 3, True,  3, 80, False, False),
    ("sppf_3pool_k5_416",    3, True,  5, 416, False, False),
    ("sppf_3pool_k5_with_resize", 3, True, 5, 80, True, False),
    ("sppf_3pool_k5_with_conv",  3, True, 5, 80, False, True),
]

for params in tests:
    path = make_sppf_model(*params)
    sz = os.path.getsize(path)
    desc = f"pools={params[1]} {'chained' if params[2] else 'parallel'} k={params[3]} size={params[4]}"
    if params[5]:
        desc += " +Resize"
    if params[6]:
        desc += " +Conv"
    print(f"  {os.path.basename(path):<42} ({sz:>6} B)  {desc}")

print(f"\nDone. {len(tests)} models generated in {out_dir}/")
