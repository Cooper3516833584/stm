"""Generate tests for grouped convolution and dynamic Slice on VSINPU."""
import onnx, numpy as np, os
from onnx import helper, TensorProto

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# ================================================================
# Test 1: Depthwise Conv (groups=in_channels=out_channels)
# This simulates /model.10/m/m.0/attn/pe/conv/Conv groups=128
# ================================================================
C = 64
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, 32, 32])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, C, 32, 32])
# Depthwise: groups=C, each input channel convolved separately
W = helper.make_tensor('W', TensorProto.FLOAT, [C, 1, 3, 3],
    np.random.randn(C, 1, 3, 3).astype(np.float32).tobytes(), raw=True)
B = helper.make_tensor('B', TensorProto.FLOAT, [C],
    np.zeros(C, dtype=np.float32).tobytes(), raw=True)
dw = helper.make_node('Conv', ['X', 'W', 'B'], ['Y'],
    kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=C)
graph = helper.make_graph([dw], 'dwconv', [X], [Y], [W, B])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)], ir_version=8)
onnx.save(model, os.path.join(out_dir, "test_npu_dwconv.onnx"))
print(f"Test 1: Depthwise Conv (groups={C}) -> test_npu_dwconv.onnx")

# ================================================================
# Test 2: SPPF + Depthwise Conv (combine the two patterns)
# ================================================================
C, H, W = 64, 80, 80
mid_c = 32
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, H, W])
out_c = mid_c * 4

cv1 = helper.make_node('Conv', ['X', 'W1', 'B1'], ['cv1_out'], kernel_shape=[1, 1])
m   = helper.make_node('MaxPool', ['cv1_out'], ['m_out'],  kernel_shape=[5,5], strides=[1,1], pads=[2,2,2,2], dilations=[1,1])
m_1 = helper.make_node('MaxPool', ['m_out'],   ['m1_out'], kernel_shape=[5,5], strides=[1,1], pads=[2,2,2,2], dilations=[1,1])
m_2 = helper.make_node('MaxPool', ['m1_out'],  ['m2_out'], kernel_shape=[5,5], strides=[1,1], pads=[2,2,2,2], dilations=[1,1])
cat = helper.make_node('Concat', ['cv1_out', 'm_out', 'm1_out', 'm2_out'], ['cat_out'], axis=1)

# Add a depthwise conv after SPPF
W_dw = helper.make_tensor('W_dw', TensorProto.FLOAT, [out_c, 1, 3, 3],
    np.random.randn(out_c, 1, 3, 3).astype(np.float32).tobytes(), raw=True)
B_dw = helper.make_tensor('B_dw', TensorProto.FLOAT, [out_c],
    np.zeros(out_c, dtype=np.float32).tobytes(), raw=True)
dw_c = helper.make_node('Conv', ['cat_out', 'W_dw', 'B_dw'], ['Y'],
    kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=out_c)

Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, out_c, H, W])
W1 = helper.make_tensor('W1', TensorProto.FLOAT, [mid_c, C, 1, 1],
    np.random.randn(mid_c, C, 1, 1).astype(np.float32).tobytes(), raw=True)
B1 = helper.make_tensor('B1', TensorProto.FLOAT, [mid_c],
    np.zeros(mid_c, dtype=np.float32).tobytes(), raw=True)

graph = helper.make_graph([cv1, m, m_1, m_2, cat, dw_c], 'sppf_dwconv',
    [X], [Y], [W1, B1, W_dw, B_dw])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)], ir_version=8)
onnx.save(model, os.path.join(out_dir, "test_npu_sppf_dwconv.onnx"))
print(f"Test 2: SPPF + Depthwise Conv -> test_npu_sppf_dwconv.onnx")

# ================================================================
# Test 3: Dynamic Slice (params come from Shape/Gather, not const)
# Simulating the 2 dynamic slices in the VSINPU model
# ================================================================
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 64, 32, 32])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 32, 32, 32])

# Simulate starts=[0], ends=[32], axes=[1], steps=[1] with Constant nodes
starts = helper.make_node('Constant', [], ['starts'],
    value=helper.make_tensor('starts_val', TensorProto.INT64, [1],
        np.array([0], dtype=np.int64).tobytes(), raw=True))
ends = helper.make_node('Constant', [], ['ends'],
    value=helper.make_tensor('ends_val', TensorProto.INT64, [1],
        np.array([32], dtype=np.int64).tobytes(), raw=True))
axes = helper.make_node('Constant', [], ['axes'],
    value=helper.make_tensor('axes_val', TensorProto.INT64, [1],
        np.array([1], dtype=np.int64).tobytes(), raw=True))
steps = helper.make_node('Constant', [], ['steps'],
    value=helper.make_tensor('steps_val', TensorProto.INT64, [1],
        np.array([1], dtype=np.int64).tobytes(), raw=True))
slc = helper.make_node('Slice', ['X', 'starts', 'ends', 'axes', 'steps'], ['sliced'])
ident = helper.make_node('Identity', ['sliced'], ['Y'])

graph = helper.make_graph([starts, ends, axes, steps, slc, ident],
    'slice_const', [X], [Y], [])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)], ir_version=8)
onnx.save(model, os.path.join(out_dir, "test_npu_slice_const.onnx"))
print(f"Test 3: Slice with Constant nodes -> test_npu_slice_const.onnx")

# ================================================================
# Test 4: Gather + Shape (used in road model, but test on tree too)
# ================================================================
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 64, 16, 16])
Y = helper.make_tensor_value_info('Y', TensorProto.INT64, [1])
shape = helper.make_node('Shape', ['X'], ['shape_out'])
gather_idx = helper.make_node('Constant', [], ['gather_idx'],
    value=helper.make_tensor('gidx', TensorProto.INT64, [1],
        np.array([2], dtype=np.int64).tobytes(), raw=True))
gather = helper.make_node('Gather', ['shape_out', 'gather_idx'], ['Y'], axis=0)
graph = helper.make_graph([shape, gather_idx, gather], 'shape_gather', [X], [Y], [])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)], ir_version=8)
onnx.save(model, os.path.join(out_dir, "test_npu_gather.onnx"))
print(f"Test 4: Shape + Gather -> test_npu_gather.onnx")

# ================================================================
# Test 5: Grouped Conv (not depthwise, groups=64)
# ================================================================
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 128, 16, 16])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 128, 16, 16])
W = helper.make_tensor('W', TensorProto.FLOAT, [128, 2, 3, 3],
    np.random.randn(128, 2, 3, 3).astype(np.float32).tobytes(), raw=True)
B = helper.make_tensor('B', TensorProto.FLOAT, [128],
    np.zeros(128, dtype=np.float32).tobytes(), raw=True)
gconv = helper.make_node('Conv', ['X', 'W', 'B'], ['Y'],
    kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=64)
graph = helper.make_graph([gconv], 'gconv64', [X], [Y], [W, B])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)], ir_version=8)
onnx.save(model, os.path.join(out_dir, "test_npu_gconv64.onnx"))
print(f"Test 5: Grouped Conv groups=64 -> test_npu_gconv64.onnx")

print(f"\nDone. Files in {out_dir}/:")
for f in sorted(os.listdir(out_dir)):
    if f.startswith("test_npu_dw") or f.startswith("test_npu_sppf_dw") or \
       f.startswith("test_npu_slice_const") or f.startswith("test_npu_gather") or \
       f.startswith("test_npu_gconv"):
        sz = os.path.getsize(os.path.join(out_dir, f))
        print(f"  {f} ({sz} bytes)")
