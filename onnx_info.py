import onnx

onnx_path = r"E:/ddsp-pytorch-master/DDSP/ckpt/sax/sax_48k_last.onnx"

model = onnx.load(onnx_path)

print("=== Inputs ===")
for inp in model.graph.input:
    dims = [d.dim_param if d.dim_param else d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f" - {inp.name} {dims}")

print("\n=== Outputs ===")
for out in model.graph.output:
    dims = [d.dim_param if d.dim_param else d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f" - {out.name} {dims}")
