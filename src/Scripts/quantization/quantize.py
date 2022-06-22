from os.path import join, abspath, dirname
from onnxruntime.quantization import quantize_dynamic, QuantType

# import onnx

# Snippet extraido de mi compañero Jorge Ruiz Gómez en:
# https://github.com/JorgeRuizDev/SpotMyFM/blob/main/Ludwig/mir-backend/models/quantize.py


def quantize_onnx_model(onnx_model_path):
    quantized_model_path = onnx_model_path.replace(".onnx", "_quantized.onnx")

    # onnx_opt_model = onnx.load(onnx_model_path)
    quantize_dynamic(
        onnx_model_path, quantized_model_path, weight_type=QuantType.QUInt8
    )

    print(f"quantized model saved to:{quantized_model_path}")


if __name__ == "__main__":

    path_to_model = "../../../models/WLASL_9.onnx"
    path = abspath(join(dirname(__file__), path_to_model))

    print(f"Quantizing model on path {path}")
    quantize_onnx_model(path)
