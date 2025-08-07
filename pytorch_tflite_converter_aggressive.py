import torch
import os
from models.MU_Net import MU_Net
import onnx
import tensorflow as tf

def load_model(pth_path):
    model = MU_Net(n_channels=3)
    model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    model.eval()
    return model

def export_to_onnx(model, onnx_path, input_shape=(1, 3, 224, 224)):
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=13,
        do_constant_folding=True
    )
    print(f"Exported to ONNX: {onnx_path}")

def convert_onnx_to_tflite_full_int8_quantization(onnx_path, tflite_path, input_shape=(1, 3, 224, 224)):
    """
    Convert ONNX to TFLite with full INT8 quantization
    This should get you closer to 270-300 KB target
    """
    import onnx2tf
    import numpy as np
    
    tf_model_path = "tf_model"
    
    # Convert ONNX to TensorFlow SavedModel
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=tf_model_path,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True
    )
    
    # Create representative dataset with realistic image data range
    def representative_data_gen():
        for _ in range(100):  # Increase samples for better calibration
            # Generate realistic image data in [0, 1] range
            # This is crucial for proper quantization of depthwise convolutions
            sample = np.random.uniform(0.0, 1.0, input_shape).astype(np.float32)
            yield [sample]
    
    # Convert to TFLite with INT8 quantization (less aggressive than full INT8)
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    # Use less aggressive quantization to avoid depthwise conv issues
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Keep input/output as float32 to avoid depthwise conv calibration issues
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    # Don't force input/output to INT8 - this causes issues with depthwise convs
    # converter.inference_input_type = tf.int8  
    # converter.inference_output_type = tf.int8  
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the quantized model
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"INT8 quantization completed: {tflite_path}")

def convert_onnx_to_tflite_robust_quantization(onnx_path, tflite_path, input_shape=(1, 3, 224, 224)):
    """
    Convert ONNX to TFLite with robust quantization for models with depthwise convolutions
    """
    import onnx2tf
    import numpy as np
    
    tf_model_path = "tf_model"
    
    # Convert ONNX to TensorFlow SavedModel
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=tf_model_path,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True
    )
    
    # Create representative dataset with normalized image data
    def representative_data_gen():
        for _ in range(200):  # More samples for stable calibration
            # Generate data that mimics actual image preprocessing
            # Use ImageNet-like normalization
            sample = np.random.normal(0.0, 1.0, input_shape).astype(np.float32)
            # Clip to reasonable range
            sample = np.clip(sample, -3.0, 3.0)
            yield [sample]
    
    # Convert to TFLite with hybrid quantization (safer for depthwise convs)
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    # Use hybrid quantization - weights are INT8, activations stay float
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Allow both INT8 and regular ops for compatibility
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the quantized model
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"Robust quantization completed: {tflite_path}")

def convert_onnx_to_tflite_dynamic_quantization(onnx_path, tflite_path):
    """
    Convert ONNX to TFLite with dynamic quantization only
    This reduces bit precision of weights without using any dataset
    """
    import onnx2tf
    
    tf_model_path = "tf_model"
    
    # Convert ONNX to TensorFlow SavedModel
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=tf_model_path,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True
    )
    
    # Convert to TFLite with dynamic quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    # Enable dynamic quantization - this reduces bit precision of weights
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the quantized model
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"Dynamic quantization completed: {tflite_path}")

def verify_quantized_model(tflite_path, input_shape=(1, 3, 224, 224)):
    """
    Verify the quantized TFLite model
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n=== Model Details ===")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output dtype: {output_details[0]['dtype']}")
    
    # Test inference with float32 input (dynamic quantization keeps input/output as float32)
    test_input = torch.randn(*input_shape).numpy().astype('float32')
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"✅ Test inference successful")
    print(f"Output shape: {output_data.shape}")
    
    # Model size
    model_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
    print(f"Model size: {model_size:.2f} MB")

if __name__ == "__main__":
    pth_model_path = "models/MU_Net_distilled.pth"
    onnx_model_path = "models/MU_Net.onnx"
    tflite_model_path = "models/MU_Net_quantized.tflite"
    tflite_robust_path = "models/MU_Net_quantized_robust.tflite"

    # Ensure output directory exists
    os.makedirs("models", exist_ok=True)
    
    print("Starting quantization pipeline...")
    
    # Step 1: Load PyTorch model and export to ONNX
    print("1. Loading PyTorch model...")
    model = load_model(pth_model_path)
    
    print("2. Exporting to ONNX...")
    export_to_onnx(model, onnx_model_path)
    
    # Step 3: Try robust quantization first (safer for depthwise convolutions)
    print("3. Converting to TFLite with robust quantization...")
    try:
        convert_onnx_to_tflite_robust_quantization(onnx_model_path, tflite_robust_path)
        print("4. Verifying robust quantized model...")
        verify_quantized_model(tflite_robust_path)
        print(f"\n✅ Robust quantization completed successfully!")
        print(f"Quantized model saved to: {tflite_robust_path}")
    except Exception as e:
        print(f"❌ Robust quantization failed: {e}")
        print("Falling back to dynamic quantization...")
        
        # Fallback to dynamic quantization
        try:
            convert_onnx_to_tflite_dynamic_quantization(onnx_model_path, tflite_model_path)
            print("4. Verifying dynamic quantized model...")
            verify_quantized_model(tflite_model_path)
            print(f"\n✅ Dynamic quantization completed!")
            print(f"Quantized model saved to: {tflite_model_path}")
        except Exception as e2:
            print(f"❌ Dynamic quantization also failed: {e2}")
            print("Consider:")
            print("1. Using a different model architecture without depthwise convolutions")
            print("2. Model pruning before conversion")
            print("3. Reducing model architecture complexity")
            exit(1)
    
    # Try aggressive INT8 quantization as well (if robust worked)
    try:
        print("\n5. Attempting aggressive INT8 quantization...")
        convert_onnx_to_tflite_full_int8_quantization(onnx_model_path, tflite_model_path)
        print("6. Verifying aggressive quantized model...")
        verify_quantized_model(tflite_model_path)
        print(f"\n✅ Aggressive INT8 quantization also completed!")
        print(f"Aggressive model saved to: {tflite_model_path}")
    except Exception as e:
        print(f"⚠️  Aggressive INT8 quantization failed (expected): {e}")
        print("Using the robust quantization result instead.")
    
    print("\nTarget size: 270-300 KB")
    print("If size is still too large, consider:")
    print("1. Model pruning before conversion")
    print("2. Using 16-bit quantization instead")
    print("3. Reducing model architecture complexity")