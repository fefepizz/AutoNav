// Cargo.toml
/*
[package]
name = "mu_net_inference"
version = "0.1.0"
edition = "2021"

[dependencies]
microflow = "0.1.0"
microflow-macros = "0.1.0"

# Optional: for no_std embedded targets
# [dependencies.microflow]
# version = "0.1.0"
# default-features = false

[profile.release]
opt-level = "z"     # Optimize for size
lto = true          # Link-time optimization
codegen-units = 1   # Better optimization
panic = "abort"     # Reduce binary size
*/

use microflow::model;

// Define your model struct and apply the microflow model macro
// This will generate the predict() method at compile time
#[model("models/MU_Net_quantized_int8.tflite")]
struct MUNetModel;

fn main() {
    // Example input data - adjust dimensions based on your model
    // For a typical image model with shape (1, 3, 224, 224)
    let input_data: [[[f32; 224]; 224]; 3] = [[[0.0; 224]; 224]; 3];
    
    // Perform inference
    let prediction = MUNetModel::predict(&input_data);
    
    println!("Prediction result: {:?}", prediction);
}

// Alternative example with flattened input for easier data handling
#[cfg(feature = "flattened_input")]
mod flattened_example {
    use microflow::model;
    use super::*;
    
    // If your model expects flattened input
    #[model("models/MU_Net_quantized_int8.tflite")]
    struct MUNetModelFlat;
    
    pub fn run_flattened_inference() {
        // Calculate total input size: 1 * 3 * 224 * 224 = 150,528
        const INPUT_SIZE: usize = 3 * 224 * 224;
        let input_data: [f32; INPUT_SIZE] = [0.0; INPUT_SIZE];
        
        let prediction = MUNetModelFlat::predict(&input_data);
        println!("Flattened prediction: {:?}", prediction);
    }
}

// For embedded/no_std environments
#[cfg(feature = "embedded")]
mod embedded_example {
    #![no_std]
    #![no_main]
    
    use microflow::model;
    use panic_halt as _; // Panic handler for embedded
    
    #[model("models/MU_Net_quantized_int8.tflite")]
    struct MUNetEmbedded;
    
    #[no_mangle]
    pub extern "C" fn main() -> ! {
        // Initialize your input data (from sensors, camera, etc.)
        let input_data: [[[f32; 224]; 224]; 3] = [[[0.0; 224]; 224]; 3];
        
        // Perform inference
        let prediction = MUNetEmbedded::predict(&input_data);
        
        // Process prediction results
        // (send via UART, control actuators, etc.)
        
        loop {
            // Main application loop
        }
    }
}

// Helper functions for data preprocessing
mod preprocessing {
    /// Convert RGB image bytes to the format expected by the model
    pub fn preprocess_image_data(
        image_bytes: &[u8], 
        width: usize, 
        height: usize
    ) -> [[[f32; 224]; 224]; 3] {
        let mut processed = [[[0.0f32; 224]; 224]; 3];
        
        // Resize and normalize your input data here
        // This is a placeholder - implement based on your preprocessing needs
        for c in 0..3 {
            for h in 0..224 {
                for w in 0..224 {
                    // Example: normalize pixel values to [0, 1] or [-1, 1]
                    if h < height && w < width {
                        let pixel_idx = (h * width + w) * 3 + c;
                        if pixel_idx < image_bytes.len() {
                            processed[c][h][w] = image_bytes[pixel_idx] as f32 / 255.0;
                        }
                    }
                }
            }
        }
        
        processed
    }
    
    /// Post-process model output for classification
    pub fn postprocess_classification(output: &[f32]) -> (usize, f32) {
        let mut max_idx = 0;
        let mut max_val = output[0];
        
        for (idx, &val) in output.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }
        
        (max_idx, max_val)
    }
    
    /// Apply softmax to get probabilities
    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        
        exp_logits.iter().map(|&x| x / sum_exp).collect()
    }
}

// Example usage with real data processing
#[cfg(feature = "example_usage")]
mod example_usage {
    use super::*;
    use preprocessing::*;
    
    pub fn run_complete_inference_example() {
        // Simulate loading image data (replace with actual image loading)
        let image_width = 640;
        let image_height = 480;
        let image_bytes = vec![128u8; image_width * image_height * 3]; // Dummy gray image
        
        // Preprocess the image
        let input_data = preprocess_image_data(&image_bytes, image_width, image_height);
        
        // Run inference
        let raw_output = MUNetModel::predict(&input_data);
        
        // Post-process results
        let (predicted_class, confidence) = postprocess_classification(&raw_output);
        println!("Predicted class: {}, Confidence: {:.4}", predicted_class, confidence);
        
        // Get probabilities
        let probabilities = softmax(&raw_output);
        println!("Class probabilities: {:?}", probabilities);
    }
}

// Benchmarking and performance testing
#[cfg(feature = "benchmark")]
mod benchmark {
    use super::*;
    use std::time::Instant;
    
    pub fn benchmark_inference() {
        let input_data: [[[f32; 224]; 224]; 3] = [[[0.5; 224]; 224]; 3];
        
        // Warm up
        for _ in 0..10 {
            let _ = MUNetModel::predict(&input_data);
        }
        
        // Benchmark
        let num_runs = 100;
        let start = Instant::now();
        
        for _ in 0..num_runs {
            let _ = MUNetModel::predict(&input_data);
        }
        
        let duration = start.elapsed();
        let avg_time = duration / num_runs;
        
        println!("Average inference time: {:?}", avg_time);
        println!("FPS: {:.2}", 1.0 / avg_time.as_secs_f64());
    }
}