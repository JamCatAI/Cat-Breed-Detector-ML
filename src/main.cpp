#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <thread>
#include <mutex>

// 🧠 Load label list
std::vector<std::string> load_labels(const std::string& file_path) {
    std::ifstream file(file_path);
    std::vector<std::string> labels;
    std::string line;
    while (getline(file, line)) labels.push_back(line);
    return labels;
}

// 🧠 Apply softmax to convert logits to probabilities
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> exps(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());

    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(logits[i] - max_logit);
        sum += exps[i];
    }
    for (float& val : exps) val /= sum;
    return exps;
}

// 📊 Get top-K indexes by value
std::vector<int> top_k_indices(const std::vector<float>& values, int k) {
    std::vector<int> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [&](int a, int b) { return values[a] > values[b]; });
    return std::vector<int>(indices.begin(), indices.begin() + k);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./cat_detector <image_path>" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return 1;
    }

    // 🎨 Resize and normalize image
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(224, 224));
    resized.convertTo(resized, CV_32F, 1.0 / 255);

    std::vector<float> input_tensor_values;
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 224; ++y)
            for (int x = 0; x < 224; ++x)
                input_tensor_values.push_back(resized.at<cv::Vec3f>(y, x)[c]);

    // 🧠 Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cat-detector");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4); // 🧵 Multi-threaded

    Ort::Session session(env, "models/cat_breed.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    const char* input_name = session.GetInputName(0, allocator);
    std::array<int64_t, 4> input_shape{1, 3, 224, 224};
    size_t input_tensor_size = 3 * 224 * 224;

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator.GetInfo(), input_tensor_values.data(), input_tensor_size, input_shape.data(), 4);

    const char* output_name = session.GetOutputName(0, allocator);

    // 🕒 Measure inference time
    auto start = std::chrono::high_resolution_clock::now();

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();

    // 🧠 Convert logits to softmax probabilities
    float* raw_output = output_tensors.front().GetTensorMutableData<float>();
    std::vector<float> logits(raw_output, raw_output + 100);
    std::vector<float> probs = softmax(logits);

    // 🎯 Get top-K predictions
    int top_k = 3;
    std::vector<int> top_indices = top_k_indices(probs, top_k);
    auto labels = load_labels("include/labels.txt");

    std::cout << "🐱 Top " << top_k << " Cat Breed Predictions:\n";
    for (int i = 0; i < top_k; ++i) {
        int idx = top_indices[i];
        std::cout << "  " << i + 1 << ". " << labels[idx]
                  << " — " << (probs[idx] * 100.0f) << "%\n";
    }

    std::cout << "\n⚡ Inference Time: " << duration << " ms\n";
    return 0;
}
