#include "PaddleOCRWrapper.hpp"
#include <iostream>
#include "Utils.hpp" // Sử dụng các hàm tiền xử lý ảnh

PaddleOCRWrapper::PaddleOCRWrapper(const std::string& modelDir) 
    : modelDir(modelDir) {
    // Khởi tạo mô hình
    initializeModel(modelDir);
}

void PaddleOCRWrapper::initializeModel(const std::string& modelDir) {
    try {
        // Load PaddleOCR model và các thành phần cần thiết
        ocrModel = std::make_unique<paddle_infer::Predictor>(Utils::loadPaddleModel(modelDir));
        std::cout << "Model loaded successfully from: " << modelDir << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing model: " << e.what() << std::endl;
    }
}

std::string PaddleOCRWrapper::detect(const cv::Mat& image) {
    // Tiền xử lý ảnh
    cv::Mat preprocessedImage = Utils::preprocessImage(image);

    // Chuyển đổi ảnh thành dữ liệu đầu vào cho mô hình
    auto inputTensor = Utils::convertMatToTensor(preprocessedImage);

    // Thực hiện dự đoán
    std::vector<std::string> ocrResult;
    try {
        ocrModel->Run(inputTensor);
        ocrResult = Utils::parseOCRResult(ocrModel->GetOutputTensor());
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
    }

    // Ghép kết quả OCR thành chuỗi
    std::string result;
    for (const auto& line : ocrResult) {
        result += line + "\n";
    }

    return result;
}

PaddleOCRWrapper::~PaddleOCRWrapper() {
    // Giải phóng tài nguyên nếu cần
    std::cout << "PaddleOCRWrapper is destroyed." << std::endl;
}
