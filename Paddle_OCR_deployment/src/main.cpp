#include "PaddleOCRWrapper.hpp"
#include <iostream>

int main() {
    const std::string modelPath = "../../models/model.onnx";
    const std::string testImage = "../../data/1.jpg";

    PaddleOCRWrapper ocrProcessor(modelPath);
    std::string result = ocrProcessor.processImage(testImage);

    std::cout << "OCR Result: " << result << std::endl;
    return 0;
}
