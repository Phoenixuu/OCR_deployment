#include "Utils.hpp"
#include <iostream>

namespace Utils {
    cv::Mat readImage(const std::string& filepath) {
        cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Failed to read image from: " + filepath);
        }
        return image;
    }

    void logMessage(const std::string& message) {
        std::cout << "[LOG]: " << message << std::endl;
    }

    void checkError(bool condition, const std::string& errorMessage) {
        if (!condition) {
            throw std::runtime_error(errorMessage);
        }
    }
}
