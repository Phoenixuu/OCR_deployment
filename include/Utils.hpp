#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <string>

namespace Utils {
    // Hàm để đọc hình ảnh từ file
    cv::Mat readImage(const std::string& filepath);

    // Hàm để ghi log
    void logMessage(const std::string& message);

    // Hàm để kiểm tra lỗi
    void checkError(bool condition, const std::string& errorMessage);
}

#endif // UTILS_HPP
