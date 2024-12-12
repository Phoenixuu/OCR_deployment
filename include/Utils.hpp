#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <string>

namespace Utils {
    cv::Mat readImage(const std::string& filepath);

    void logMessage(const std::string& message);

    void checkError(bool condition, const std::string& errorMessage);
}

#endif // UTILS_HPP
