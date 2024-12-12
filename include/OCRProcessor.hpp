#ifndef OCRPROCESSOR_HPP
#define OCRPROCESSOR_HPP

#include<string>

class OCRProcessor {
public:
	virtual ~OCRProcessor() = default;
	virtual std::string processImage(const std::string& imagePath) = 0;
};

#endif