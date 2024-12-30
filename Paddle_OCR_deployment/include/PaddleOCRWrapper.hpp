#ifndef PADDLEOCRWRAPPER_HPP
#define PADDLEOCRWRAPPER_HPP

#include "OCRProcessor.hpp"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <memory>

class PaddleOCRWrapper : public OCRProcessor {
public:
	PaddleOCRWrapper(const std::string& modelPath);
	std::string processImage(const std::string& imagePath) override;

private:
	Ort::Env env_;
	Ort::Session session_;
};

#endif