#ifndef ORTModel_hpp
#define ORTModel_hpp

#include <opencv2/core/core.hpp>

#include "ORTRunner.hpp"

class ORTModel
{
    public:
        ORTModel(std::shared_ptr<Runner> shpRunner);
        ~ORTModel();

        void run(const cv::Mat& mImage);

    protected:
        virtual void preprocess(cv::Mat& mImage) = 0;
        virtual void postprocess() = 0;

    protected:
        std::shared_ptr<ORTRunner> shpORTRunner;

        std::unordered_map<std::string, std::vector<size_t>> umpIOTensors;
        std::unordered_map<std::string, std::vector<int64_t>> umpIOTensorsShape;

        std::vector<float> inputOrtValues;
        std::vector<std::vector<float>> outputOrtValues;
};

#endif // ORTModel_hpp