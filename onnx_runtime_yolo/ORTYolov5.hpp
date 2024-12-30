#ifndef ORTYoloV5_hpp
#define ORTYoloV5_hpp

#include <functional>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ORTModel.hpp"
#include "Types.hpp"

class ORTYoloV5 : public ORTModel
{
    public:
        ORTYoloV5(std::shared_ptr<ORTRunner> shpORTRunner, std::function<void (const std::vector<stObject_t>&)> fnCallback);
        ~ORTYoloV5();

        void setLabels(std::string& strFileLabel);

        void setScoreThreshold(float fScoreThreshold);
        void setNMSThreshold(float fNMSThreshold);

    protected:
        void preprocess(cv::Mat& mImage) override;
        void postprocess() override;

    private:
        std::string m_strInputName = "images";
        std::string m_strOutputName = "output";

        int m_iNumClasses = 2;
        int m_iWidthModel, m_iHeightModel;
        int m_iOutputWidth, m_iOutputHeight;

        int m_iInputWidth, m_iInputHeight;
        float m_fRatioWidth, m_fRatioHeight;

        float m_fScoreThreshold = 0.5f;
        float m_fNMSThreshold = 0.4f;

        std::function<void (const std::vector<stObject_t>&)> m_fnCallback;
};

#endif // ORTYoloV5_hpp