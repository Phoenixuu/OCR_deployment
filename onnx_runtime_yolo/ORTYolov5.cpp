#include "ORTYoloV5.hpp"

#include <iostream>

ORTYoloV5::ORTYoloV5(std::shared_ptr<ORTRunner> shpORTRunner, std::function<void (const std::vector<stObject_t>&)> fnCallback)
    : ORTModel(shpORTRunner), m_fnCallback(fnCallback)
{
    m_iWidthModel = umpIOTensorsShape[m_strInputName].d[3];
    m_iHeightModel = umpIOTensorsShape[m_strInputName].d[2];
    
    m_iOutputWidth = umpIOTensorsShape[m_strOutputName].d[2];
    m_iOutputHeight = umpIOTensorsShape[m_strOutputName].d[1];
}

ORTYoloV5::~ORTYoloV5()
{
    
}

void ORTYoloV5::setLabels(std::string& strFileLabel)
{

}

void ORTYoloV5::setScoreThreshold(float fScoreThreshold)
{
    m_fScoreThreshold = fScoreThreshold;
}

void ORTYoloV5::setNMSThreshold(float fNMSThreshold)
{
    m_fNMSThreshold = fNMSThreshold;
}

void ORTYoloV5::preprocess(cv::Mat& mImage)
{
    m_iInputWidth = mImage.cols;
    m_iInputHeight = mImage.rows;

    m_fRatioWidth = 1.0f / (m_iWidthModel / static_cast<float>(m_iInputWidth));
    m_fRatioHeight = 1.0f / (m_iHeightModel / static_cast<float>(m_iInputHeight));

    cv::Mat mInput;
    cv::resize(mImage, mInput, cv::Size(m_iModelWidth, m_iModelHeight), 0, 0, cv::INTER_LINEAR);

    cv::cvtColor(mInput, mInput, cv::COLOR_BGR2RGB);
    cv::Mat mFloat;
    mInput.convertTo(mFloat, CV_32FC3, 1.f / 255.f);

    cv::Mat mChannels[3];
    cv::split(mFloat, mChannels);

    inputOrtValues.clear();
    for (auto& channel : mChannels)
    {
        std::vector<float> fVec(channel.begin<float>(), channel.end<float>());
        outputOrtValues.insert(inputOnnxValues.end(), fVec.begin(), fVec.end());
    }
}

void ORTYoloV5::postprocess()
{
    std::vector<float> fObjectArray;
    fObjectArray = this->outputOnnxValues[0];
    std::vector<int64_t> iDims = this->shpRunner->m_outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<stOutput_t> stObjects;
    float fMaxWidth = this->uiInputWidth - 1;
    float fMaxHeight = this->uiInputHeight - 1;

    for (int i = 0; i < fObjectArray.size(); i += iDims[2])
    {
        if (fObjectArray[i + 4] > m_fThreshold)
        {
            stOutput_t stObject;
            int iMaxID = 0;
            float fMaxConfedence = -1;
            for (int index = i + 5; index < i + iDims[2]; index++)
            {
                if (fMaxConfedence < fObjectArray[index])
                {
                    fMaxConfedence = fObjectArray[index];
                    iMaxID = index - i - 5;
                }
            }
            int iXCenter = int(fObjectArray[i]);
            int iYCenter = int(fObjectArray[i + 1]);
            int iW = int(fObjectArray[i + 2]);
            int iH = int(fObjectArray[i + 3]);
            int iX1 = ((iXCenter - iW / 2) - this->fPaddingX) / this->fDetScale;
            int iY1 = ((iYCenter - iH / 2) - this->fPaddingY) / this->fDetScale;
            iW = iW / this->fDetScale;
            iH = iH / this->fDetScale;
            int iX2 = iX1 + iW;
            int iY2 = iY1 + iH;
            IN_RANGE(iX1, 0, fMaxWidth);
            IN_RANGE(iX2, 0, fMaxWidth);
            IN_RANGE(iY1, 0, fMaxHeight);
            IN_RANGE(iY2, 0, fMaxHeight);

            stObject.riBox = cv::Rect(cv::Point(iX1, iY1), cv::Point(iX2, iY2));
            stObject.fProb = fObjectArray[i + 4];
            stObject.iID = iMaxID;
            stObjects.push_back(stObject);
        }
    }

    std::sort(stObjects.begin(), stObjects.end(), 
                [] (const stOutput_t& a, const stOutput_t& b)
                {
                    return a.fProb > b.fProb;
                });

    std::vector<stOutput_t> stOutputs;
    while (stObjects.size() > 0)
    {
        stOutput_t stOutput = stObjects[0];
        stOutputs.push_back(stOutput);
        stObjects.erase(stObjects.begin());
        float fArea = stOutput.riBox.area();
        for (int i = 0; i < stObjects.size(); i++)
        {
            stOutput_t stOutputI = stObjects[i];
            if (stOutputI.iID != stOutput.iID)
            {
                continue;
            }
            float fInter = (stOutput.riBox & stOutputI.riBox).area();
            float fAreaI = stOutputI.riBox.area();
            float fOvr = fInter / (fArea + fAreaI - fInter);
            if (fOvr >= m_fNMSThreshold)
            {
                stObjects.erase(stObjects.begin() + i);
                i--;
            }
        }
    }
    m_fnModelCallback(stOutputs);
}
