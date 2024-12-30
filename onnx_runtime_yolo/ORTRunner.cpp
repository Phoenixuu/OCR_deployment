#include "ORTRunner.hpp"

ORTRunner::ORTRunner(const std::string& strModelPath)
{
    m_env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, strModelPath.c_str());
    m_sessionOptions = Ort::SessionOptions();
    m_session = new Ort::Session(m_env, strModelPath.c_str(), m_sessionOptions);

    // _inputName = std::move(m_session->GetInputNameAllocated(0, m_ortAllocator));
    // m_inputNames.push_back(_inputName.get());

    // size_t numOutputNodes = m_session->GetOutputCount();
    // for (int i = 0; i < numOutputNodes; i++)
    // {
    //     _outputName = std::move(m_session->GetOutputNameAllocated(i, m_ortAllocator));
    //     m_outputNames.push_back(_outputName.get());
    // }

    m_ortMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
}

ORTRunner::~ORTRunner()
{
    delete m_session;
}

void ORTRunner::runModel(std::vector<float>& inputOrtValues, std::vector<std::vector<float>>& outputOrtValues)
{
    std::vector<Ort::Value> ortInputTensors;

    outputOrtValues.clear();

    // Ort::MemoryInfo ortMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    
    size_t inputTensorSize = vectorProduct(m_inputTensorShape);

    ortInputTensors.push_back(
        Ort::Value::CreateTensor<float>(m_ortMemoryInfo, 
                                        inputOrtValues.data(), 
                                        inputTensorSize, 
                                        m_inputTensorShape.data(), 
                                        m_inputTensorShape.size()));
    
    m_outputTensors = m_session->Run(Ort::RunOptions{nullptr}, 
                                    m_inputNames.data(), 
                                    ortInputTensors.data(), 
                                    1, 
                                    m_outputNames.data(), 
                                    m_outputNames.size());

    for (auto& i : m_outputTensors)
    {
        auto* rawOutput = i.GetTensorData<float>();
        std::vector<int64_t> outputShape = i.GetTensorTypeAndShapeInfo().GetShape();
        size_t outputTensorSize = vectorProduct(outputShape);
        std::vector<float> outputTensor(rawOutput, rawOutput + outputTensorSize);
        outputOrtValues.push_back(outputTensor);
    }
}
