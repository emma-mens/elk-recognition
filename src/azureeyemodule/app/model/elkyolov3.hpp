// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

#include "azureeyemodel.hpp"


namespace model {

// Every new model is created as a C++ class that extends from AzureEyeModel.
//
// AzureEyeModel is an abstract base class. You must implement certain methods
// in your subclass to instantiate it.
class ElkYoloV3Model : public AzureEyeModel
{
public:
    /**
     * Constructor for our new class.
     *
     * We take the following arguments:
     * @param labelfile:   Path to a file that contains a series of labels for the classes (one on each row).
     * @param modelfpaths: The file paths to the model file(s). Your G-API graph can include more than just one model file if you'd like.
     *                     See the OCR model for an example.
     * @param mvcmd:       This is the Myriad X VPU's firmware file. We need to pass this in to restart the VPU.
     * @param videofile:   If non-empty, this should be a file where you want to save the output video of your model. The device will fill up
     *                     quite fast if you enable this (which is done via the command line)!
     * @param resolution:  The resolution mode of the camera.
     */
    ElkYoloV3Model(const std::string &labelfile, const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution);

    /**
     * This method is an override from the abstract parent class. It must be implemented to instantiate this class.
     *
     * This is the method that gets called to start streaming inferences from your network. This method
     * should initialize the accelerator, construct the G-API graph, compile the graph, then run the graph.
     * It should check periodically (ideally every frame) for this->restarting, which is a boolean flag
     * that will be set in the parent class if the network is ever told to stop running.
     *
     * It takes a pipeline object as a work-around for an obscure bug in the Intel Myriad X model compiler tool
     * that we run if you give it an OpenVINO IR model rather than a .blob model. Hopefully one day, we can remove
     * this parameter. But you don't have to worry about it. It's already taken care of for you by main.cpp.
     *
     * Technically, this is the only method you need to implement (other than the constructor). Typically, you will
     * want to have several private methods to break down this otherwise monolithic one.
     */
    void run(cv::GStreamingCompiled* pipeline) override;

private:
    /** Where the label file is. */
    std::string labelfpath;

    /** The list of classes that we detect. */
    std::vector<std::string> class_labels;

    /** Just a simple method that prints out some useful information for debugging. */
    void log_parameters() const;

    /** Here's the method that compiles the G-API graph. It's where we describe the graph itself. */
    cv::GStreamingCompiled compile_cv_graph() const;

    /** Here's the method that pulls data through the compiled G-API graph. */
    bool pull_data(cv::GStreamingCompiled &pipeline);

    /** Here's where we compose the IoT messages and log messages. */
    void handle_inference_output(const cv::optional<cv::Mat> &out_nn, const cv::optional<int64_t> &out_nn_ts,
                                                                const cv::optional<int64_t> &out_nn_seqno, const cv::optional<std::vector<float>> &out_coverages,
                                                                const cv::Size &size,
                                                                cv::Mat &last_nn, std::vector<float> last_coverages);

    /** Here's where we compose the RTSP streams. */
    void handle_bgr_output(cv::optional<cv::Mat> &out_raw_mat, cv::Mat &last_raw_mat, const cv::Mat &last_nn);
};

} // namespace model
