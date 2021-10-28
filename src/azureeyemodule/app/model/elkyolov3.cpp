// Put this in azureeyemodule/app/model/unetsemseg.cpp

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <assert.h>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

// Third party includes
#include <opencv2/gapi/mx.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/highgui.hpp>

// Local includes
#include "azureeyemodel.hpp"
#include "elkyolov3.hpp"
#include "../device/device.hpp"
#include "../iot/iot_interface.hpp"
#include "../kernels/elk_yolo_kernels.hpp"
#include "../streaming/rtsp.hpp"
#include "../util/helper.hpp"
#include "../util/labels.hpp"

namespace model {

/** Declare a SemanticSegmentationUNet network type. Takes one matrix and outputs another. */
G_API_NET(ElkYoloV3, <cv::GMat(cv::GMat)>, "com.microsoft.elk-yolov3");

// Here is our constructor. We don't need to do anything, as it is all taken care of by the parent class.
ElkYoloV3Model::ElkYoloV3Model(const std::string &labelfile, const std::vector<std::string> &modelfpaths, const std::string &mvcmd, const std::string &videofile, const cv::gapi::mx::Camera::Mode &resolution)
    : AzureEyeModel{ modelfpaths, mvcmd, videofile, resolution }, labelfpath(labelfile), class_labels({})
{
}

// Here is the run method - the only method we technically need to implement.
// We will implement several methods though, so that you can see how we typically
// do it. Feel free to do whatever you want though.
void ElkYoloV3Model::run(cv::GStreamingCompiled *pipeline)
{
    // The model is expected to run until this->restarting flips to true.
    // So we should check this flag, but we'll get to that later, in the G-API
    // graph loop. Here, we run forever and break manually if our pull_data() method
    // says to.
    while (true)
    {
        // We need to block until the Myriad X VPU is up and running. You should call this method here.
        this->wait_for_device();

        // Because this class has a possible label file, let's load that in.
        label::load_label_file(this->class_labels, this->labelfpath);

        // Let's log what model we are running, so that we can be sure we are running the right model
        // when debugging and examining the logs.
        this->log_parameters();

        // Some boilerplate.
        // Build the camera pipeline with G-API
        *pipeline = this->compile_cv_graph();
        util::log_info("starting the pipeline...");
        pipeline->start();

        // Pull data through the pipeline
        bool ran_out_naturally = this->pull_data(*pipeline);

        if (!ran_out_naturally)
        {
            break;
        }
    }
}

// Here's the simple log_parameters() method. All we do is print some info.
void ElkYoloV3Model::log_parameters() const
{
    std::string msg = "blobs: ";
    for (const auto &blob : this->modelfiles)
    {
        msg += blob + ", ";
    }
    msg.append(", firmware: " + this->mvcmd);
    msg.append(", parser: UnetSemanticSegmentationModel");
    msg.append(", label: " + this->labelfpath);
    msg.append(", classes: {");
    for (const auto &label : this->class_labels)
    {
        msg.append(label).append(", ");
    }
    msg.append("}");

    // Feel free to use the util::log_* methods. You can adjust the logging verbosity
    // in the module twin for this IoT module.
    util::log_info(msg);
}

// Here's where we actually define the G-API graph. It will look pretty similar to the one
// we came up with in the mock eye application.
cv::GStreamingCompiled ElkYoloV3Model::compile_cv_graph() const
{
    // Declare an empty GMat - the beginning of the pipeline.
    // The Percept camera's images will fill this node one
    // frame at a time.
    // In the future, we hope to add support for video as an input source,
    // but for now it is always the camera.
    cv::GMat in;

    // We must preprocess the input frame using the custom Myriad X back-end.
    cv::GMat preproc = cv::gapi::mx::preproc(in, this->resolution);

    // If you want H.264 output, here's how to get it.
    cv::GArray<uint8_t> h264;
    cv::GOpaque<int64_t> h264_seqno;
    cv::GOpaque<int64_t> h264_ts;
    std::tie(h264, h264_seqno, h264_ts) = cv::gapi::streaming::encH264ts(preproc);

    // We have BGR output and H264 output in the same graph.
    // In this case, BGR always must be desynchronized from the main path
    // to avoid internal queue overflow.
    // copy() is required only to maintain the graph contracts
    // (there must be an operation following desync()). No real copy happens.
    //
    // We have briefly covered asynchronous G-API graphs before. We will
    // talk more about this in the pull() method. But for now,
    // understand that the point is that this branch of the pipeline
    // will be asynchronous.
    //
    // This path will just feed directly to the output, but asynchronously
    // from the other nodes.
    cv::GMat raw_input = cv::gapi::copy(cv::gapi::streaming::desync(preproc));

    // Here's another asynchronous path through the graph. This
    // one will go through the neural network.
    cv::GMat network_input = cv::gapi::streaming::desync(preproc);

    // Here's some more boilerplate. This gets you a frame index
    // and a timestamp. You don't need it, but we include it here
    // to show you how to get frame numbers and timestamps.
    cv::GOpaque<int64_t> nn_seqno = cv::gapi::streaming::seqNo(network_input);
    cv::GOpaque<int64_t> nn_ts = cv::gapi::streaming::timestamp(network_input);

    // Here's where we run the network. Again, this path is asynchronous.
    auto nn = cv::gapi::infer<ElkYoloV3>(network_input);

    // Here's where we parse the output of our network with our custom parser code.
    // We'll implement this shortly (and it will look just like the code we used
    // in the mock app).
    cv::GMat parsed_nn;
    cv::GArray<float> predictions;
    std::tie(parsed_nn, predictions) = cv::gapi::streaming::parse_yolov3(nn);

    // Now specify the computation's boundaries
    auto graph = cv::GComputation(cv::GIn(in),                                        // Here's the input
                                  cv::GOut(h264, h264_seqno, h264_ts,                 // main path: H264 (~constant framerate)
                                           raw_input,                                 // desynchronized path: BGR frames, one at a time
                                           nn_seqno, nn_ts, parsed_nn, predictions));   // desynchronized path: Inferences and post-processing

    // Pass the network .blob file in (instead of an OpenVINO IR .xml and .bin file)
    auto networks = cv::gapi::networks(cv::gapi::mx::Params<ElkYoloV3Model>{ this->modelfiles.at(0) });

    // Create the kernels. Notice that we need some custom kernels for the Myriad X, which we get
    // by calling cv::gapi::mx::kernels().
    auto kernels = cv::gapi::combine(cv::gapi::mx::kernels(), cv::gapi::kernels<cv::gapi::streaming::GOCVParseYoloV3>());

    // Compile the graph in streamnig mode, set all the parameters (including the firmware file).
    auto pipeline = graph.compileStreaming(cv::gapi::mx::Camera::params(), cv::compile_args(networks, kernels, cv::gapi::mx::mvcmdFile{ this->mvcmd }));

    // Specify the Azure Percept's Camera as the input to the pipeline.
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::mx::Camera>());

    return pipeline;
}

// This function is where we execute the graph. We run a while loop until the pipeline
// runs out of frames (not something that currently happens, but we are hoping to allow
// for video files as an input source) or until we are told to exit.
// We return whether we should quit from the outer loop or not.
bool ElkYoloV3Model::pull_data(cv::GStreamingCompiled &pipeline)
{
    // Here are all the variables we need for the output nodes.
    // Note that the are wrapped in cv::optional<>.
    // This is crucial: the output of a graph node that
    // is part of a desynchronized graph (one that has desync called in it)
    // is always an optional<>. Optionals MAY or MAY NOT have anything at all.
    // I'll explain this a bit more in a few lines.
    //
    // For now, here are all the output node variables.
    cv::optional<cv::Mat> out_raw_mat;
    cv::optional<cv::Mat> out_nn;
    cv::optional<std::vector<float>> out_predictions;
    cv::optional<std::vector<uint8_t>> out_h264;
    cv::optional<int64_t> out_h264_seqno;
    cv::optional<int64_t> out_h264_ts;
    cv::optional<int64_t> out_nn_ts;
    cv::optional<int64_t> out_nn_seqno;

    // Because the outputs from the desynchronized G-API graph arrive
    // at different times, we cache the latest one each time it arrives.
    cv::Mat last_raw_mat(256, 256, CV_8UC3, cv::Scalar(0, 0, 0)); // Give it some arbitrary dimensions (we'll overwrite it once we get something from the pipeline).
    cv::Mat last_nn(256, 256, CV_8UC3, cv::Scalar(0, 0, 0));      // Ditto
    std::vector<float> last_predictions;

    // If we have a path to a video file, let's open it here.
    std::ofstream ofs;
    if (!this->videofile.empty())
    {
        ofs.open(this->videofile, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
    }

    // Pull the data from the pipeline while it is running
    //
    // Since this graph contains desync() calls, each output node is filled with an optional
    // every time we call pull(), whether the data is ready or not. If it is not ready,
    // the optional will not contain anything useful, so we will have to check each time we want to use it.
    while (pipeline.pull(cv::gout(out_h264, out_h264_seqno, out_h264_ts, out_raw_mat, out_nn_seqno, out_nn_ts, out_nn, out_predictions)))
    {
        // This method is in the super class. No need to worry about it.
        this->handle_h264_output(out_h264, out_h264_ts, out_h264_seqno, ofs);

        // Here's a method for handling the inference outputs. It will compose the IoT messages
        // and log messages based on the output from the network.
        // We'll create this one in this class.
        //
        // This method moves out_nn to last_nn (if there is a value in the optional).
        // So after this method, use last_nn instead of out_nn. We also move out_coverages into last_coverages here.
        auto size = last_raw_mat.size();
        this->handle_inference_output(out_nn, out_nn_ts, out_nn_seqno, out_predictions, size, last_nn, last_predictions);

        // Here's a method for composing our result RTSP stream.
        // We'll create this one in this class.
        //
        // This method moves out_raw_mat into last_raw_mat.
        this->handle_bgr_output(out_raw_mat, last_raw_mat, last_nn);

        if (this->restarting)
        {
            // We've been interrupted. Tell anyone who is looking at the RTSP stream.
            this->cleanup(pipeline, last_raw_mat);
            return false;
        }
    }

    // Ran out of frames
    return true;
}

void UnetSemanticSegmentationModel::handle_inference_output(const cv::optional<cv::Mat> &out_nn, const cv::optional<int64_t> &out_nn_ts,
                                                            const cv::optional<int64_t> &out_nn_seqno, const cv::optional<std::vector<float>> &out_predictions,
                                                            const cv::Size &size,
                                                            cv::Mat &last_nn, std::vector<float> last_predictions)
{
    // This is important!
    // We only want to run the logic in this method if the neural network output node
    // was actually filled with something this time around. Otherwise, we will just
    // use the last_nn for our RTSP stream.
    if (!out_nn.has_value())
    {
        return;
    }

    // The below objects are on the same desynchronized path in the G-API graph.
    // We can therefore assume that if one of them has a value, they all have
    // a value. But let's assert for sanity.
    CV_Assert(out_nn_ts.has_value());
    CV_Assert(out_nn_seqno.has_value());
    CV_Assert(out_nn.has_value());
    CV_Assert(out_predictions.has_value());

    // Let's compose an IoT message in JSON that will send how much of each class
    // is found in the current frame.
    //
    // Let's use the following schema:
    //
    // {
    //   "Coverages": {
    //                  "<label>": <float> amount,
    //                  "<label>": <float> amount,
    //                  etc.
    //                }
    // }
    std::string msg = "{\"Predictions\": {";
    // for (size_t i = 0; i < out_coverages.value().size(); i++)
    // {
    auto label = util::get_label(static_cast<int>(out_predictions.value().at(0)), this->class_labels);
    msg.append("\"").append(label).append("\":" );
    msg.append(std::to_string(out_predictions.value().at(1)));
    msg.append(", ");
    // }
    // We need to remove the trailing comma and space
    msg = msg.substr(0, msg.length() - 2);
    msg.append("}}");

    // This is important!
    // Here is where we cache our latest network output.
    // We use this value for drawing frames until we get a new one.
    //
    // Dereferencing a cv::optional gets you the value. Same as calling value().
    last_nn = std::move(*out_nn);
    last_predictions = std::move(*out_predictions);

    // Let's resize our segmentation mask here.
    cv::resize(last_nn, last_nn, size, 0, 0, cv::INTER_LINEAR);

    // Now we log an inference.
    // This is done through an adaptive logger, so that the logging is done
    // at a decaying frequency (so that we don't fill up the log files in a day).
    // This method call is also why we can't label this whole method const :(
    this->log_inference(msg);

    // Now send the composed JSON message as well.
    iot::msgs::send_message(iot::msgs::MsgChannel::NEURAL_NETWORK, msg);
}

void UnetSemanticSegmentationModel::handle_bgr_output(cv::optional<cv::Mat> &out_raw_mat, cv::Mat &last_raw_mat, const cv::Mat &last_nn)
{
    // Just like in the handle_inference_output method, we need to make sure
    // that the branch of the G-API graph we are dealing with actually had outputs
    // at this iteration of the while loop. Otherwise, there's nothing for us to do here.
    if (!out_raw_mat.has_value())
    {
        return;
    }

    // Move the out_raw_mat into last_raw_mat.
    last_raw_mat = std::move(*out_raw_mat);

    // Let's create a copy of the raw output frame.
    cv::Mat result_mat;
    last_raw_mat.copyTo(result_mat);

    // We will actually need another copy (one for saving frames
    // from the camera, if the user has enabled this in the module twin).
    cv::Mat original;
    last_raw_mat.copyTo(original);

    // Now let's feed the raw RTSP stream with the raw frame.
    rtsp::update_data_raw(last_raw_mat);

    // Now compose the result frame onto the copy of the original.
    // (We initialized last_nn with some arbitrary dimensions, so only
    // try to compose a result image if we have something useful in last_nn)
    if (last_nn.size() == result_mat.size())
    {
        result_mat = (last_raw_mat / 2) + (last_nn / 2);
    }

    // If we have a status message for the user, we display
    // it on the result RTSP stream's frame.
    if (this->status_msg.empty())
    {
        rtsp::update_data_result(result_mat);
    }
    else
    {
        cv::Mat bgr_with_status;
        result_mat.copyTo(bgr_with_status);

        util::put_text(bgr_with_status, this->status_msg);
        rtsp::update_data_result(bgr_with_status);
    }

    // Maybe save and export the retraining data at this point
    this->save_retraining_data(original);
}

} // namespace model
