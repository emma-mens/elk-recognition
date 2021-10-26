// Put this in a file called mock-eye-module/modules/segmentation/unet_semseg.cpp

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Standard library includes
#include <iomanip>
#include <random>
#include <string>
#include <vector>

// Third party includes
#include <opencv2/core/utility.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

// Our includes
#include "../../kernels/utils.hpp"
#include "../../kernels/yolov3.hpp"
#include "../device.hpp"
#include "../parser.hpp"
#include "yolov3.hpp"

namespace elk_yolov3 {

// This macro is used to tell G-API what types this network is going to take in and output.
// In our case, we are going to take in a single image (represented as a CV Mat, in G-API called a GMat)
// and output a tensor of shape {Batch size (which will be 1), N_CLASSES, 416 pixels high, 416 pixels wide},
// which we will again represent as a GMat.
//
// The tag at the end can be anything you want.
//
// In case you are wondering, it is <output(input)>
G_API_NET(ElkYoloV3, <cv::GMat(cv::GMat)>, "com.microsoft.elk-yolov3-network");

// First version:
//
// All it does is take in the video file, frame by frame, and pipe it through a do-nothing G-API graph,
// and then display the frames.
void compile_and_run(const std::string &video_fpath, const std::string &modelfpath, const std::string &weightsfpath, const device::Device &device, bool show, const std::vector<std::string> &labels)
{
    // Create the network itself. Here we are using the cv::gapi::ie namespace, which stands for Inference Engine.
    // On the device, we have a custom back end, namespaced as cv::gapi::mx instead.
    auto network = cv::gapi::ie::Params<ElkYoloV3>{ modelfpath, weightsfpath, device::device_to_string(device) };

    // Graph construction //////////////////////////////////////////////////////

    // Construct the input node. We will fill this in with OpenCV Mat objects as we get them from the video source.
    cv::GMat in;

    // This is the only thing we are doing so far: copying the input to the output.
    auto raw_input = cv::gapi::copy(in);
    auto nn = cv::gapi::infer<ElkYoloV3>(in);
    auto parsed_nn = cv::gapi::custom::parse_yolov3(nn);
    auto graph_outs = cv::GOut(raw_input, parsed_nn);


    // Graph compilation ///////////////////////////////////////////////////////

    // Set up the inputs and outpus of the graph.
    auto comp = cv::GComputation(cv::GIn(in), std::move(graph_outs));

    auto kernels = cv::gapi::kernels<cv::gapi::custom::GOCVParseYoloV3>();

    // Now compile the graph into a pipeline object that we will use as
    // an abstract black box that takes in images and outputs images.
    auto compiled_args = cv::compile_args(kernels, cv::gapi::networks(network));
    auto pipeline = comp.compileStreaming(std::move(compiled_args));


    // Graph execution /////////////////////////////////////////////////////////

    // Select a video source - either the webcam or an input file.
    if (!video_fpath.empty())
    {
        pipeline.setSource(cv::gin(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(video_fpath)));
    }
    else
    {
        pipeline.setSource(cv::gin(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(-1)));
    }

    // Now start the pipeline
    pipeline.start();

    // Set up all the output nodes.
    // Each data container needs to match the type of the G-API item that we used as a stand in
    // in the GOut call above. And the order of these data containers needs to match the order
    // that we specified in the GOut call above.
    //
    // Also, it is possible to make the G-API graph asynchronous so that each
    // item is delivered as quickly as it can. In fact, we do this in the Azure Percept azureeyemodule's application
    // so that we can output the raw RTSP stream at however fast it comes in, regardless of how fast the neural
    // network is running.
    //
    // In synchronous mode (the default), no item is output until all the output nodes have
    // something to output.
    //
    // We'll just use synchronous mode here and we'll discuss asynchronous mode later when we port to the device.
    cv::Mat out_raw_mat;
    cv::Mat out_nn;
    auto pipeline_outputs = cv::gout(out_raw_mat, out_nn);

    // Pull the information through the compiled graph, filling our output nodes at each iteration.
    while (pipeline.pull(std::move(pipeline_outputs)))
    {
        if (show)
        {
            cv::imshow("Out", out_raw_mat);
            cv::waitKey(1);

	    // Now let's print our network's output dimensions
            // If you have been following along so far, these dimensions should be {1, 5, 256, 256}
            std::cout << " Dimensions: ";
            for (auto i = 0; i < out_nn.size.dims(); i++)
            {
                std::cout << std::to_string(out_nn.size[i]);
                if (i != (out_nn.size.dims() - 1))
                {
                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
        }
    }
}

} // namespace elk_yolov3

