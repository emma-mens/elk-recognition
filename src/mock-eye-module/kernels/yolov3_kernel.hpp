// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// Standard libary includes
#include <tuple>
#include <vector>

// Third party includes
#include <opencv2/opencv.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

// void getMaxForRow(std::vector<float> array, int& oClass, float& oConf);
// bool isFinite(std::vector<float> arr);
// void getBestPrediction(std::vector<std::vector<float>> mat, std::vector<float> &pred, int minHW, float confThresh);
// 
namespace cv {
namespace gapi {
namespace custom {


#define N_CLASSES 8
// Op for parsing the output of the U-Net semantic segmentation network.
// This macro creates a G-API op. Remember, an op is the interface for a G-API "function",
// which can be written into a G-API graph.
//
// Since it is just the interface and not the implementation, we merely describe the input and the
// output of the operation here, while leaving the actual implementation up to the kernel we will
// write down below.
//
// For a little explanation: the macro takes three arguments:
// G_API_OP(op name, op function signature, op tag)
//
// The op name can be whatever you want. We have adopted the Intel convention of labeling ops
// GWhatever using camel case. Kernels are also camel case and follow the format GOCVWhatever.
// C++ wrapper functions use snake_case. Feel free to do whatever you want though.
//
// The function signature looks like this <output args(input args)>. Because C++ does not have native
// support for multiple return values (like Python), we need to wrap the multiple return values into
// a std::tuple if you have multiple outputs. Right now, we only have one output.
//
// A note about the types: in order to insert an op into a G-API graph, you need to make sure the types
// are mapped from what you actually want into the G-API type system.
// G-API types are simple: they have support for primatives, for cv::Mat objects as GMat objects,
// vectors as GArray objects, and everything else as GOpaque<type>.
//
// The tag can be whatever you want, and frankly, I'm not even sure what it's used for...
G_API_OP(GParseYoloV3, <std::tuple<GMat, GArray<float>>(GMat)>, "org.microsoft.gparseyolov3")
{
    // This boilerplate is required within curly braces following the macro.
    // You declare a static function called outMeta, which must take in WhateverDesc
    // versions of the types and output the same thing.
    //
    // Notice that we have mapped our inputs from GMat -> GMatDesc
    // (and added const and reference declaration syntax to the input).
    // static GMatDesc outMeta(const GMatDesc&)
    // {
    //     // This must be the right shape and type of the output. Otherwise you will get
    //     // an ominous error about the meta data being wrong or about a Mat object resizing in a kernel.
    //     return cv::GMatDesc(CV_32FC1, {1, 2535, 13});
    // }
    static std::tuple<GMatDesc, GArrayDesc> outMeta(const GMatDesc &in)
    {
        // This must be the right shape and type of the output. Otherwise you will get
        // an ominous error about the meta data being wrong.
        auto desc = empty_gmat_desc().withSize({2535, 13}).withType(CV_8U, 3);
        return {desc, empty_array_desc()};
    }
};

static void getMaxForRow(std::vector<float> array, int& oClass, float& oConf) {
  float best = -1;
  int bestPos = -1;
  int start = 5;
  // for yolov3 class predictions start at column 5
  for (int i = start; (unsigned)i < array.size(); i++) {
    if (array.at(i) > best) {
      best = array.at(i);
      bestPos = i - start;
    }
  }
  oClass = bestPos;
  oConf = best;
}

static bool isFinite(std::vector<float> arr) {
  for (int i = 1; (unsigned)i < arr.size(); i++) {
    if (std::isinf(arr.at(i))) return false;
  }
  return true;
}

static void getBestPrediction(std::vector<std::vector<float>> mat, std::vector<float> &pred, int minHW, float confThresh) {
   int bestClass = -1;
   float bestConf = -1;
   int objClass;
   float objConf, x, y;
   int N = mat.size();
   std::vector<float> currArray;
   for (int i = 0; i < N; i++) {
     currArray = mat.at(i);
     getMaxForRow(currArray, objClass, objConf);
     objConf *= currArray.at(4); // for yolov3 column 4 is the confidence that current bounding box contains an object
                              // multiply to with conditional confidence that object is the given class for final probability
     x = currArray.at(2);
     y = currArray.at(3);

     if (x > minHW && y > minHW && objConf > confThresh && isFinite(currArray)) {
       // this is a candidate prediction
       if (objConf > bestConf) {
         bestConf = objConf;
         bestClass = objClass;
       }
     }
   }
   pred.push_back(bestClass);
   pred.push_back(bestConf);
}

// Input: 32FC1 {1, 2535, 13}  -- (i.e., 1, 2535, 13) # yolov3 output for 8 classes
// Output: 32FC1 {1, 2}    -- (i.e., 1, 2) # top prediction and class
static void get_final_predictions(const cv::Mat &mat, std::vector<float> &preds) {

    CV_Assert(mat.dims == 2);

    int minHW = 2;
    float confThresh = 0.3;

    std::vector<std::vector<float>> vMat;

    for (int i = 0; i < mat.rows; ++i) {
	std::vector<float> inner;
	for (int j = 0; j < mat.cols; j++) {
	    inner.push_back(mat.at<float>(i,j));
	}
	vMat.push_back(inner);
    }
    getBestPrediction(vMat, preds, minHW, confThresh);

}

// This is the kernel declaration for the op.
// A single op can have several kernel implementations. That's kind of the whole point.
// The idea behind G-API is twofold: one is to make a declarative style computer vision
// pipeline syntax, and the other is to separate the declaration of the computer vision pipeline
GAPI_OCV_KERNEL(GOCVParseYoloV3, GParseYoloV3)
{
    //
    // <cv::Mat(cv::Mat)>
    //
    // but because we need to map our return value to an output reference,
    // the actual signature of this function is:
    // <void(const Mat&, Mat&)>
    static void run(const Mat &in_img, Mat &out_img, std::vector<float> &predictions)
    {
        // Here's where we will implement all the logic for post-processing our neural network
        // Our network outputs a shape {Batch Size, Predictions, bboxInfo} tensor.
	CV_Assert(in_img.size.dims() == 3);

	// std::cout << in_img.size << " in size before" << std::endl;
        // std::cout << out_img.size << " out size before" << std::endl;


        // CV_Assert(in_img.type() == out_img.type());
        // CV_Assert(in_img.size == out_img.size);

	const int height = in_img.size[1];
        const int width = in_img.size[2];

	// Squeeze the batch dimension
        // Batch Size is always going to be 1 on our device, so let's just remove that.
        const cv::Mat squeezed_img = in_img.reshape(0, {height, width});
	// std::cout << squeezed_img.size << " squeezed size" << std::endl;

        get_final_predictions(squeezed_img, predictions);
	// std::cout << "final predictions " << predictions.at(0) << " " << predictions.at(1) << std::endl;
        // in_img.copyTo(out_img);
	// squeezed_img.copyTo(out_img);

	// std::cout << in_img.size << " in size after " << std::endl;
        // std::cout << out_img.size << " out size after " << std::endl;
        // Note that we CANNOT do this, because it will throw an exception that the G-API detected that
        // a kernel parameter was reallocated.
        //out_img = in_img.clone();
    }
};

// Don't forget to declare the C++ wrapper function in our header!
GAPI_EXPORTS std::tuple<GMat, GArray<float>> parse_yolov3(const GMat &in);

} // namespace custom
} // namespace gapi
} // namespace cv
