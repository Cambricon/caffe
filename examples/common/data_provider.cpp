/*
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#if defined(USE_OPENCV)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "include/data_provider.hpp"
#include "include/pipeline.hpp"

#include "include/command_option.hpp"
#include "include/common_functions.hpp"

using std::string;
using std::vector;

template <typename Dtype, template <typename> class Qtype>
bool DataProvider<Dtype, Qtype>::imageIsEmpty() {
  if (this->imageList.empty())
    return true;

  return false;
}

template <typename Dtype, template <typename> class Qtype>
void DataProvider<Dtype, Qtype>::readOneBatch() {
  vector<cv::Mat> rawImages;
  vector<string> imageNameVec;
  string file_id , file;
  cv::Mat prev_image;
  int image_read = 0;

  while (image_read < this->inNum_) {
    if (!this->imageList.empty()) {
      file = file_id = this->imageList.front();
      this->imageList.pop();
      if (file.find(" ") != string::npos)
        file = file.substr(0, file.find(" "));
      cv::Mat img;
      if (FLAGS_yuv) {
        img = convertYuv2Mat(file, inGeometry_);
      } else {
        img = cv::imread(file, -1);
      }
      if (img.data) {
        ++image_read;
        prev_image = img;
        imageNameVec.push_back(file_id);
        rawImages.push_back(img);
      } else {
        LOG(INFO) << "failed to read " << file;
      }
    } else {
      if (image_read) {
        cv::Mat img;
        ++image_read;
        prev_image.copyTo(img);
        rawImages.push_back(img);
        imageNameVec.push_back("null");
      } else {
        // if the que is empty and no file has been read, no more runs
        return;
      }
    }
  }

  this->inImages_.push_back(rawImages);
  this->imageName_.push_back(imageNameVec);
}

template <typename Dtype, template <typename> class Qtype>
void DataProvider<Dtype, Qtype>::preRead() {
  while (this->imageList.size()) {
    this->readOneBatch();
  }
}

template <typename Dtype, template <typename> class Qtype>
void DataProvider<Dtype, Qtype>::WrapInputLayer(vector<vector<cv::Mat> >* wrappedImages,
                                  float* inputData) {
  //  Parameter images is a vector [ ----   ] <-- images[in_n]
  //                                |
  //                                |-> [ --- ] <-- channels[3]
  // This method creates Mat objects, and places them at the
  // right offset of input stream
  int width = this->runner_->w();
  int height = this->runner_->h();
  int channels = FLAGS_yuv ? 1 : this->runner_->c();
  for (int i = 0; i < this->runner_->n(); ++i) {
    wrappedImages->push_back(vector<cv::Mat> ());
    for (int j = 0; j < channels; ++j) {
      if (FLAGS_yuv) {
        cv::Mat channel(height, width, CV_8UC1, reinterpret_cast<char*>(inputData));
        (*wrappedImages)[i].push_back(channel);
        inputData += width * height / 4;
      } else {
        cv::Mat channel(height, width, CV_32FC1, inputData);
        (*wrappedImages)[i].push_back(channel);
        inputData += width * height;
      }
    }
  }
}

template <typename Dtype, template <typename> class Qtype>
void DataProvider<Dtype, Qtype>::Preprocess(const vector<cv::Mat>& sourceImages,
    vector<vector<cv::Mat> >* destImages) {
  /* Convert the input image to the input image format of the network. */
  CHECK(sourceImages.size() == destImages->size())
    << "Size of sourceImages and destImages doesn't match";
  for (int i = 0; i < sourceImages.size(); ++i) {
    if (FLAGS_yuv) {
      cv::Mat sample_yuv;
      sourceImages[i].convertTo(sample_yuv, CV_8UC1);
      cv::split(sample_yuv, (*destImages)[i]);
      continue;
    }
    cv::Mat sample;
    if (sourceImages[i].channels() == 3 && inChannel_ == 1)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGR2GRAY);
    else if (sourceImages[i].channels() == 4 && inChannel_ == 1)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGRA2GRAY);
    else if (sourceImages[i].channels() == 4 && inChannel_ == 3)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGRA2BGR);
    else if (sourceImages[i].channels() == 1 && inChannel_ == 3)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_GRAY2BGR);
    else
      sample = sourceImages[i];
    cv::Mat sample_resized;
    if (sample.size() != inGeometry_)
      cv::resize(sample, sample_resized, inGeometry_);
    else
      sample_resized = sample;
    cv::Mat sample_float;
    if (this->inChannel_ == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
    else
      sample_resized.convertTo(sample_float, CV_32FC1);
    cv::Mat sample_normalized;
    bool int8 = (FLAGS_int8 != -1) ? FLAGS_int8 : FLAGS_fix8;
    if (!int8 && (!meanFile_.empty() || !meanValue_.empty())) {
      cv::subtract(sample_float, mean_, sample_normalized);
      if (FLAGS_scale != 1) {
        sample_normalized *= FLAGS_scale;
      }
    } else {
      sample_normalized = sample_float;
    }
    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, (*destImages)[i]);
  }
}

template <typename Dtype, template <typename> class Qtype>
void DataProvider<Dtype, Qtype>::SetMean() {
  if (!this->meanFile_.empty())
    SetMeanFile();

  if (!this->meanValue_.empty())
    SetMeanValue();
}

template <typename Dtype, template <typename> class Qtype>
void DataProvider<Dtype, Qtype>::SetMeanValue() {
  if (FLAGS_yuv) return;
  cv::Scalar channel_mean;
  CHECK(this->meanFile_.empty()) <<
    "Cannot specify mean file and mean value at the same time";
  stringstream ss(this->meanValue_);
  vector<float> values;
  string item;
  while (getline(ss, item, ',')) {
    float value = std::atof(item.c_str());
    values.push_back(value);
  }
  CHECK(values.size() == 1 || values.size() == this->inChannel_) <<
    "Specify either one mean value or as many as channels: " << inChannel_;
  vector<cv::Mat> channels;
  for (int i = 0; i < inChannel_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(this->inGeometry_.height, this->inGeometry_.width, CV_32FC1,
        cv::Scalar(values[i]));
    channels.push_back(channel);
  }
  cv::merge(channels, this->mean_);
}

INSTANTIATE_ALL_CLASS(DataProvider);

#endif  // USE_OPENCV
