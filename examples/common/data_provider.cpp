/*
All modification made by Cambricon Corporation: © 2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2019, the respective contributors
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
cv::Mat DataProvider<Dtype, Qtype>::ResizeMethod(cv::Mat sample, int inputDim,
    int mode) {
  int left_x, top_y, new_h, new_w;
  float img_w, img_h, img_scale;
  cv::Mat sample_temp;
  cv::Mat sample_temp_416;
  cv::Mat sample_temp_bgr;
  cv::Rect select;
  switch (mode) {
    case 0:  // resize source image into inputdim * inputdim
      cv::resize(sample, sample_temp, cv::Size(inputDim, inputDim));
      if (inGeometry_.width > inputDim || inGeometry_.height > inputDim) {
        LOG(INFO) <<"input size overrange inputdim X inputdim, you can try again"
                  << " by setting preprocess_option value to 0.";
        exit(1);
      }
      left_x = inputDim / 2 - inGeometry_.width / 2;
      top_y = inputDim / 2 - inGeometry_.height / 2;
      break;
    case 1:  // resize source image into inputdim * N, N is bigger than inputdim
      img_w = sample.cols;
      img_h = sample.rows;
      img_scale = img_w < img_h ? (inputDim / img_w) : (inputDim / img_h);
      new_w = std::round(img_w * img_scale);
      new_h = std::round(img_h * img_scale);
      cv::resize(sample, sample_temp, cv::Size(new_w, new_h));
      if (inGeometry_.width > new_w || inGeometry_.height > new_h) {
        LOG(INFO) <<"input size overrange inputdim X N, you can try again"
                  << " by setting preprocess_option value to 0.";
        exit(1);
      }
      left_x = new_w / 2 - inGeometry_.width / 2;
      top_y = new_h / 2 - inGeometry_.height / 2;
      break;
    case 2:  // resize source image into inputdim * n, n is samller than inputdim
      img_w = sample.cols;
      img_h = sample.rows;
      img_scale = img_w < img_h ? (inputDim / img_h) : (inputDim / img_w);
      new_w = std::floor(img_w * img_scale);
      new_h = std::floor(img_h * img_scale);
      cv::resize(sample, sample_temp, cv::Size(new_w, new_h), CV_INTER_LINEAR);
      if (inChannel_ == 3)
        sample_temp_416 = cv::Mat(inGeometry_.height, inGeometry_.height,
                                CV_8UC3, cv::Scalar(128, 128, 128));
      if (inChannel_ == 4)
        sample_temp_416 = cv::Mat(inGeometry_.height, inGeometry_.height,
                                CV_8UC4, cv::Scalar(128, 128, 128, 128));
      sample_temp.copyTo(sample_temp_416(
                        cv::Range((static_cast<float>(inGeometry_.height) - new_h) / 2,
                          (static_cast<float>(inGeometry_.height) - new_h) / 2 + new_h),
                        cv::Range((static_cast<float>(inGeometry_.height) - new_w) / 2,
                          (static_cast<float>(inGeometry_.height) - new_w) / 2 + new_w)));
      //  BGR(A)->RGB(A)
      if (inChannel_ == 3){
        cv::cvtColor(sample_temp_416, sample_temp_bgr, cv::COLOR_BGR2RGB);
        sample_temp_bgr.convertTo(sample_temp, CV_32FC3, 1);
      }
      if (inChannel_ == 4) {
        cv::cvtColor(sample_temp_416, sample_temp_bgr, cv::COLOR_BGRA2RGBA);
        sample_temp_bgr.convertTo(sample_temp, CV_32FC4, 1);
      }
      left_x = 0;
      top_y = 0;
      break;
    default:
      break;
  }
  select = cv::Rect(cv::Point(left_x, top_y), inGeometry_);
  return sample_temp(select);
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
    else if (sourceImages[i].channels() == 3 && inChannel_ == 4)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_BGR2BGRA);
    else if (sourceImages[i].channels() == 1 && inChannel_ == 4)
      cv::cvtColor(sourceImages[i], sample, cv::COLOR_GRAY2BGRA);
    else
      sample = sourceImages[i];
    cv::Mat sample_resized;
    if (sample.size() != inGeometry_) {
      switch (FLAGS_preprocess_option) {
        case 0:
          cv::resize(sample, sample_resized, inGeometry_);
          break;
        case 1:  // 256 x N
          sample_resized = ResizeMethod(sample, 256, 1);
          break;
        case 2:  // 256 x 256
          sample_resized = ResizeMethod(sample, 256, 0);
          break;
        case 3:  // 320 x N for inception-v3
          sample_resized = ResizeMethod(sample, 320, 1);
          break;
        case 4:  // n * 416 for yolov2
          sample_resized = ResizeMethod(sample, 416, 2);
          break;
        default:
          cv::resize(sample, sample_resized, inGeometry_);
          break;
      }
    } else {
      sample_resized = sample;
    }

    cv::Mat sample_float;
    if (this->inChannel_ == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
    else if (this->inChannel_ == 4)
      sample_resized.convertTo(sample_float, CV_32FC4);
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
