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

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <sstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "boost/filesystem.hpp"

#include "caffe/layers/detection_pose_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void DetectionPoseOutputLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const DetectionPoseOutputParameter& detection_pose_output_param =
      this->layer_param_.detection_pose_output_param();
  CHECK(detection_pose_output_param.has_num_classes())
      << "Must specify num_classes";
  num_classes_ = detection_pose_output_param.num_classes();
  share_location_ = detection_pose_output_param.share_location();
  num_loc_classes_ = share_location_ ? 1 : num_classes_;

  CHECK(detection_pose_output_param.has_num_poses())
      << "Must specify num_poses";
  num_poses_ = detection_pose_output_param.num_poses();
  share_pose_ = detection_pose_output_param.share_pose();
  num_pose_classes_ = share_pose_ ? 1 : num_classes_;
  background_label_id_ = detection_pose_output_param.background_label_id();
  code_type_ = detection_pose_output_param.code_type();
  variance_encoded_in_target_ =
      detection_pose_output_param.variance_encoded_in_target();
  keep_top_k_ = detection_pose_output_param.keep_top_k();
  confidence_threshold_ =
      detection_pose_output_param.has_confidence_threshold() ?
      detection_pose_output_param.confidence_threshold() : -FLT_MAX;

  // Parameters used in nms.
  nms_threshold_ = detection_pose_output_param.nms_param().nms_threshold();
  CHECK_GE(nms_threshold_, 0.) << "nms_threshold must be non negative.";
  top_k_ = -1;
  if (detection_pose_output_param.nms_param().has_top_k()) {
    top_k_ = detection_pose_output_param.nms_param().top_k();
  }
  const SaveOutputParameter& save_output_param =
      detection_pose_output_param.save_output_param();
  output_directory_ = save_output_param.output_directory();
  if (!output_directory_.empty() &&
      !boost::filesystem::is_directory(output_directory_)) {
    if (!boost::filesystem::create_directories(output_directory_)) {
        LOG(FATAL) << "Failed to create directory: " << output_directory_;
    }
  }
  output_name_prefix_ = save_output_param.output_name_prefix();
  need_save_ = output_directory_ == "" ? false : true;
  output_format_ = save_output_param.output_format();
  if (save_output_param.has_label_map_file()) {
    string label_map_file = save_output_param.label_map_file();
    if (label_map_file.empty()) {
      // Ignore saving if there is no label_map_file provided.
      LOG(WARNING) << "Provide label_map_file if output results to files.";
      need_save_ = false;
    } else {
      LabelMap label_map;
      CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
          << "Failed to read label map file: " << label_map_file;
      CHECK(MapLabelToName(label_map, true, &label_to_name_))
          << "Failed to convert label to name.";
      CHECK(MapLabelToDisplayName(label_map, true, &label_to_display_name_))
          << "Failed to convert label to display name.";
    }
  } else {
    need_save_ = false;
  }
  if (save_output_param.has_name_size_file()) {
    string name_size_file = save_output_param.name_size_file();
    if (name_size_file.empty()) {
      // Ignore saving if there is no name_size_file provided.
      LOG(WARNING) << "Provide name_size_file if output results to files.";
      need_save_ = false;
    } else {
      std::ifstream infile(name_size_file.c_str());
      CHECK(infile.good())
          << "Failed to open name size file: " << name_size_file;
      // The file is in the following format:
      //    name height width
      //    ...
      string name;
      int height, width;
      while (infile >> name >> height >> width) {
        names_.push_back(name);
        sizes_.push_back(std::make_pair(height, width));
      }
      infile.close();
      if (save_output_param.has_num_test_image()) {
        num_test_image_ = save_output_param.num_test_image();
      } else {
        num_test_image_ = names_.size();
      }
      CHECK_LE(num_test_image_, names_.size());
    }
  } else {
    need_save_ = false;
  }
  name_count_ = 0;
  visualize_ = detection_pose_output_param.visualize();
  if (visualize_) {
    visualize_threshold_ = 0.6;
    if (detection_pose_output_param.has_visualize_threshold()) {
      visualize_threshold_ = detection_pose_output_param.visualize_threshold();
    }
    data_transformer_.reset(
        new DataTransformer<Dtype>(this->layer_param_.transform_param(),
                                   this->phase_));
    data_transformer_->InitRand();
  }
}

template <typename Dtype>
void DetectionPoseOutputLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (need_save_) {
    CHECK_LE(name_count_, names_.size());
    if (name_count_ % num_test_image_ == 0) {
      // Clean all outputs.
      if (output_format_ == "VOC") {
        boost::filesystem::path output_directory(output_directory_);
        for (map<int, string>::iterator it = label_to_name_.begin();
             it != label_to_name_.end(); ++it) {
          if (it->first == background_label_id_) {
            continue;
          }
          std::ofstream outfile;
          boost::filesystem::path file(
              output_name_prefix_ + it->second + ".txt");
          boost::filesystem::path out_file = output_directory / file;
          outfile.open(out_file.string().c_str(), std::ofstream::out);
        }
      }
    }
  }
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  num_priors_ = bottom[3]->height() / 4;
  CHECK_EQ(num_priors_ * num_loc_classes_ * 4, bottom[0]->channels())
      << "Number of priors must match number of location predictions.";
  CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
      << "Number of priors must match number of confidence predictions.";
  CHECK_EQ(num_priors_ * num_pose_classes_ * num_poses_, bottom[2]->channels())
      << "Number of priors must match number of pose predictions.";


  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  // Since the number of bboxes to be kept is unknown before nms, we manually
  // set it to (fake) 1.
  top_shape.push_back(1);
  top_shape.push_back(9);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void DetectionPoseOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* conf_data = bottom[1]->cpu_data();
  const Dtype* pose_data = bottom[2]->cpu_data();
  const Dtype* prior_data = bottom[3]->cpu_data();
  const int num = bottom[0]->num();

  // Retrieve all location predictions.
  vector<LabelBBox> all_loc_preds;
  GetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
                    share_location_, &all_loc_preds);

  // Retrieve all pose predictions.
  vector<map<int, vector<vector<float> > > > all_pose_preds;
  GetPosePredictions(pose_data, num, num_poses_,
      num_priors_, num_pose_classes_,
      share_pose_, &all_pose_preds);


  // Retrieve all confidences.
  vector<map<int, vector<float> > > all_conf_scores;
  GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
                      &all_conf_scores);

  // Retrieve all prior bboxes. It is same within a batch since we assume all
  // images in a batch are of same dimension.
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

  int num_kept = 0;
  vector<map<int, vector<int> > > all_indices;
  vector<LabelBBox> all_decode_bboxes;
  for (int i = 0; i < num; ++i) {
    // Decode predictions into bboxes.
    LabelBBox decode_bboxes;
    for (int c = 0; c < num_loc_classes_; ++c) {
      int label = share_location_ ? -1 : c;
      if (label == background_label_id_) {
        // Ignore background class.
        continue;
      }
      if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for label " << label;
      }
      DecodeBBoxes(prior_bboxes, prior_variances,
                   code_type_, variance_encoded_in_target_,
                   all_loc_preds[i][label], &(decode_bboxes[label]));
    }
    all_decode_bboxes.push_back(decode_bboxes);

    // For each class, perform nms
    map<int, vector<float> >& conf_scores = all_conf_scores[i];
    map<int, vector<int> > indices;
    map<int, map<int, map<int, float> > > overlaps;
    int num_det = 0;
    for (int c = 0; c < num_classes_; ++c) {
      if (c == background_label_id_) {
        // Ignore background class.
        continue;
      }
      if (conf_scores.find(c) == conf_scores.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find confidence predictions for label " << c;
      }
      int label = share_location_ ? -1 : c;
      if (decode_bboxes.find(label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for label " << label;
        continue;
      }
      ApplyNMS(decode_bboxes[label], conf_scores[c], nms_threshold_,
               top_k_, share_location_, &(overlaps[label]), &(indices[c]));
      num_det += indices[c].size();
    }
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      vector<pair<float, pair<int, int> > > score_index_pairs;
      for (map<int, vector<int> >::iterator it = indices.begin();
           it != indices.end(); ++it) {
        int label = it->first;
        const vector<int>& label_indices = it->second;
        if (conf_scores.find(label) == conf_scores.end()) {
          // Something bad happened for current label.
          LOG(FATAL) << "Could not find location predictions for " << label;
          continue;
        }
        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          CHECK_LT(idx, conf_scores[label].size());
          score_index_pairs.push_back(std::make_pair(
                  conf_scores[label][idx], std::make_pair(label, idx)));
        }
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScorePairDescend<pair<int, int> >);
      score_index_pairs.resize(keep_top_k_);
      // Store the new indices.
      map<int, vector<int> > new_indices;
      for (int j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        new_indices[label].push_back(idx);
      }
      all_indices.push_back(new_indices);
      num_kept += keep_top_k_;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
  }

  // Now we know how many predictions are being kept
    vector<int> top_shape(2, 1);
    top_shape.push_back(num_kept);
    top_shape.push_back(9);
    top[0]->Reshape(top_shape);
    Dtype* top_data = top[0]->mutable_cpu_data();

  const Dtype* data = bottom[4]->cpu_data();
  float scale = 1;

  std::vector<float>mean_values_;

  mean_values_.push_back(104.0);
  mean_values_.push_back(117.0);
  mean_values_.push_back(123.0);

  const int channels = bottom[4]->channels();
  const int height = bottom[4]->height();
  const int width = bottom[4]->width();
#ifdef USE_OPENCV
  const int img_type = channels == 3 ? CV_8UC3 : CV_8UC1;
  cv::Mat orig_img(height, width, img_type, cv::Scalar(0, 0, 0));
  for (int h = 0; h < height; ++h) {
    uchar* ptr = orig_img.ptr<uchar>(h);
    int img_idx = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channels; ++c) {
        int idx = (c * height + h) * width + w;
        ptr[img_idx++] =
            static_cast<uchar>(data[idx] / scale + mean_values_[c]);
      }
    }
  }
#endif
  int count = 0;
  boost::filesystem::path output_directory(output_directory_);
  for (int i = 0; i < num; ++i) {
    map<int, vector<float> >& conf_scores = all_conf_scores[i];
    const LabelBBox& decode_bboxes = all_decode_bboxes[i];
    map<int, vector<vector<float> > >& pose_preds = all_pose_preds[i];
    // Get pose predictions for the im in this batch

    for (map<int, vector<int> >::iterator it = all_indices[i].begin();
         it != all_indices[i].end(); ++it) {
      int label = it->first;
      if (conf_scores.find(label) == conf_scores.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find confidence predictions for " << label;
        continue;
      }
      int loc_label = share_location_ ? -1 : label;
      if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for " << loc_label;
        continue;
      }
      int pose_label = share_pose_ ? -1 : label;
      // LOG(INFO) << "Pose label is " << pose_label;
      // Check this label
      if (pose_preds.find(pose_label) == pose_preds.end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find Pose predictions for " << pose_label;
        continue;
      }

      const vector<NormalizedBBox>& bboxes =
          decode_bboxes.find(loc_label)->second;
      const vector<vector<float> >& poses = pose_preds.find(pose_label)->second;

      vector<int>& indices = it->second;
      std::ofstream outfile;
      // TODO(Yang): I didn't modify the output:need_save_
      if (need_save_) {
        CHECK(label_to_name_.find(label) != label_to_name_.end())
            << "Cannot find label: " << label << " in the label map.";
        if (output_format_ == "VOC") {
          boost::filesystem::path file(
              output_name_prefix_ + label_to_name_[label] + ".txt");
          boost::filesystem::path out_file = output_directory / file;
          outfile.open(out_file.string().c_str(),
                       std::ofstream::out | std::ofstream::app);
        }
        CHECK_LT(name_count_, names_.size());
      }
      for (int j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        top_data[count * 9] = i;
        top_data[count * 9 + 1] = label;
        top_data[count * 9 + 2] = conf_scores[label][idx];
        NormalizedBBox clip_bbox;
        ClipBBox(bboxes[idx], &clip_bbox);
        top_data[count * 9 + 3] = clip_bbox.xmin();
        top_data[count * 9 + 4] = clip_bbox.ymin();
        top_data[count * 9 + 5] = clip_bbox.xmax();
        top_data[count * 9 + 6] = clip_bbox.ymax();
        vector<float> target_pose = poses[idx];
        vector<float>::iterator result;
        result = max_element(target_pose.begin(), target_pose.end());
        top_data[count * 9 + 7] = distance(target_pose.begin(), result);
        top_data[count * 9 + 8] = *(result);
        if (need_save_) {
          NormalizedBBox scale_bbox;
          ScaleBBox(clip_bbox, sizes_[name_count_].first,
                    sizes_[name_count_].second, &scale_bbox);
          if (output_format_ == "VOC") {
            outfile << names_[name_count_];
            outfile << " " << conf_scores[label][idx];
            outfile << " " << static_cast<int>(scale_bbox.xmin());
            outfile << " " << static_cast<int>(scale_bbox.ymin());
            outfile << " " << static_cast<int>(scale_bbox.xmax());
            outfile << " " << static_cast<int>(scale_bbox.ymax());
            outfile << std::endl;
            outfile.flush();
          } else if (output_format_ == "COCO") {
            boost::property_tree::ptree xmin, ymin, width, height;
            xmin.put<float>("", round(scale_bbox.xmin() * 100) / 100.);
            ymin.put<float>("", round(scale_bbox.ymin() * 100) / 100.);
            width.put<float>("",
                round((scale_bbox.xmax() - scale_bbox.xmin()) * 100) / 100.);
            height.put<float>("",
                round((scale_bbox.ymax() - scale_bbox.ymin()) * 100) / 100.);

            boost::property_tree::ptree cur_bbox;
            cur_bbox.push_back(std::make_pair("", xmin));
            cur_bbox.push_back(std::make_pair("", ymin));
            cur_bbox.push_back(std::make_pair("", width));
            cur_bbox.push_back(std::make_pair("", height));

            boost::property_tree::ptree cur_det;
            cur_det.put<int>("image_id", atoi(names_[name_count_].c_str()));
            cur_det.put<int>("category_id",
                atoi(label_to_name_[label].c_str()));
            cur_det.add_child("bbox", cur_bbox);
            cur_det.put<float>("score", conf_scores[label][idx]);

            detections_.push_back(std::make_pair("", cur_det));
          }
        }
        ++count;
      }
      if (need_save_) {
        outfile.close();
      }
    }
    if (need_save_) {
      ++name_count_;
      if (name_count_ % num_test_image_ == 0) {
        if (output_format_ == "COCO") {
          boost::filesystem::path output_directory(output_directory_);
          boost::filesystem::path file(output_name_prefix_ + ".json");
          boost::filesystem::path out_file = output_directory / file;
          std::ofstream outfile;
          outfile.open(out_file.string().c_str(), std::ofstream::out);

          boost::regex exp("\"(null|true|false|-?[0-9]+(\\.[0-9]+)?)\"");
          boost::property_tree::ptree output;
          output.add_child("detections", detections_);
          std::stringstream ss;
          write_json(ss, output);
          std::string rv = boost::regex_replace(ss.str(), exp, "$1");
          outfile << rv.substr(rv.find("["), rv.rfind("]") - rv.find("["))
              << std::endl << "]" << std::endl;
        }
      }
      if (name_count_ == names_.size()) {
        // reset count after a full iterations through the DB.
        name_count_ = 0;
      }
    }
  }
}

STUB_GPU(DetectionPoseOutputLayer);

INSTANTIATE_CLASS(DetectionPoseOutputLayer);

}  // namespace caffe
