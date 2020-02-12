/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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

#ifndef INCLUDE_CAFFE_UTIL_UPGRADE_PROTO_HPP_
#define INCLUDE_CAFFE_UTIL_UPGRADE_PROTO_HPP_

#include <string>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Return true iff the net is not the current version.
bool NetNeedsUpgrade(const NetParameter& net_param);

#ifdef USE_MLU
bool NetNeedsSsdOptimization(const NetParameter& net_param);
void ParseNetSsdLocParameter(const NetParameter& param,
    vector<string>* ssd_bottoms,
    vector<int>* dropped_layers,
    string bottom_blob);
void ParseNetSsdConfParameter(const NetParameter& param,
    vector<string>* ssd_bottoms,
    vector<int>* dropped_layers,
    string bottom_blob);
void SetEmptyWeightsBias(LayerParameter* const layer);
void UpdateConvWeights(LayerParameter* const layer,
    const vector<vector<float>>& alphabeta);
void HackInplaceBlobs(NetParameter* param);
bool IsInt8Net(const NetParameter& param);
vector<vector<float>> GetBnAlphaBeta(const LayerParameter& layer);
vector<vector<float>> GetScaleAlphaBeta(const LayerParameter& layer);
vector<vector<int>> GetConvBnScaleStruct(const NetParameter& param,
        const NetParameter* param_without_weight = NULL);
void DeleteOptimizedLayers(NetParameter param,
    NetParameter* const param_optimized,
    const vector<int>& layer_dropped);
#endif
int NetGetLayerIndexByTopName(const NetParameter& net_param, const string top);

// Check for deprecations and upgrade the NetParameter as needed.
bool UpgradeNetAsNeeded(const string& param_file, NetParameter* param);

// Read parameters from a file into a NetParameter proto message.
void ReadNetParamsFromTextFileOrDie(const string& param_file,
                                    NetParameter* param);
void ReadNetParamsFromBinaryFileOrDie(const string& param_file,
                                      NetParameter* param,
                                      int opt_level = 0,
                                      const NetParameter* param_without_weights = NULL);
void ReadNetParamsFromBinaryMemOrDie(void* buffer, int buffer_size, NetParameter* param);
void ReadNetParamsFromTextMemOrDie(void* buffer, int buffer_size,
    NetParameter* param);

#ifdef USE_MLU
// If the value of use_global_stats_ isn't equal in Batch Norm layer bewteen prototxt
// anad caffemodel, set equal value decided by prototxt file
void UpdateUseGlobalStats(NetParameter* target_param, const NetParameter& source_param);
#endif

// Return true iff any layer contains parameters specified using
// deprecated V0LayerParameter.
bool NetNeedsV0ToV1Upgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade a V0NetParameter into a
// NetParameter (including upgrading padding layers and LayerParameters).
bool UpgradeV0Net(const NetParameter& v0_net_param, NetParameter* net_param);

// Upgrade NetParameter with padding layers to pad-aware conv layers.
// For any padding layer, remove it and put its pad parameter in any layers
// taking its top blob as input.
// Error if any of these above layers are not-conv layers.
void UpgradeV0PaddingLayers(const NetParameter& param,
                            NetParameter* param_upgraded_pad);

// Upgrade a single V0LayerConnection to the V1LayerParameter format.
bool UpgradeV0LayerParameter(const V1LayerParameter& v0_layer_connection,
                             V1LayerParameter* layer_param);

V1LayerParameter_LayerType UpgradeV0LayerType(const string& type);

// Return true iff any layer contains deprecated data transformation parameters.
bool NetNeedsDataUpgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade old transformation fields
// into a TransformationParameter.
void UpgradeNetDataTransformation(NetParameter* net_param);

// Return true iff the Net contains any layers specified as V1LayerParameters.
bool NetNeedsV1ToV2Upgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade a NetParameter with
// deprecated V1LayerParameters.
bool UpgradeV1Net(const NetParameter& v1_net_param, NetParameter* net_param);

bool UpgradeV1LayerParameter(const V1LayerParameter& v1_layer_param,
                             LayerParameter* layer_param);

const char* UpgradeV1LayerType(const V1LayerParameter_LayerType type);

// Return true iff the Net contains input fields.
bool NetNeedsInputUpgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade input fields into layers.
void UpgradeNetInput(NetParameter* net_param);

void UpdateInputBlobDim(NetParameter* net_param);

// Return true iff the Net contains batch norm layers with manual local LRs.
bool NetNeedsBatchNormUpgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade batch norm layers.
void UpgradeNetBatchNorm(NetParameter* net_param);

// Return true iff the solver contains any old solver_type specified as enums
bool SolverNeedsTypeUpgrade(const SolverParameter& solver_param);

bool UpgradeSolverType(SolverParameter* solver_param);

// Check for deprecations and upgrade the SolverParameter as needed.
bool UpgradeSolverAsNeeded(const string& param_file, SolverParameter* param);

// Read parameters from a file into a SolverParameter proto message.
void ReadSolverParamsFromTextFileOrDie(const string& param_file,
                                       SolverParameter* param);

}  // namespace caffe

#endif   // INCLUDE_CAFFE_UTIL_UPGRADE_PROTO_HPP_
