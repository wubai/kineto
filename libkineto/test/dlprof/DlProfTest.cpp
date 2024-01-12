#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>

#include "hc/profile.h"
#include "hc/activity.h"
#include "hc/platform.h"
#include "hc/defs.h"
#include "hc/device.h"
#include "hc/context.h"
#include "hc/stream.h"
#include "json.hpp"

using namespace dl::hc;
using json = nlohmann::json;

// Common setup / teardown and helper functions
class DlProfTest : public ::testing::Test {
 protected:
  void SetUp() override {
    platform = Platform::GetInstance();
    ASSERT_EQ(platform->GetDevice(&device, 0), kHcSuccess);
    ASSERT_EQ(device->GetID(), 0);

    ASSERT_EQ(device->CreateContext(&ctx, kHcCtxTypeCuda), kHcSuccess);
    ASSERT_EQ(HCCreateProfileDataCollector(&profiler, ctx), kHcSuccess);

    profiler_ = std::shared_ptr<ProfileDataCollector>(
        profiler, [](ProfileDataCollector *p) { HCDestroyProfileDataCollector(p);});
  }

  int* createAscendHostMemory(int n) {
    int* hostArray = new int[n];
    for (int i = 0; i < n; i++) {
        hostArray[i] = i;
    }
    return hostArray;
  }

  void releaseAscendHostMemory(int* ptr) {
    delete[] ptr;
  }

  void TearDown() override {
      profiler_.reset();
  }

  ProfileDataCollector *profiler = nullptr;
  std::shared_ptr<ProfileDataCollector> profiler_;
  dl::hc::Platform *platform = nullptr;
  Device *device = nullptr;
  Context *ctx = nullptr;
};

TEST_F(DlProfTest, callid) {
    int* hostArray = nullptr;
    int* deviceArray = nullptr;
    int dataCount = 1000;
    std::unordered_map<int, int> info_pre; // size : callid
    std::unordered_map<int, std::pair<int, int>> info_post; // size: callid
    std::set<int> traceid_set;
    hostArray = createAscendHostMemory(dataCount);
    cudaMalloc((void**)&deviceArray, dataCount * sizeof(int));

    profiler_->AddMetrics(".*\\/timestamp");
    profiler_->Begin();
    ASSERT_EQ(profiler_->BeginPass(0), kHcSuccess);
    for (int i = 1; i < dataCount; i++) {
        ProfileFunctionCallBegin();
        info_pre[i * sizeof(int)] = ProfileFunctionCallId();
        cudaMemcpy(deviceArray, hostArray, i * sizeof(int), cudaMemcpyHostToDevice);
        ProfileFunctionCallEnd();
    }
    profiler_->EndPass();
    profiler_->End();

    ProfileMetricParser *parser = nullptr;
    std::shared_ptr<ProfileReport> report;
    HCCreateProfileParser(&parser, profiler_.get());
    ASSERT_EQ(parser->Parse(report, 0), kHcSuccess);

    std::stringstream ss;
    report->ExportJson(ss);
    json data = json::parse(ss);

    for (auto& element : data["metrics"]) {
        int trace_id = element["trace_id"].template get<int>();
        traceid_set.insert(trace_id);
    }

    for (auto& element : data["command_repo"]) {
        int trace_id = element["trace_id"].template get<int>();
        auto it = traceid_set.find(trace_id);
        if (it != traceid_set.end()) {
            if (element.contains("args")) {
                std::array<int, 1> size =  element["args"]["size"].template get<std::array<int,1>>();
                info_post[trace_id] = std::make_pair(size[0], -1);
            }
        }
    }
    std::cout << std::endl;
    for (auto& element : data["submits"]) {
        int trace_id = element["trace_id"].template get<int>();
        auto it = traceid_set.find(trace_id);
        if (it != traceid_set.end()) {
            int call_id = element["call_id"].template get<int>();
            info_post[trace_id] = std::make_pair(info_post[trace_id].first, call_id);
        }
    }

    ASSERT_EQ(info_pre.size(), info_post.size());

    // traverse the info_post, compare it with info_pre
    for (const auto& pair : info_post) {
        int size = pair.second.first;
        int call_id = pair.second.second;
        auto it = info_pre.find(size);
        if (it != info_pre.end()) {
            ASSERT_EQ(call_id, it->second);
        } else {
            FAIL() << "can't find corresponding call_id!" << std::endl;
        }
    }

    cudaFree(deviceArray);
    releaseAscendHostMemory(hostArray);
}