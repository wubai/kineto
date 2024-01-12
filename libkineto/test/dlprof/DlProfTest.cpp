#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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
    std::cout << "dlprof setup" << std::endl;
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
    std::cout << "hello dlprof" << std::endl;
    int* hostArray = nullptr;
    int* deviceArray = nullptr;
    int dataCount = 11;
    std::vector<uint32_t> callid;
    hostArray = createAscendHostMemory(dataCount);
    cudaMalloc((void**)&deviceArray, dataCount * sizeof(int));

    profiler_->AddMetrics(".*\\/timestamp");
    profiler_->Begin();
    ASSERT_EQ(profiler_->BeginPass(0), kHcSuccess);
    for (int i = 1; i < dataCount; i++) {
        ProfileFunctionCallBegin();
        callid.push_back(ProfileFunctionCallId());
        cudaMemcpy(deviceArray, hostArray, i * sizeof(int), cudaMemcpyHostToDevice);
        ProfileFunctionCallEnd();
    }
    profiler_->EndPass();
    profiler_->End();

    ProfileMetricParser *parser = nullptr;
    std::shared_ptr<ProfileReport> report;
    HCCreateProfileParser(&parser, profiler_.get());
    ASSERT_EQ(parser->Parse(report, 0), kHcSuccess);
//    report->DumpMetrics(std::cout);

    std::stringstream ss;
    report->ExportJson(ss);
    std::string str = ss.str();
    json data = json::parse(str);
    std::cout << data["metrics"] << std::endl;

    for (auto& element : data["metrics"]) {
        std::cout << element << '\n';
    }
    std::cout << std::endl;
    for (auto& element : data["command_repo"]) {
        std::cout << element << '\n';
    }
    std::cout << std::endl;
    for (auto& element : data["submits"]) {
        std::cout << element << '\n';
    }

    cudaFree(deviceArray);
    releaseAscendHostMemory(hostArray);
}