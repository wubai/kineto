/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <strings.h>
#include <time.h>
#include <chrono>

#ifdef __linux__
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#endif

#include "include/libkineto.h"
#include "include/Config.h"
#include "src/CuptiActivityProfiler.h"
#include "src/ActivityTrace.h"
#include "src/DlprofActivityApi.h"
#include "src/output_base.h"
#include "src/output_json.h"
#include "src/output_membuf.h"

#include "src/Logger.h"

using namespace std::chrono;
using namespace KINETO_NAMESPACE;

namespace {
const TraceSpan& defaultTraceSpan() {
  static TraceSpan span(0, 0, "Unknown", "");
  return span;
}
}

// Provides ability to easily create a few test CPU-side ops
struct MockCpuActivityBuffer : public CpuTraceBuffer {
  MockCpuActivityBuffer(int64_t startTime, int64_t endTime) {
    span = TraceSpan(startTime, endTime,"Test trace");
    gpuOpCount = 0;
  }

  void addOp(std::string name, int64_t startTime, int64_t endTime, int64_t correlation) {
    GenericTraceActivity op(span, ActivityType::CPU_OP, name);
    op.startTime = startTime;
    op.endTime = endTime;
    op.device = systemThreadId();
    op.resource = systemThreadId();
    op.id = correlation;
    emplace_activity(std::move(op));
    span.opCount++;
  }
};

// Provides ability to easily create a few test dlprof ops
struct MockCuptiActivityBuffer {
//  void addRuntimeActivity(
//      int32_t id,
//      int64_t start_us, int64_t end_us, int64_t correlation) {
//    auto& act = createActivity<GenericTraceActivity>(
//        start_us, end_us, correlation);
//    act.kind = CUPTI_ACTIVITY_KIND_RUNTIME;
//    act.cbid = cbid;
//    act.threadId = threadId();
//    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
//  }
//
//  void addDriverActivity(
//      CUpti_driver_api_trace_cbid_enum cbid,
//      int64_t start_us, int64_t end_us, int64_t correlation) {
//    auto& act = createActivity<CUpti_ActivityAPI>(
//        start_us, end_us, correlation);
//    act.kind = CUPTI_ACTIVITY_KIND_DRIVER;
//    act.cbid = cbid;
//    act.threadId = threadId();
//    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
//  }
//
//  void addKernelActivity(
//      int64_t start_us, int64_t end_us, int64_t correlation) {
//    auto& act = createActivity<CUpti_ActivityKernel4>(
//        start_us, end_us, correlation);
//    act.kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
//    act.deviceId = 0;
//    act.contextId = 0;
//    act.streamId = 1;
//    act.name = "kernel";
//    act.gridX = act.gridY = act.gridZ = 1;
//    act.blockX = act.blockY = act.blockZ = 1;
//    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
//  }

  void addMemcpyActivity(
      int64_t start_us, int64_t end_us, int64_t correlation) {
    GenericTraceActivity* act = new GenericTraceActivity;
    act->activityType = ActivityType::GPU_MEMCPY;
    act->activityName = "DL_MEMCPY";
    act->device = 0;
    uint32_t streamId = 2;
    act->addMetadata("stream", fmt::format("{}", streamId));
    act->startTime = start_us * 1000;
    act->endTime = end_us * 1000;

    act->flow.id = correlation;
    act->flow.type = kLinkAsyncCpuGpu;
    activities.push_back(act);
  }

//  void addSyncActivity(
//      int64_t start_us, int64_t end_us, int64_t correlation,
//      CUpti_ActivitySynchronizationType type, int64_t stream = 1) {
//    auto& act = createActivity<CUpti_ActivitySynchronization>(
//        start_us, end_us, correlation);
//    act.kind = CUPTI_ACTIVITY_KIND_SYNCHRONIZATION;
//    act.type = type;
//    act.contextId = 0;
//    act.streamId = stream;
//    activities.push_back(reinterpret_cast<CUpti_Activity*>(&act));
//  }

//  template<class T>
//  T& createActivity(
//      int64_t start_us, int64_t end_us, int64_t correlation) {
//    T& act = *static_cast<T*>(malloc(sizeof(T)));
//    bzero(&act, sizeof(act));
//    act.start = start_us * 1000;
//    act.end = end_us * 1000;
//    act.correlationId = correlation;
//    return act;
//  }

  ~MockCuptiActivityBuffer() {
    for (GenericTraceActivity* act : activities) {
      free(act);
    }
  }

  std::vector<GenericTraceActivity*> activities;
};

// Mock parts of the CuptiActivityApi
class MockCuptiActivities : public DlprofActivityApi {
 public:
  virtual int processActivities(
    ActivityLogger& logger, std::function<const ITraceActivity*(int32_t)> linkedActivity,
    int64_t startTime, int64_t endTime)  {
//    for (CUpti_Activity* act : activityBuffer->activities) {
//      handler(act);
//    }
//    return {activityBuffer->activities.size(), 100};
  }

//  virtual std::unique_ptr<CuptiActivityBufferMap>
//  activityBuffers() override {
//    auto map = std::make_unique<CuptiActivityBufferMap>();
//    auto buf = std::make_unique<CuptiActivityBuffer>(100);
//    uint8_t* addr = buf->data();
//    (*map)[addr] = std::move(buf);
//    return map;
//  }

//  void bufferRequestedOverride(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
//    this->bufferRequested(buffer, size, maxNumRecords);
//  }

  std::unique_ptr<MockCuptiActivityBuffer> activityBuffer;
};


// Common setup / teardown and helper functions
class CuptiActivityProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    profiler_ = std::make_unique<CuptiActivityProfiler>(
        cuptiActivities_, /*cpu only*/ false);
    cfg_ = std::make_unique<Config>();
    cfg_->validate(std::chrono::system_clock::now());
    loggerFactory.addProtocol("file", [](const std::string& url) {
        return std::unique_ptr<ActivityLogger>(new ChromeTraceLogger(url));
    });
  }

  std::unique_ptr<Config> cfg_;
  MockCuptiActivities cuptiActivities_;
  std::unique_ptr<CuptiActivityProfiler> profiler_;
  ActivityLoggerFactory loggerFactory;
};

void checkTracefile(const char* filename) {
#ifdef __linux__
  // Check that the expected file was written and that it has some content
  int fd = open(filename, O_RDONLY);
  if (!fd) {
    perror(filename);
  }
  EXPECT_TRUE(fd);
  // Should expect at least 100 bytes
  struct stat buf{};
  fstat(fd, &buf);
  EXPECT_GT(buf.st_size, 100);
  close(fd);
#endif
}


TEST_F(CuptiActivityProfilerTest, SyncTrace) {

  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules(
      {"CuptiActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  CuptiActivityProfiler profiler(cuptiActivities_, /*cpu only*/ false);
  int64_t start_time_us = 100;
  int64_t duration_us = 300;
  auto start_time = time_point<system_clock>(microseconds(start_time_us));
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + microseconds(duration_us));

  profiler.recordThreadInfo();

  // Log some cpu ops
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_us, start_time_us + duration_us);
  cpuOps->addOp("op1", 120, 150, 1);
  cpuOps->addOp("op2", 130, 140, 2);
  cpuOps->addOp("op3", 200, 250, 3);
  cpuOps->addOp("op4", 260, 280, 4);
  profiler.transferCpuTrace(std::move(cpuOps));

  // And some GPU ops
  auto gpuOps = std::make_unique<MockCuptiActivityBuffer>();
//  gpuOps->addRuntimeActivity(CUDA_LAUNCH_KERNEL, 133, 138, 1);
//  gpuOps->addRuntimeActivity(CUDA_MEMCPY, 210, 220, 2);
//  gpuOps->addRuntimeActivity(CUDA_LAUNCH_KERNEL, 230, 245, 3);
//  gpuOps->addDriverActivity(CU_LAUNCH_KERNEL, 265, 275, 4);
//  gpuOps->addRuntimeActivity(CUDA_STREAM_SYNC, 246, 340, 5);
//  gpuOps->addKernelActivity(150, 170, 1);
  gpuOps->addMemcpyActivity(240, 250, 2);
//  gpuOps->addKernelActivity(260, 320, 3);
//  gpuOps->addKernelActivity(330, 350, 4);
//  gpuOps->addSyncActivity(321, 323, 5, CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE);
  // Add wait event on kernel stream 1
//  gpuOps->addSyncActivity(
//      324, 326, 6, CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT,
//      1 /*stream*/);
  // This event should be ignored because it is not on a stream that has no GPU kernels
//  gpuOps->addSyncActivity(
//      326, 330, 7, CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT,
//      4 /*stream*/);
  cuptiActivities_.activityBuffer = std::move(gpuOps);

  // Have the profiler process them
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);

  // Profiler can be reset at this point - logger owns the activities
  profiler_->reset();

  // Wrapper that allows iterating over the activities
  ActivityTrace trace(std::move(logger), loggerFactory);
  EXPECT_EQ(trace.activities()->size(), 15);
  std::map<std::string, int> activityCounts;
  std::map<int64_t, int> resourceIds;
  for (auto& activity : *trace.activities()) {
    activityCounts[activity->name()]++;
    resourceIds[activity->resourceId()]++;
  }
  for (const auto& p : activityCounts) {
    LOG(INFO) << p.first << ": " << p.second;
  }
  EXPECT_EQ(activityCounts["op1"], 1);
  EXPECT_EQ(activityCounts["op2"], 1);
  EXPECT_EQ(activityCounts["op3"], 1);
  EXPECT_EQ(activityCounts["op4"], 1);
  EXPECT_EQ(activityCounts["cudaLaunchKernel"], 2);
  EXPECT_EQ(activityCounts["cuLaunchKernel"], 1);
  EXPECT_EQ(activityCounts["cudaMemcpy"], 1);
  EXPECT_EQ(activityCounts["cudaStreamSynchronize"], 1);
  EXPECT_EQ(activityCounts["kernel"], 3);
  EXPECT_EQ(activityCounts["Stream Sync"], 1);
  EXPECT_EQ(activityCounts["Memcpy HtoD (Pinned -> Device)"], 1);

  auto sysTid = systemThreadId();
  // Ops and runtime events are on thread sysTid along with the flow start events
  EXPECT_EQ(resourceIds[sysTid], 9);
  // Kernels and sync events are on stream 1, memcpy on stream 2
  EXPECT_EQ(resourceIds[1], 5);
  EXPECT_EQ(resourceIds[2], 1);

#ifdef __linux__
  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  mkstemps(filename, 5);
  trace.save(filename);
  // Check that the expected file was written and that it has some content
  int fd = open(filename, O_RDONLY);
  if (!fd) {
    perror(filename);
  }
  EXPECT_TRUE(fd);
  // Should expect at least 100 bytes
  struct stat buf{};
  fstat(fd, &buf);
  EXPECT_GT(buf.st_size, 100);
#endif
}

