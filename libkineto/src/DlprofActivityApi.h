#pragma once

#include <vector>
#include <map>
#include <set>
#include <atomic>
#include <functional>

#ifdef HAS_DLPROF
#include "hc/profile.h"
#include "hc/activity.h"
#include "hc/platform.h"
#include "hc/defs.h"
#include "hc/device.h"
#include "hc/context.h"
#include "hc/stream.h"
//#include "json.hpp"
#endif

#include "ActivityType.h"
#include "GenericTraceActivity.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

class DlprofActivityApi {
 public:
  enum CorrelationFlowType {
    Default,
    User
  };

  DlprofActivityApi();
  DlprofActivityApi(const DlprofActivityApi&) = delete;
  DlprofActivityApi& operator=(const DlprofActivityApi&) = delete;

  virtual ~DlprofActivityApi();

  static DlprofActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableActivities(
    const std::set<ActivityType>& selected_activities);
  void disableActivities(
    const std::set<ActivityType>& selected_activities);
  void clearActivities();
  void teardownContext() {}

  int processActivities(ActivityLogger& logger,
                        std::function<const ITraceActivity*(int32_t)> linkedActivity,
                        int64_t startTime, int64_t endTime);

  void setMaxBufferSize(int size);

  std::atomic_bool stopCollection{false};

 private:
//  bool registered_{false};

  //Name cache
//  uint32_t nextStringId_{2};
//  std::map<uint32_t, std::string> strings_;
//  std::map<std::string, uint32_t> reverseStrings_;
//  std::map<activity_correlation_id_t, uint32_t> kernelNames_;
//
//  std::map<activity_correlation_id_t, GenericTraceActivity> kernelLaunches_;

  // Enabled Activity Filters
//  uint32_t activityMask_{0};
//  uint32_t activityMaskSnapshot_{0};
//  bool isLogged(libkineto::ActivityType atype);

};

} // namespace KINETO_NAMESPACE
