#include <cstring>
#include <chrono>
#include <functional>
#include <time.h>

#include "DlprofActivityApi.h"
#include "output_base.h"
#include "ThreadUtil.h"

#include "Logger.h"
namespace KINETO_NAMESPACE {

using namespace libkineto;
DlprofActivityApi& DlprofActivityApi::singleton() {
  static DlprofActivityApi instance;
  return instance;
}

std::unique_ptr<libkineto::CpuTraceBuffer> DlprofActivityApi::activityBuffers() {
    LOG(0) << "DlprofActivityApi::activityBuffers";
    auto cputrace = std::make_unique<libkineto::CpuTraceBuffer>();
    return std::move(cputrace);
}

void DlprofActivityApi::pushCorrelationID(int id, CorrelationFlowType type) {
    LOG(0) << "DlprofActivityApi pushCorrelationID(" << id << ")";
}

void DlprofActivityApi::popCorrelationID(CorrelationFlowType type) {
    LOG(0) << "DlprofActivityApi popCorrelationID";
}

void DlprofActivityApi::setMaxBufferSize(int size) {
    LOG(0) << "setMaxBufferSize";
}
void DlprofActivityApi::clearActivities() {
    LOG(0) << "clearActivities";
}

void DlprofActivityApi::enableActivities(
    const std::set<ActivityType>& selected_activities) {
    LOG(INFO) << "dlprof enableCuptiActivities default: all";
}

void DlprofActivityApi::disableActivities(
    const std::set<ActivityType>& selected_activities) {
    LOG(0) << "dlprof: disableCuptiActivities";
}

const int DlprofActivityApi::processActivities(libkineto::CpuTraceBuffer&,
                                      std::function<void(const GenericTraceActivity*)> handler){
    LOG(0) << "new DlprofActivityApi::processActivities";
    return 0;
}

DlprofActivityApi::~DlprofActivityApi() {
    LOG(0) << "~DlprofActivityApi";
}
}