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

DlprofActivityApi::DlprofActivityApi() {
    LOG(0) << "DlprofActivityApi";
}

void DlprofActivityApi::pushCorrelationID(int id, CorrelationFlowType type) {
    LOG(0) << "pushCorrelationID(" << id << ")";
}

void DlprofActivityApi::popCorrelationID(CorrelationFlowType type) {
    LOG(0) << "popCorrelationID";
}

void DlprofActivityApi::setMaxBufferSize(int size) {
    LOG(0) << "setMaxBufferSize";
}
void DlprofActivityApi::clearActivities() {
    LOG(0) << "clearActivities";
}

void DlprofActivityApi::enableActivities(
    const std::set<ActivityType>& selected_activities) {
    LOG(0) << "enableCuptiActivities";
}

void DlprofActivityApi::disableActivities(
    const std::set<ActivityType>& selected_activities) {
    LOG(0) << "disableCuptiActivities";
}

DlprofActivityApi::~DlprofActivityApi() {
    LOG(0) << "~DlprofActivityApi";
}
}