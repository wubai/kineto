/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once


#include <list>
#include <memory>

#include "libkineto.h"
#include "CuptiActivityBuffer.h"

namespace KINETO_NAMESPACE {

struct ActivityBuffers {
  std::list<std::unique_ptr<libkineto::CpuTraceBuffer>> cpu;
#ifdef HAS_CUPTI
  std::unique_ptr<CuptiActivityBufferMap> gpu;
#endif

#ifdef HAS_DLPROF
  std::unique_ptr<libkineto::CpuTraceBuffer> gpu;
#endif
  // Add a wrapper object to the underlying struct stored in the buffer
  template<class T>
  const ITraceActivity& addActivityWrapper(const T& act) {
    wrappers_.push_back(std::make_unique<T>(act));
    return *wrappers_.back().get();
  }

 private:
  std::vector<std::unique_ptr<const ITraceActivity>> wrappers_;
};

} // namespace KINETO_NAMESPACE
