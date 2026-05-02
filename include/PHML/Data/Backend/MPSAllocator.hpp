#pragma once

#ifdef PHML_WITH_MPS

namespace PHML::Data {

class AllocatorRegistry;

// Constructs an MPSAllocator and registers it for Device::mps() in registry.
// Defined in src/Data/Backend/MPSAllocator.cpp.
void register_mps_allocator(AllocatorRegistry& registry);

} // namespace PHML::Data

#endif // PHML_WITH_MPS
