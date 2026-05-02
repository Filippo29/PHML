target_sources(PHML
    PRIVATE
        core/version.cpp
        Data/Tensor.cpp
)

if(PHML_WITH_MPS)
    target_sources(PHML
        PRIVATE
            Data/Backend/MPSAllocator.cpp
            Data/Backend/MetalContext.cpp
            Data/Backend/MPSOps.cpp
    )
endif()
