function(phml_add_test test_name test_source)
	add_executable(${test_name} ${test_source})
	target_link_libraries(${test_name} PRIVATE PHML::PHML)
	add_test(NAME PHML_${test_name} COMMAND ${test_name})
endfunction()

set(PHML_TEST_SOURCES
    unit/Data/test_tensor.cpp
)

foreach(test_source IN LISTS PHML_TEST_SOURCES)
	get_filename_component(test_name ${test_source} NAME_WE)
	phml_add_test(${test_name} ${test_source})
endforeach()
