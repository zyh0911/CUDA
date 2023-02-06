add_library(libcode merge_sort_in_thrust.cu)
target_include_directories(libcode PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(libcode PRIVATE ${CUDA_LIBRARIES} )
target_compile_options(libcode PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)