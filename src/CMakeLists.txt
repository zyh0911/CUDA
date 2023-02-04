add_subdirectory(code)
add_subdirectory(utility)
add_executable(src main.cpp)
target_include_directories(src PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(src PRIVATE libcode libutil)
target_compile_options(src PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -O3>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)