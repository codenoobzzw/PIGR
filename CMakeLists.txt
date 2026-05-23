cmake_minimum_required(VERSION 3.16)
project(pigr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Keep runtime outputs aligned with original project layout.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER ${cfg} cfg_upper)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_SOURCE_DIR}/bin)
endforeach()

option(PIGR_NATIVE_OPTIMIZATION "Enable native SIMD/CPU optimization flags" ON)

# Original project relies on OpenMP in graph/refiner code paths.
find_package(OpenMP REQUIRED)

add_library(pigr_runner_common INTERFACE)
target_include_directories(pigr_runner_common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(pigr_runner_common INTERFACE PIGR_ENABLE_REFINER=1)
target_link_libraries(pigr_runner_common INTERFACE OpenMP::OpenMP_CXX)

# Keep parity with original experiment targets for SIMD-heavy distance kernels.
if(PIGR_NATIVE_OPTIMIZATION AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pigr_runner_common INTERFACE -Ofast -march=native -mtune=native)
endif()

# On Linux, some toolchains still require explicit librt for clock/time APIs.
if(UNIX AND NOT APPLE)
  target_link_libraries(pigr_runner_common INTERFACE rt)
endif()

function(add_pigr_runner target_name source_file)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} PRIVATE pigr_runner_common)
endfunction()

add_pigr_runner(run_hnsw_final src/run_hnsw_final.cpp)
add_pigr_runner(run_nsg_final  src/run_nsg_final.cpp)
add_pigr_runner(run_hcnng_final src/run_hcnng_final.cpp)

add_executable(build_indices src/build_indices.cpp)
target_link_libraries(build_indices PRIVATE pigr_runner_common)

# Notes:
# - Runners are intentionally minimal: only set paths/knobs then call pigr::refiner::<algo>::run(cfg).
# - Refiner code paths are enabled through the PIGR_ENABLE_REFINER compile definition.
