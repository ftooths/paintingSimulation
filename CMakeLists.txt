cmake_minimum_required(VERSION 3.20)
project(paintcore VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PAINTCORE_BUILD_TESTS "Build C++20 native unit & physics tests" ON)
option(PAINTCORE_WITH_OPENMP "Enable OpenMP multi-core acceleration" ON)

if(PAINTCORE_WITH_OPENMP)
    find_package(OpenMP QUIET)
endif()

# Shared / Dynamic Native Library (paintcore.dll / libpaintcore.so)
add_library(paintcore SHARED
    cpp/src/paintcore_c_api.cpp
)
target_include_directories(paintcore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

if(MSVC)
    target_compile_options(paintcore PRIVATE /O2 /W4 /arch:AVX2 /fp:fast)
else()
    target_compile_options(paintcore PRIVATE -O3 -Wall -Wextra -march=native -ffast-math)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(paintcore PUBLIC OpenMP::OpenMP_CXX)
endif()

# Native Verification Test Suite Executable
if(PAINTCORE_BUILD_TESTS)
    enable_testing()
    add_executable(test_paintcore
        cpp/tests/test_paintcore_main.cpp
    )
    target_include_directories(test_paintcore PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    )
    if(MSVC)
        target_compile_options(test_paintcore PRIVATE /O2 /W4 /arch:AVX2)
    else()
        target_compile_options(test_paintcore PRIVATE -O3 -Wall -Wextra)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_paintcore PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME PaintCoreSuite COMMAND test_paintcore)
endif()
