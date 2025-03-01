cmake_minimum_required(VERSION 2.8.3)
project(unitree_legged_sdk_python_wrapper)

# check arch and os
message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
    set(ARCH amd64)
endif()
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64.*")
    set(ARCH arm64)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/cpp/${ARCH})
file(GLOB_RECURSE sources "cpp/*.cpp")

set(EXTRA_LIBS -pthread rt m libunitree_legged_sdk.so)
# set(CMAKE_CXX_FLAGS "-O3 -no-pie")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -fPIC")
set(CMAKE_CXX_STANDARD 14)
# python
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../lib/python/${ARCH}) # .so .dylib

# find_package(Python COMPONENTS Interpreter Development)
# message("-- Python version: ${Python_VERSION}")
# message("-- Python include dirs: ${Python_INCLUDE_DIRS}")
# message("-- Python libraries: ${Python_LIBRARIES}")

# set(Python_ROOT_DIR /usr)
# set(Python_EXECUTABLE /usr/bin/python3.8)
# set(PYTHON_EXECUTABLE /usr/bin/python3.8)
# set(PYBIND11_PYTHON_VERSION 3.8)

add_subdirectory(third-party/pybind11)

# find_package(pybind11 REQUIRED)

pybind11_add_module(aliengo_py aliengo_py.cpp)
target_link_libraries(aliengo_py PRIVATE ${EXTRA_LIBS} unitree_legged_sdk)
