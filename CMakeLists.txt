# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

cmake_minimum_required(VERSION 3.20)
project(swapnet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core
  src/cpp/graph.cpp
  src/cpp/qap.cpp
  src/cpp/milp.cpp
  src/cpp/node_swap.cpp
  src/cpp/pipeline.cpp
  src/cpp/bindings.cpp
)

target_include_directories(_core PRIVATE src/cpp)

if(MSVC)
  target_compile_options(_core PRIVATE /W4 /O2)
else()
  target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

install(TARGETS _core DESTINATION swapnet)
