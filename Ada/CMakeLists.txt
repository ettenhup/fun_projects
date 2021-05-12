cmake_minimum_required(VERSION 3.16)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

set(this_project ada)

# third party software
set(thirdparty dependencies)

project(${this_project} CXX)

#enable_testing()
#add_subdirectory(${thirdparty}/googletest)

set(headers
  src/DifferentiableFunction.h
  src/ElementalFunctions.h
  src/Node.h
  src/EdgeFunction.h
  )

# src files
set(sources
  src/ada_main.cpp
  )

# add the executable
add_executable(${this_project} ${sources} ${headers})
