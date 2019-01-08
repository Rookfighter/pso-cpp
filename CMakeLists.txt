# CMakeLists.txt
#
#     Author: Fabian Meyer
# Created On: 23 May 2018

cmake_minimum_required(VERSION 3.5)
project(psocpp)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
endif(CMAKE_COMPILER_IS_GNUCXX)

add_subdirectory(dep)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
    ${CATCH2_INCLUDE_DIR}
)

install(FILES include/psocpp.h DESTINATION include)

if(${BUILD_TESTS})
    # Compile tests
    SET(TEST_SRC
        "test/main.cpp"
        "test/test_pso.cpp"
    )
    add_executable(run_tests ${TEST_SRC})
endif(${BUILD_TESTS})
