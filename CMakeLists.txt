cmake_minimum_required(VERSION 3.10)

project(COMPUTERVISION-FINALPROJECT26)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

file(GLOB SRC_FILES src/*.cpp)

add_executable(main ${SRC_FILES})

# Include path
target_include_directories(main
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

# Link OpenCV
target_link_libraries(main
    PRIVATE
        ${OpenCV_LIBS}
)