cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 20)

add_executable(mrope-bug-cli main.cpp)

# Get the absolute path to dice.jpg
get_filename_component(DICE_IMAGE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dice.jpg" ABSOLUTE)

# Add it as a preprocessor definition
target_compile_definitions(mrope-bug-cli PRIVATE DICE_IMAGE_PATH="${DICE_IMAGE_PATH}")

target_include_directories(mrope-bug-cli PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/common"
        "${CMAKE_SOURCE_DIR}/ggml/include")
target_link_libraries(mrope-bug-cli llama mtmd common ggml)