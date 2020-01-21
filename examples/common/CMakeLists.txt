# Collect source files
file(GLOB ONLINE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/on_*.cpp)
file(GLOB OFFLINE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/off_*.cpp)
set(COMMON_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/blocking_queue.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/queue.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/pipeline.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/data_provider.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/clas_processor.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/common_functions.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/gflags_common.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/simple_interface.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/runner_strategy.cpp")
set(CNRT_UTIL_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/../include/offline")
set(MLU_UTIL_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/../include/caffe/mlu")
set(COM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")

include_directories(${CNRT_INCLUDE_DIRS} ${CNRT_UTIL_INCLUDE_DIRS}
                    ${MLU_UTIL_INCLUDE_DIRS} ${COM_INCLUDE_DIRS}
                    ${Caffe_INCLUDE_DIRS})

add_library(OffComLib STATIC ${OFFLINE_SRCS} ${COMMON_SRCS})
add_library(OnComLib STATIC ${ONLINE_SRCS} ${COMMON_SRCS})

target_link_libraries(OffComLib ${Caffe_LINKER_LIBS})
target_link_libraries(OnComLib ${Caffe_LINK})
