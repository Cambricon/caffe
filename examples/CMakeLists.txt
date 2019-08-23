cmake_minimum_required(VERSION 2.8.7)

set(Caffe_ALL_DEFINITIONS ${Caffe_DEFINITIONS})
list(REMOVE_ITEM Caffe_ALL_DEFINITIONS PRIVATE PUBLIC)
add_definitions(${Caffe_ALL_DEFINITIONS})

add_subdirectory(clas_offline_multicore)
add_subdirectory(clas_online_multicore)
add_subdirectory(clas_online_singlecore)
add_subdirectory(clas_offline_singlecore)
add_subdirectory(ssd)
add_subdirectory(yolo_v2)
add_subdirectory(yolo_v3)
add_subdirectory(common)
