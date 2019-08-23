# Collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE com_srcs ${CMAKE_CURRENT_SOURCE_DIR}/post_process/*.cpp)
set(POST_PROCESS_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/post_process")
set(CNRT_UTIL_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include/offline")
set(MLU_UTIL_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include/caffe/mlu")
set(COM_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/examples/common/include")

# Build each source file independently
foreach(source ${srcs})
  if(NOT (${source} MATCHES "post_process"))
    # get file name
    get_filename_component(name ${source} NAME_WE)

    # get folder name
    get_filename_component(path ${source} PATH)
    get_filename_component(folder ${path} NAME_WE)

    # target
    add_executable(${name} ${source} ${com_srcs})
    include_directories(${CNRT_INCLUDE_DIRS} ${CNRT_UTIL_INCLUDE_DIRS},
                        ${MLU_UTIL_INCLUDE_DIRS} ${POST_PROCESS_INCLUDE_DIRS}
                        ${COM_INCLUDE_DIRS} ${Caffe_INCLUDE_DIRS})
    target_link_libraries(${name} ${Caffe_LINK} ${CNRT_LIBS} OnComLib)
    caffe_default_properties(${name})

    # set back RUNTIME_OUTPUT_DIRECTORY
    caffe_set_runtime_directory(${name} "${PROJECT_BINARY_DIR}/examples/${folder}")
    caffe_set_solution_folder(${name} examples)

    # Install
    install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})

    if(UNIX OR APPLE)
      # Funny command to make tutorials work
      # TODO: remove in future as soon as naming is standardized everywhere
      set(__outname ${PROJECT_BINARY_DIR}/examples/${folder}/${name}${Caffe_POSTFIX})
      add_custom_command(TARGET ${name} POST_BUILD
                         COMMAND ln -sf "${__outname}" "${__outname}.bin")
    endif()
  endif()
endforeach()
