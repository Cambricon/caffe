# Try to find the MLU libraries and headers

#find_library doesn't work while cross-compiling arm64 on dev branch
#although it worked while I compile caffe with cnml on at branch
#right now we simplifying the library finding
set(CNML_INCLUDE_DIR ${NEUWARE_HOME}/include)
set(CNRT_INCLUDE_DIR ${NEUWARE_HOME}/include)
set(CNPLUGIN_INCLUDE_DIR ${NEUWARE_HOME}/include)

set(SUFFIX "64")
set(CNML_LIBRARY ${NEUWARE_HOME}/lib${SUFFIX}/libcnml.so)
set(CNRT_LIBRARY ${NEUWARE_HOME}/lib${SUFFIX}/libcnrt.so)
set(CNPLUGIN_LIBRARY ${NEUWARE_HOME}/lib${SUFFIX}/libcnplugin.so)

include(FindPackageHandleStandardArgs)
#find_package_handle_standard_args(MLU DEFAULT_MSG ${MLU_INCLUDE_DIRS} ${MLU_LIBS})
find_package_handle_standard_args(MLU DEFAULT_MSG CNML_INCLUDE_DIR CNRT_INCLUDE_DIR CNPLUGIN_INCLUDE_DIR CNML_LIBRARY CNRT_LIBRARY CNPLUGIN_LIBRARY)

if(MLU_FOUND)
  message(STATUS "Found MLU (include: ${CNML_INCLUDE_DIR} ${CNRT_INCLUDE_DIR} ${NPLUGIN_INCLUDE_DIR}
  library: ${CNML_LIBRARY} ${CNRT_LIBRARY} ${CNPLUGIN_LIBRARY})")
  caffe_parse_header(MLU_VERSION_LINES MLU_VERSION_MAJOR MLU_VERSION_MINOR MLU_VERSION_PATCH)
  set(MLU_VERSION "${MLU_VERSION_MAJOR}.${MLU_VERSION_MINOR}.${MLU_VERSION_PATCH}")
endif()
