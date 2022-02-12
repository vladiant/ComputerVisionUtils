cmake_minimum_required(VERSION 3.10)

project(ImageViewGtk3 LANGUAGES C)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

add_executable(${PROJECT_NAME} main.c)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
)

target_compile_definitions(
    ${PROJECT_NAME}
    PRIVATE
        ${GTK3_DEFINITIONS}
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
        ${GTK3_INCLUDE_DIRS}
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        ${GTK3_LIBRARIES}
)
