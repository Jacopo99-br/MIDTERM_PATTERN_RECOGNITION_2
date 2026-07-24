# Versione minima di CMake richiesta
cmake_minimum_required(VERSION 3.18)

# Nome del progetto
project(MioProgetto VERSION 1.0 LANGUAGES CXX CUDA)

#se non si specifica il tipo di build imposta 'release'. Attiva l'ottimizzazione -03
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Imposta lo standard C++ (es. C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

find_package(OpenMP COMPONENTS CXX REQUIRED)

# Trova i nostri file sorgente
# (Al momento solo main.cpp, ma potresti aggiungerne altri)
add_executable(
    Project_TSPR
    src/main.cpp
    src/timeseries.cpp
    src/timeSeriesCUDA.cu
)

target_include_directories(Project_TSPR PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(Project_TSPR PUBLIC OpenMP::OpenMP_CXX)

set_property(TARGET Project_TSPR PROPERTY CUDA_ARCHITECTURES native)

# Messaggio opzionale per vedere dove verrà creato l'eseguibile
message(STATUS "Build type impostata su: ${CMAKE_BUILD_TYPE}")
message(STATUS "Eseguibile creato in: ${CMAKE_BINARY_DIR}")