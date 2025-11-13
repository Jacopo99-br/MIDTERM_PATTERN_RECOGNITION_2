# Versione minima di CMake richiesta
cmake_minimum_required(VERSION 3.10)

# Nome del progetto
project(MioProgetto VERSION 1.0)

# Imposta lo standard C++ (es. C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenMP REQUIRED)

# Trova i nostri file sorgente
# (Al momento solo main.cpp, ma potresti aggiungerne altri)
add_executable(
    Project_TSPR           # Questo sarà il nome del tuo file .exe o eseguibile
    src/main.cpp
    src/timeseries.cpp
)

target_include_directories(Project_TSPR PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(Project_TSPR PUBLIC OpenMP::OpenMP_CXX)

# Messaggio opzionale per vedere dove verrà creato l'eseguibile
message(STATUS "Eseguibile creato in: ${CMAKE_BINARY_DIR}")