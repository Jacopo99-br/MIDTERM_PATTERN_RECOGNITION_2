# Versione minima di CMake richiesta
cmake_minimum_required(VERSION 3.18)

# Nome del progetto
project(PRALLEL_PRJCT VERSION 1.0 LANGUAGES CXX)

#se non si specifica il tipo di build imposta 'release'. Attiva l'ottimizzazione -03
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Imposta lo standard C++ (es. C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenMP COMPONENTS CXX REQUIRED)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    # --- RAMO 1: GPU NVIDIA DISPONIBILE (es. Google Colab) ---
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)

    message(STATUS "[CUDA DETECTED] Abilitato supporto GPU CUDA.")

    # Includiamo tutti i sorgenti, compreso il file .cu
    add_executable(Project_TSPR
        src/main.cpp
        src/timeseries.cpp
        src/timeSeriesCUDA.cu
    )

    # Definiamo USE_CUDA per il codice C++ nel main.cpp
    target_compile_definitions(Project_TSPR PRIVATE USE_CUDA)

    # Passa -fopenmp al compilatore host quando nvcc compila
    target_compile_options(Project_TSPR PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)

    # Architettura: 75 per Tesla T4 (Colab), native se compili su una macchina con GPU locale
    if(DEFINED ENV{COLAB_GPU} OR NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_property(TARGET Project_TSPR PROPERTY CUDA_ARCHITECTURES 75)
    else()
        set_property(TARGET Project_TSPR PROPERTY CUDA_ARCHITECTURES native)
    endif()
else()
    # --- RAMO 2: SOLO CPU (es. PC Locale senza GPU NVIDIA) ---
    message(STATUS "[NO CUDA] Nessun compilatore CUDA trovato. Compilazione SOLO CPU (OpenMP).")

    # Includiamo solo i file C++ standard
    add_executable(Project_TSPR
        src/main.cpp
        src/timeseries.cpp
    )
endif()

# Directory degli header
target_include_directories(Project_TSPR PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Link di OpenMP all'eseguibile
target_link_libraries(Project_TSPR PUBLIC OpenMP::OpenMP_CXX)
# Forza il linking statico su Windows con GCC/MinGW per evitare conflitti di DLL
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(Project_TSPR PRIVATE -static -static-libgcc -static-libstdc++)
endif()

# Messaggio opzionale per vedere dove verrà creato l'eseguibile
message(STATUS "Build type impostata su: ${CMAKE_BUILD_TYPE}")
message(STATUS "Eseguibile creato in: ${CMAKE_BINARY_DIR}")