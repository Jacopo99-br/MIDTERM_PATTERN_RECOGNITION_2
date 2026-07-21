#ifndef TIMESERIES_CUDA_CUH
#define TIMESERIES_CUDA_CUH

#include <vector>
#include "timeseries.h" // Per accedere alla struttura TimeSeries_SoA se necessaria

// ============================================================================
// DICHIARAZIONE FUNZIONI WRAPPER GPU (Interfaccia C++)
// Queste funzioni vengono chiamate da main.cpp
// ============================================================================

// Ricerca di una singola query su GPU
std::vector<int> CUDASearch_SoA(const TimeSeries_SoA& dataset, 
                           const std::vector<double>& single_query);

// Ricerca parallela di query multiple su GPU
std::vector<std::vector<int>> CUDAMultiQuerySearch_SoA(const TimeSeries_SoA& dataset, 
                                             const std::vector<std::vector<double>>& all_queries);

#endif // TIMESERIES_CUDA_CUH