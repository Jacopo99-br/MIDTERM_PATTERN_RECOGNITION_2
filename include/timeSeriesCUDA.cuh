#ifndef TIMESERIES_CUDA_CUH
#define TIMESERIES_CUDA_CUH

#include <vector>
#include "timeseries.h"

// Dimensione massima della query memorizzabile nella Constant Memory della GPU (es. 1024 double)
#ifndef MAX_QUERY_LEN
#define MAX_QUERY_LEN 1024
#endif

#define DATASET_NUM_SERIES 2728
#define DATASET_SERIES_LEN 5120

double* uploadDatasetToGPU(const TimeSeries_SoA& dataset);
double* uploadQueriesToGPU(const std::vector<std::vector<double>>& queries);

void freeGPUMemory(double* d_ptr);

std::vector<int> CUDASearch_SoA(const TimeSeries_SoA& dataset, 
                               const std::vector<double>& query);




std::vector<std::vector<int>> CUDAMultiQuerySearch_SoA(const double* d_dataset, 
                                                       const std::vector<std::vector<double>>& queries, 
                                                       int num_series, 
                                                       int series_length);

#endif // TIMESERIES_CUDA_CUH