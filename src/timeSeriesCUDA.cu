#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include "timeSeriesCUDA.cuh"



// -----------------------------------------------------------------------------
// MEMORIA COSTANTE (Constant Memory)
// -----------------------------------------------------------------------------
//__constant__ double d_const_query[MAX_QUERY_LEN];

// -----------------------------------------------------------------------------
// MACRO PER IL CONTROLLO DEGLI ERRORI CUDA
// -----------------------------------------------------------------------------
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct alignas(16) ReductionItem{
    double dist;
    int idx;
    // 4 byte di padding automatici inseriti dal compilatore per allineare la struttura a 16 byte
};

double* uploadDatasetToGPU(const TimeSeries_SoA& dataset){
    int series_len = dataset.serie_length;
    int num_series = dataset.all_data_flat.size() / series_len;
    size_t total_bytes = num_series * series_len * sizeof(double);
    
    double* d_dataset = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_dataset, total_bytes));
    CHECK_CUDA(cudaMemcpy(d_dataset, dataset.all_data_flat.data(), total_bytes, cudaMemcpyHostToDevice));

    return d_dataset;
}

double* uploadQueriesToGPU(const std::vector<std::vector<double>>& queries) {
    int num_queries = queries.size();
    int query_len = queries[0].size();
    size_t total_bytes = num_queries * query_len * sizeof(double);

    // Si appiattisce le query in un unico buffer continuo
    std::vector<double> h_flat_queries(num_queries * query_len);
    for (int q = 0; q < num_queries; ++q) {
        for (int j = 0; j < query_len; ++j) {
            h_flat_queries[q * query_len + j] = queries[q][j];
        }
    }

    double* d_queries = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_queries, total_bytes));
    CHECK_CUDA(cudaMemcpy(d_queries, h_flat_queries.data(), total_bytes, cudaMemcpyHostToDevice));

    return d_queries;
}

void freeGPUMemory(double* d_ptr) {
    if (d_ptr) {
        CHECK_CUDA(cudaFree(d_ptr));
    }
}

// =============================================================================
// KERNEL CUDA 2D
// =============================================================================
__global__ void search_kernel_SoA(const double* __restrict__ d_data, 
                                  const double* __restrict__ d_queries, 
                                  int num_series, 
                                  int series_len, 
                                  int query_len, 
                                  int* d_results) 
{
    const int series_idx = blockIdx.x;
    const int query_idx = blockIdx.y; // Indice della query corrente

    // 2. Controllo dei limiti per evitare accessi fuori matrice
    if (series_idx < num_series) {        
        extern __shared__ double s_mem[]; // Memoria condivisa per la query corrente
        double* s_series = s_mem;
        
        size_t series_bytes = series_len * sizeof(double);
        size_t aligned_offset_bytes = (series_bytes + 15) & ~15;
        
        ReductionItem* s_red = (ReductionItem*)(s_mem + aligned_offset_bytes / sizeof(double));

        const int series_offset = series_idx * series_len;
        for (int t = threadIdx.x; t < series_len; t += blockDim.x) {
            s_series[t] = d_data[series_offset + t]; // Accesso SoA
        }

        __syncthreads(); // Sincronizzazione dei thread nel blocco

    // -------------------------------------------------------------------------
    // 2. CALCOLO SLIDING WINDOW (100% Shared Memory)
    // -------------------------------------------------------------------------
        const int query_offset = query_idx * query_len;
        const int max_start = series_len - query_len;
        double min_distance = DBL_MAX;
        int best_match_idx = -1;

        for (int i = threadIdx.x; i <= max_start; i += blockDim.x) {
            double distance = 0.0;

            #pragma unroll 4
            for (int j = 0; j < query_len; ++j) {
                double diff = s_series[i + j] - d_queries[query_offset + j];
                distance += fabs(diff); // L1 distance
            }
            if (distance < min_distance) {
                min_distance = distance;
                best_match_idx = i;
            }
        }

        s_red[threadIdx.x].dist = min_distance;
        s_red[threadIdx.x].idx = best_match_idx;
        __syncthreads();

        for(int stride = blockDim.x /2 ; stride >0; stride /=2){ //RIDUZIONE VISTA ALBERO
            if(threadIdx.x < stride){
                if(s_red[threadIdx.x + stride].dist < s_red[threadIdx.x].dist){
                    s_red[threadIdx.x] = s_red[threadIdx.x + stride];
                }
            }
            __syncthreads();
        }

        if(threadIdx.x == 0){
            int out_idx = query_idx * num_series + series_idx;
            d_results[out_idx] = s_red[0].idx;
        }
    }
}


// =============================================================================
// FUNZIONE HOST (CPU): CUDASearch_SoA
// =============================================================================

/*
std::vector<int> CUDASearch_SoA(const TimeSeries_SoA& dataset, 
                               const std::vector<double>& query) 
{

    int num_series = 0;
    int series_len = dataset.serie_length;
    if (series_len > 0 && !dataset.all_data_flat.empty()) {
        num_series = dataset.all_data_flat.size() / series_len;
    } else if (!dataset.all_data.empty()) {
        num_series = dataset.all_data.size();
    }     
               // CORRETTO: usa 'serie_length'
    
    int query_len = query.size();
    int total_elements = num_series * series_len;
// Nel ciclo di trasposizione (CPU -> True SoA):
    /*
    std::vector<double> flat_data(total_elements);
    for (int s = 0; s < num_series; ++s) {
        for (int t = 0; t < series_len; ++t) {
            flat_data[t * num_series + s] = dataset.all_data[s][t]; // CORRETTO
        }
    }
    */

/*
    double* d_data = nullptr;
    int* d_results = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_data, total_elements * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_results, num_series * sizeof(int)));

    // 3. Trasferimento dati da RAM a VRAM e Query in Constant Memory
    //CHECK_CUDA(cudaMemcpy(d_data, flat_data.data(), total_elements * sizeof(double), cudaMemcpyHostToDevice));
    // Se all_data_flat è già trasposta in formato SoA:
    CHECK_CUDA(cudaMemcpy(d_data, dataset.all_data_flat.data(), total_elements * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(d_const_query, query.data(), query_len * sizeof(double)));

    // 4. Configurazione Griglia/Blocchi (11 blocchi da 256 thread)
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_series + threadsPerBlock - 1) / threadsPerBlock;

    // 5. Esecuzione del Kernel
    search_kernel_SoA<<<blocksPerGrid, threadsPerBlock>>>(d_data, num_series, series_len, query_len, d_results);
    
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 6. Recupero risultati
    std::vector<int> results(num_series);
    CHECK_CUDA(cudaMemcpy(results.data(), d_results, num_series * sizeof(int), cudaMemcpyDeviceToHost));

    // 7. Pulizia memoria
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_results));

    return results;
}
*/

// =============================================================================
// FUNZIONE HOST (CPU): Batch Multi-Query
// =============================================================================
std::vector<std::vector<int>> CUDAMultiQuerySearch_SoA(const double* d_dataset,
                                                       const std::vector<std::vector<double>>& queries, 
                                                       int num_series, 
                                                       int series_length) 
{
    int num_queries = queries.size();
    int query_len = queries[0].size(); // Assuming all queries have the same length
    //std::vector<std::vector<int>> all_results(num_queries, std::vector<int>(num_series));
    double* d_queries = uploadQueriesToGPU(queries);

    int* d_results = nullptr;
    size_t results_bytes = num_queries * num_series * sizeof(int);
    CHECK_CUDA(cudaMalloc((void**)&d_results, results_bytes));

    int threadsPerBlock = 128;
    dim3 blocksPerGrid(num_series, num_queries); // per definire dimensioni blocchi e grighlie di thread lungo X,Y,Z

    size_t series_bytes = series_length * sizeof(double);
    size_t pitched_series_bytes = (series_bytes + 15) & ~15; // Allineamento a 16 byte
    size_t sharedMemBytes = pitched_series_bytes + (threadsPerBlock * sizeof(ReductionItem)); // Memoria condivisa per la serie + riduzione
    
    std::cout << "[CUDA] num_series calcolato: " << num_series 
              << " | series_len: " << series_length 
              << " | num_queries: " << num_queries 
              << " | query_len: " << query_len << std::endl;
    
    search_kernel_SoA<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(d_dataset, d_queries, num_series, series_length, query_len, d_results);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int>h_flat_results(num_queries * num_series);
    CHECK_CUDA(cudaMemcpy(h_flat_results.data(), d_results, results_bytes, cudaMemcpyDeviceToHost));

    std::vector<std::vector<int>> all_results(num_queries, std::vector<int>(num_series));
    for (int q = 0; q < num_queries; ++q) {
        for (int s = 0; s < num_series; ++s) {
            all_results[q][s] = h_flat_results[q * num_series + s];
        }
    }

    CHECK_CUDA(cudaFree(d_queries));
    CHECK_CUDA(cudaFree(d_results));
    return all_results;
}