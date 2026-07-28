#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include "timeSeriesCUDA.cuh"



// -----------------------------------------------------------------------------
// MEMORIA COSTANTE (Constant Memory)
// -----------------------------------------------------------------------------
__constant__ double d_const_query[MAX_QUERY_LEN];

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

double* uploadDatasetToGPU(const TimeSeries_SoA& dataset){
    int series_len = dataset.serie_length;
    int num_series = dataset.all_data_flat.size() / series_len;
    size_t total_bytes = num_series * series_len * sizeof(double);
    double* d_dataset = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_dataset, total_bytes));
    CHECK_CUDA(cudaMemcpy(d_dataset, dataset.all_data_flat.data(), total_bytes, cudaMemcpyHostToDevice));

    return d_dataset;
}

void freeGPUMemory(double* d_dataset) {
    if (d_dataset) {
        CHECK_CUDA(cudaFree(d_dataset));
    }
}


// =============================================================================
// KERNEL CUDA
// =============================================================================
__global__ void search_kernel_SoA(const double* __restrict__ d_data, 
                                  int num_series, 
                                  int series_len, 
                                  int query_len, 
                                  int* d_results) 
{
    // 1. Calcola l'indice globale del thread (rappresenta l'ID della serie: 0..2727)
    const int series_idx = blockIdx.x * blockDim.x + threadIdx.x; //var built-in nvcc

    // 2. Controllo dei limiti per evitare accessi fuori matrice
    if (series_idx < num_series) {        
        const int max_start_idx = series_len - query_len;
        
        // Registro locale per mantenere la distanza minima trovata
        double min_distance = DBL_MAX;
        int best_match_idx = -1;

        // 3. CICLO ESTERNO: Scorre le finestre i (da 0 a 5120 - query_len)
        for (int i = 0; i <= max_start_idx; ++i) {
            double current_dist = 0.0;
            const int base_idx = i * num_series + series_idx;
            // pre calcolo dell'offset per la finestra corrente

            // 4. CICLO INTERNO: Calcola la distanza con la query (SAD)
            #pragma unroll 4 
            for (int j = 0; j < query_len; ++j) {
                
                int memory_idx = base_idx + j * num_series; // Calcola l'indice di memoria per la finestra corrente
                
                // Lettura coalescente dalla VRAM
                double diff = d_data[memory_idx] - d_const_query[j];
                current_dist += fabs(diff);
            }

            // Salva il miglior match
            if (current_dist < min_distance) {
                min_distance = current_dist;
                best_match_idx = i;
            }
        }

        // Salva il risultato in VRAM
        d_results[series_idx] = best_match_idx;
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
    std::vector<std::vector<int>> all_results(num_queries, std::vector<int>(num_series));

    int* d_results = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_results, num_series * sizeof(int)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_series + threadsPerBlock - 1) / threadsPerBlock;  //threadsPerBlock -1 , serve per arrotondare verso l'alto
    // totale: 11 blocchi x 256 thread --> 2816 thread, ma ne servono solo 2728, quindi alcuni thread non faranno nulla

    
    std::cout << "[CUDA] num_series calcolato: " << num_series 
              << " | series_len: " << series_length 
              << " | num_queries: " << num_queries 
              << " | query_len: " << query_len << std::endl;
    
    for (int q = 0; q < num_queries; ++q){

        // query nella constant memory
        CHECK_CUDA(cudaMemcpyToSymbol(d_const_query, queries[q].data(), query_len * sizeof(double)));
        // lancia il kernel per la query corrente
        search_kernel_SoA<<<blocksPerGrid, threadsPerBlock>>>(d_dataset, num_series, series_length, query_len, d_results);
        //recupera i risultati
        CHECK_CUDA(cudaMemcpy(all_results[q].data(), d_results, num_series * sizeof(int), cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_results));
    return all_results;
}