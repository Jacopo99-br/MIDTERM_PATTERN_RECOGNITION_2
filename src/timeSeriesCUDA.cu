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


// =============================================================================
// KERNEL CUDA: search_kernel_SoA (True SoA / Accesso Coalescente)
// =============================================================================
__global__ void search_kernel_SoA(const double* __restrict__ d_data, 
                                  int num_series, 
                                  int series_len, 
                                  int query_len, 
                                  int* d_results) 
{
    // 1. Calcola l'indice globale del thread (rappresenta l'ID della serie: 0..2727)
    int series_idx = blockIdx.x * blockDim.x + threadIdx.x; //var built-in nvcc

    // 2. Controllo dei limiti per evitare accessi fuori matrice
    if (series_idx < num_series) {        
        int max_start_idx = series_len - query_len;
        
        // Registro locale per mantenere la distanza minima trovata
        double min_distance = DBL_MAX;
        int best_match_idx = -1;

        // 3. CICLO ESTERNO: Scorre le finestre i (da 0 a 5120 - query_len)
        for (int i = 0; i <= max_start_idx; ++i) {
            double current_dist = 0.0;

            // 4. CICLO INTERNO: Calcola la distanza con la query
            for (int j = 0; j < query_len; ++j) {
                
                // --- FORMULA TRUE SOA INTERLACCIATO ---
                // Calcola il punto temporale globale t = i + j
                // I punti di tutte le 2728 serie a quel tempo t sono memorizzati consecutivamente!
                int memory_idx = (i + j) * num_series + series_idx;
                
                // Lettura coalescente dalla VRAM
                double diff = d_data[memory_idx] - d_const_query[j];
                current_dist += diff * diff;

                // Pruning (Early Exit)
                if (current_dist >= min_distance) {
                    break;
                }
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
std::vector<int> CUDASearch_SoA(const TimeSeries_SoA& dataset, 
                               const std::vector<double>& query) 
{

    int num_series = dataset.all_data.size();        // CORRETTO: usa 'all_data'
    int series_len = dataset.serie_lenght;           // CORRETTO: usa 'serie_lenght'
    int total_elements = num_series * series_len;
    int query_len = query.size();

// Nel ciclo di trasposizione (CPU -> True SoA):
    /*
    std::vector<double> flat_data(total_elements);
    for (int s = 0; s < num_series; ++s) {
        for (int t = 0; t < series_len; ++t) {
            flat_data[t * num_series + s] = dataset.all_data[s][t]; // CORRETTO
        }
    }
    */
    // 2. Allocazione VRAM
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


// =============================================================================
// FUNZIONE HOST (CPU): Batch Multi-Query
// =============================================================================
std::vector<std::vector<int>> CUDAMultiQuerySearch_SoA(const TimeSeries_SoA& dataset, 
                                                      const std::vector<std::vector<double>>& queries) 
{
    std::vector<std::vector<int>> all_results;
    all_results.reserve(queries.size());

    for (const auto& single_query : queries) {
        all_results.push_back(CUDASearch_SoA(dataset, single_query));
    }

    return all_results;
}