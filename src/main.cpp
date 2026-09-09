#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <random>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "timeseries.h"

// Inclusione condizionale: attivata solo se compilato con supporto CUDA (es. CMake su Colab)
#if defined(USE_CUDA) || defined(__CUDACC__)
    #include <cuda_runtime.h>
    #include "timeSeriesCUDA.cuh"
    #define HAS_CUDA 1
#else
    #define HAS_CUDA 0
#endif

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

// Struttura per memorizzare le metriche statistiche raccomandate
struct BenchmarkStats {
    double mean_ms;
    double stddev_ms;
    double min_ms;
    double max_ms;
};

// Calcolo di media, deviazione standard campionaria, minimo e massimo
BenchmarkStats computeStats(const vector<double>& times_ms) {
    if (times_ms.empty()) return {0.0, 0.0, 0.0, 0.0};
    
    double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    double mean = sum / times_ms.size();

    double accum = 0.0;
    for (double t : times_ms) {
        accum += (t - mean) * (t - mean);
    }
    double variance = (times_ms.size() > 1) ? (accum / (times_ms.size() - 1)) : 0.0;
    double stddev = std::sqrt(variance);

    auto [min_it, max_it] = std::minmax_element(times_ms.begin(), times_ms.end());
    return {mean, stddev, *min_it, *max_it};
}

// Verifica la disponibilità runtime di una GPU senza terminare l'applicazione in caso di assenza
bool isGpuAvailable() {
#if HAS_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaSuccess && device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        cout << "[HW DETECT] GPU Rilevata: " << prop.name << " (" << prop.multiProcessorCount << " SMs)" << endl;
        return true;
    }
    cudaGetLastError(); // Reset dell'eventuale flag di errore del driver
#endif
    return false;
}

// Genera la lista dei thread da esplorare (es. 1, 2, 4, 8, ... fino al massimo di sistema)
vector<int> getThreadCountsToTest() {
    int max_threads = omp_get_num_procs();
    vector<int> counts;
    for (int t = 1; t <= max_threads; t *= 2) {
        counts.push_back(t);
    }
    if (counts.empty() || counts.back() != max_threads) {
        counts.push_back(max_threads);
    }
    return counts;
}

// Funzione di validazione con gestione delle parità di minima distanza (minimi equivalenti)
bool validateResults(const std::vector<std::vector<int>>& cpu_results,
                     const std::vector<std::vector<int>>& gpu_results,
                     const TimeSeries_SoA& dataset,
                     const std::vector<std::vector<double>>& queries,
                     double tolerance = 1e-4) 
{
    int errors = 0;
    int false_positives_ties = 0;
    int exact_matches = 0; 
    int num_queries = queries.size();
    int num_series = dataset.all_classes.size();
    int series_len = dataset.serie_length;
    int total_elements = num_queries * num_series;

    for (int q = 0; q < num_queries; ++q) {
        int query_len = queries[q].size();

        for (int s = 0; s < num_series; ++s) {
            int cpu_idx = cpu_results[q][s];
            int gpu_idx = gpu_results[q][s];

            if (std::abs(cpu_idx - gpu_idx) <= 2) {
                exact_matches++;
                continue;
            }

            const double* series_ptr = &dataset.all_data_flat[s * series_len];
            double cpu_sad = 0.0;
            double gpu_sad = 0.0;

            for (int j = 0; j < query_len; ++j) {
                cpu_sad += std::abs(series_ptr[cpu_idx + j] - queries[q][j]);
                gpu_sad += std::abs(series_ptr[gpu_idx + j] - queries[q][j]);
            }

            if (std::abs(cpu_sad - gpu_sad) <= tolerance) {
                false_positives_ties++;
            } else {
                if (errors < 5) {
                    std::cerr << "  DISCREPANZA REALE Query [" << q << "], Serie [" << s << "]: "
                              << "CPU_idx=" << cpu_idx << " (SAD: " << cpu_sad << ") | "
                              << "GPU_idx=" << gpu_idx << " (SAD: " << gpu_sad << ")" << std::endl;
                }
                errors++;
            }
        }
    }

    if (errors == 0) {
        std::cout << "  [Validazione] SUPERATA (" << exact_matches << " exact matches, " 
                  << false_positives_ties << " minimi equivalenti su " << total_elements << ")" << std::endl;
        return true;
    } else {
        double error_rate = (double)errors / total_elements * 100.0;
        std::cerr << "  [Validazione] FALLITA: " << errors << " / " << total_elements 
                  << " discrepanze (" << error_rate << "%)." << std::endl;
        return false;
    }
}

int main() {
    // --- 1. IDENTIFICAZIONE HARDWARE ---
    int num_cores = omp_get_num_procs();
    cout << "======================================================\n";
    cout << "[HW DETECT] Thread logici CPU disponibili: " << num_cores << endl;
    
    bool run_gpu = isGpuAvailable();
    if (!run_gpu) {
        cout << "[HW DETECT] Esecuzione configurata SOLO SU CPU (OpenMP)." << endl;
    }
    cout << "======================================================\n";

    // --- 2. LOCALIZZAZIONE DATASET ---
    fs::path dataset_file = "FaultDetectionA_TEST.ts";
    fs::path input_path;

    if (fs::exists(fs::path("FaultDetectionA") / dataset_file)) {
        // Se lanciato dalla radice del progetto (MIDTERM_PATTERN_RECOGNITION)
        input_path = fs::path("FaultDetectionA") / dataset_file;
    } else if (fs::exists(dataset_file)) {
        // Se lanciato direttamente dentro FaultDetectionA
        input_path = dataset_file;
    } else if (fs::exists(fs::path("..") / "FaultDetectionA" / dataset_file)) {
        // Se lanciato dall'interno della cartella build/
        input_path = fs::path("..") / "FaultDetectionA" / dataset_file;
    } else {
        std::cerr << "ERRORE: File dataset non trovato in FaultDetectionA/" << dataset_file.string() << std::endl;
        std::cerr << "Cartella corrente: " << fs::current_path().string() << std::endl;
        return 1;
    }
    string path = input_path.string();
    mt19937 gen(1337); // Seed fisso per generare identiche query tra esecuzioni diverse

    // Parametri sperimentali
    vector<int> query_numbers = {5, 50};
    //vector<int> query_numbers = {5, 10};
    vector<int> query_lengths = {100, 500, 1000};
    //vector<int> query_lengths = {10, 50, 60};
    vector<int> thread_counts = getThreadCountsToTest();

    const int WARMUP_RUNS = 1; // Scarto cold-cache
    const int NUM_RUNS = 5;    // Ripetizioni per media/stddev

    // --- 3. CARICAMENTO E PREPARAZIONE DATI (Escluso dai timer) ---
    cout << "Caricamento dataset: " << path << endl;
    vector<string> raw_data_lines = loadRawDataset(path);
    vector<TimeSeries> datasetAoS = loadDatasetAoS(raw_data_lines);
    TimeSeries_SoA datasetSoA = loadDatasetSoA(raw_data_lines);

    int series_len = datasetSoA.serie_length;
    int num_series = datasetSoA.all_data_flat.size() / series_len;
    cout << "Dataset pronto: " << num_series << " serie da " << series_len << " campioni ciascuna." << endl;

    // Caricamento GPU (se attiva)
    const double* d_dataset_gpu = nullptr;
#if HAS_CUDA
    if (run_gpu) {
        cout << "[GPU] Caricamento dataset nella VRAM..." << endl;
        d_dataset_gpu = uploadDatasetToGPU(datasetSoA);
    }
#endif

    // --- 4. PREPARAZIONE FILE CSV ---
    fs::path output_dir = fs::path("src");
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }
    ofstream outFile((output_dir / "Search_results.csv").string());
    if (!outFile.is_open()) {
        cerr << "Impossibile creare Search_results.csv in " << output_dir.string() << endl;
        return 1;
    }
    outFile << "Platform,Format,QueryLength,NumQueries,Threads,Mean_MS,StdDev_MS,Min_MS,Max_MS\n";

    // --- 5. BENCHMARKING SU PARAMETRI ---
    for (int query_l : query_lengths) {
        for (int query_n : query_numbers) {
            cout << "\n------------------------------------------------------\n";
            cout << "Configurazione: Query Len = " << query_l << " | Numero Queries = " << query_n << endl;
            cout << "------------------------------------------------------\n";

            // Creazione del batch di query casuali estratte dal dataset
            vector<vector<double>> all_queries;
            for (int q = 0; q < query_n; ++q) {
                all_queries.push_back(RandomQuery(datasetSoA, query_l, gen));
            }

            // Benchmark CPU: esplorazione thread (1 = baseline sequenziale per speedup)
            for (int t : thread_counts) {
                omp_set_num_threads(t);
                cout << "  -> OpenMP [Threads: " << t << "]..." << endl;

                // Test AoS
                for (int w = 0; w < WARMUP_RUNS; ++w) {
                    MultiQueryParallelSearch_AoS(datasetAoS, all_queries);
                }
                vector<double> times_AoS;
                for (int r = 0; r < NUM_RUNS; ++r) {
                    auto start = high_resolution_clock::now();
                    MultiQueryParallelSearch_AoS(datasetAoS, all_queries);
                    auto end = high_resolution_clock::now();
                    times_AoS.push_back(duration<double, milli>(end - start).count());
                }
                BenchmarkStats s_aos = computeStats(times_AoS);
                outFile << "OpenMP,AoS," << query_l << "," << query_n << "," << t << ","
                        << s_aos.mean_ms << "," << s_aos.stddev_ms << "," 
                        << s_aos.min_ms << "," << s_aos.max_ms << "\n";

                // Test SoA
                for (int w = 0; w < WARMUP_RUNS; ++w) {
                    MultiQueryParallelSearch_SoA(datasetSoA, all_queries);
                }
                vector<double> times_SoA;
                for (int r = 0; r < NUM_RUNS; ++r) {
                    auto start = high_resolution_clock::now();
                    MultiQueryParallelSearch_SoA(datasetSoA, all_queries);
                    auto end = high_resolution_clock::now();
                    times_SoA.push_back(duration<double, milli>(end - start).count());
                }
                BenchmarkStats s_soa = computeStats(times_SoA);
                outFile << "OpenMP,SoA," << query_l << "," << query_n << "," << t << ","
                        << s_soa.mean_ms << "," << s_soa.stddev_ms << "," 
                        << s_soa.min_ms << "," << s_soa.max_ms << "\n";

                outFile.flush();
            }

            // Benchmark GPU: eseguito solo se compilato con CUDA e se una scheda NVIDIA è presente
#if HAS_CUDA
            if (run_gpu) {
                cout << "  -> GPU CUDA Benchmark..." << endl;
                
                // Warm-up GPU
                for (int w = 0; w < WARMUP_RUNS; ++w) {
                    CUDAMultiQuerySearch_SoA(d_dataset_gpu, all_queries, num_series, series_len);
                }

                vector<double> times_CUDA;
                vector<vector<int>> res_CUDA;
                for (int r = 0; r < NUM_RUNS; ++r) {
                    auto start = high_resolution_clock::now();
                    res_CUDA = CUDAMultiQuerySearch_SoA(d_dataset_gpu, all_queries, num_series, series_len);
                    auto end = high_resolution_clock::now();
                    times_CUDA.push_back(duration<double, milli>(end - start).count());
                }
                BenchmarkStats s_cuda = computeStats(times_CUDA);
                outFile << "CUDA,SoA," << query_l << "," << query_n << ",GPU,"
                        << s_cuda.mean_ms << "," << s_cuda.stddev_ms << "," 
                        << s_cuda.min_ms << "," << s_cuda.max_ms << "\n";
                outFile.flush();

                // Validazione correttezza CPU SoA vs GPU CUDA
                vector<vector<int>> res_cpu = MultiQueryParallelSearch_SoA(datasetSoA, all_queries);
                validateResults(res_cpu, res_CUDA, datasetSoA, all_queries, 1e-4);
            }
#endif
        }
    }

    outFile.close();
    cout << "\nBenchmark terminato. Risultati registrati in: " << (output_dir / "Search_results.csv").string() << endl;

#if HAS_CUDA
    if (run_gpu && d_dataset_gpu) {
        cout << "[GPU] Rilascio della memoria VRAM..." << endl;
        freeGPUMemory(const_cast<double*>(d_dataset_gpu));
    }
#endif

    return 0;
}