#include <iostream>
#include <vector>
//#include "timeseries.h" 
#include <omp.h>
#include <chrono>
#include <random>
#include <fstream>
#include <filesystem>
#include <cstdlib> // per system()
#include <timeSeriesCUDA.cuh>

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;
int main() {
    fs::path dataset_file = "FaultDetectionA_TEST.ts";
    fs::path input_path;

    if (fs::exists(dataset_file)) {
        input_path = dataset_file;
    } 
    else if (fs::exists(fs::path("..") / "FaultDetectionA" / dataset_file)) {
        input_path = fs::path("..") / "FaultDetectionA" / dataset_file;
    }
    else if (fs::exists(fs::path("..") / dataset_file)) {
        input_path = fs::path("..") / dataset_file;
    }
    //string path = "C:\\Users\\jbrus\\Documents\\UNI\\MAGISTRALE\\PARALLEL\\PROGETTI\\MIDTERM_PATTERN_RECOGNITION\\FaultDetectionA\\FaultDetectionA_TEST.ts";
    // https://www.timeseriesclassification.com/dataset.php?train=&test=%3E1000&leng=&class=&type=
    string path = input_path.string();
    std::random_device rd;
    std::mt19937 gen(rd());
    //const int NUM_QUERIES = 5;
    //const int QUERY_LENGTH = 100;
    vector<int> query_numbers = {5, 50, 100};
    vector<int> query_lengths = {100, 500, 1000, 3000};
    /*
    vector<int> query_numbers = {5, 10};
    vector<int> query_lengths = {100};
    */

    vector<string> raw_data_lines = loadRawDataset(path);

    vector<TimeSeries> datasetAoS = loadDatasetAoS(raw_data_lines);
    TimeSeries_SoA datasetSoA = loadDatasetSoA(raw_data_lines);
    cout << "Dataset loaded in both AoS and SoA formats." << endl;

    struct TimingPair{
        duration<double, milli> single_q;
        duration<double, milli> multi_q;
        duration<double, milli> single_q_cuda;
        duration<double, milli> multi_q_cuda;
    };

    // Matrici per i risultati [Lunghezza][NumeroQuery]
    vector<vector<TimingPair>> matrixAoS(query_lengths.size(), vector<TimingPair>(query_numbers.size()));
    vector<vector<TimingPair>> matrixSoA(query_lengths.size(), vector<TimingPair>(query_numbers.size()));

    for (size_t i = 0; i < query_lengths.size(); ++i){
        int query_l = query_lengths[i];

        for (size_t j = 0; j < query_numbers.size(); ++j){
            int query_n = query_numbers[j];
            cout << "Query length: " << query_l << ", Number of queries: " << query_n << endl;

            vector<vector<double>> all_queries;  // Query generate randomiche basate sui dati del dataset
            for(int i = 0; i < query_n; ++i) {
                all_queries.push_back(RandomQuery(datasetSoA, query_l, gen));
            }
            cout << "Create " << all_queries.size() << " query of lenght: " << query_l << endl;

            vector<double> single_query = all_queries[0];

            // Time AoS SingleSearch
            cout << "Parallel SingleSearch on AoS..." << endl;
            auto start_SingleAoS_search = high_resolution_clock::now();
            vector<int> risultati_SingleAoS = ParallelSearch_AoS(datasetAoS, single_query);
            auto end_SingleAoS_search = high_resolution_clock::now();
            duration<double, milli> time_SingleAoS_search = end_SingleAoS_search - start_SingleAoS_search;
            
            // Time SoA SingleSearch
            cout << "Parallel SingleSearch on SoA..." << endl;
            auto start_SingleSoA_search = high_resolution_clock::now();
            vector<int> risultati_SingleSoA = ParallelSearch_SoA(datasetSoA, single_query);
            auto end_SingleSoA_search = high_resolution_clock::now();
            duration<double, milli> time_SingleSoA_search = end_SingleSoA_search - start_SingleSoA_search;

            // Time AoS MultiSearch
            cout << "Parallel MultiSearch on AoS..." << endl;
            auto start_MultiAoS_search = high_resolution_clock::now();
            vector<vector<int>> risultati_MultiAoS = MultiQueryParallelSearch_AoS(datasetAoS, all_queries);
            auto end_MultiAoS_search = high_resolution_clock::now();
            duration<double, milli> time_MultiAoS_search = end_MultiAoS_search - start_MultiAoS_search;
            
            // Time SoA MultiSearch
            cout << "Parallel MultiSearch on SoA..." << endl;
            auto start_MultiSoA_search = high_resolution_clock::now();
            vector<vector<int>> risultati_MultiSoA = MultiQueryParallelSearch_SoA(datasetSoA, all_queries);
            auto end_MultiSoA_search = high_resolution_clock::now();
            duration<double, milli> time_MultiSoA_search = end_MultiSoA_search - start_MultiSoA_search;

            //----------------------------------------------------
                //CUDA Search on SoA
            cout << "CUDA Search on SoA..." << endl;
            //----------------------------------------------------
            // Time CUDA SingleSearch
            auto start_CUDA_SingleSoA_search = high_resolution_clock::now();
            vector<int> risultati_CUDA_SingleSoA = CUDASearch_SoA(datasetSoA, single_query);
            auto end_CUDA_SingleSoA_search = high_resolution_clock::now();
            duration<double, milli> time_CUDA_SingleSoA_search = end_CUDA_SingleSoA_search - start_CUDA_SingleSoA_search;
            // Time CUDA MultiSearch
            auto start_CUDA_MultiSoA_search = high_resolution_clock::now();
            vector<vector<int>> risultati_CUDA_MultiSoA = CUDAMultiQuerySearch_SoA(datasetSoA, all_queries);
            auto end_CUDA_MultiSoA_search = high_resolution_clock::now();
            duration<double, milli> time_CUDA_MultiSoA_search = end_CUDA_MultiSoA_search - start_CUDA_MultiSoA_search;


            /// memorizzare dati nelle matrici
            matrixAoS[i][j].single_q = time_SingleAoS_search;
            matrixAoS[i][j].multi_q = time_MultiAoS_search;

            matrixSoA[i][j].single_q = time_SingleSoA_search;
            matrixSoA[i][j].multi_q = time_MultiSoA_search;

            matrixSoA[i][j].single_q_cuda = time_CUDA_SingleSoA_search;
            matrixSoA[i][j].multi_q_cuda = time_CUDA_MultiSoA_search;

        }   
    }



    /*
    // trascrivo i dati in un file csv
    ofstream outFile("C:\\Users\\jbrus\\Documents\\UNI\\MAGISTRALE\\PARALLEL\\PROGETTI\\MIDTERM_PATTERN_RECOGNITION\\src\\Search_results.csv");
    outFile << "Format,QueryLength,NumQueries,Type,TimeMS\n";   

    for (size_t i = 0; i < query_lengths.size(); ++i) {
        for (size_t j = 0; j < query_numbers.size(); ++j) {
            // Scriviamo i dati AoS
            outFile << "AoS," << query_lengths[i] << "," << query_numbers[j] << ",Single," << matrixAoS[i][j].single_q.count() << "\n";
            outFile << "AoS," << query_lengths[i] << "," << query_numbers[j] << ",Multi," << matrixAoS[i][j].multi_q.count() << "\n";

            // Scriviamo i dati SoA
            outFile << "SoA," << query_lengths[i] << "," << query_numbers[j] << ",Single," << matrixSoA[i][j].single_q.count() << "\n";
            outFile << "SoA," << query_lengths[i] << "," << query_numbers[j] << ",Multi," << matrixSoA[i][j].multi_q.count() << "\n";
        }
    }
    outFile.close();
    cout << "Dati salvati in Search_results.csv" << endl;
    */


    fs::path output_dir = fs::path("..") / "src";
    fs::path output_file = output_dir / "Search_results.csv";

    // Crea la cartella 'src' se non esiste già (evita errori di apertura file)
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }

    ofstream outFile(output_file.string());

    if (!outFile.is_open()) {
        cerr << " Errore nell'apertura del file CSV in: " << output_file.string() << endl;
    } else {
        outFile << "Format,QueryLength,NumQueries,Type,TimeMS\n";   

        for (size_t i = 0; i < query_lengths.size(); ++i) {
            for (size_t j = 0; j < query_numbers.size(); ++j) {
                // Scriviamo i dati AoS
                outFile << "AoS," << query_lengths[i] << "," << query_numbers[j] << ",Single," << matrixAoS[i][j].single_q.count() << "\n";
                outFile << "AoS," << query_lengths[i] << "," << query_numbers[j] << ",Multi," << matrixAoS[i][j].multi_q.count() << "\n";

                // Scriviamo i dati SoA
                outFile << "SoA," << query_lengths[i] << "," << query_numbers[j] << ",Single," << matrixSoA[i][j].single_q.count() << "\n";
                outFile << "SoA," << query_lengths[i] << "," << query_numbers[j] << ",Multi," << matrixSoA[i][j].multi_q.count() << "\n";
            }
        }
    outFile.close();
    cout << "✅ Dati salvati con successo in: " << output_file.string() << endl;
    }
    return 0;
}