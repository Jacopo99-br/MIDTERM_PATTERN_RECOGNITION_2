#include <iostream>
#include <vector>
#include "timeseries.h" 
#include <omp.h>
#include <chrono>
#include <random>

using namespace std;
using namespace std::chrono;
int main() {
    
    string path = "C:\\Users\\jbrus\\Documents\\UNI\\MAGISTRALE\\PARALLEL\\PROGETTI\\MIDTERM_PATTERN_RECOGNITION\\FaultDetectionA\\FaultDetectionA_TEST.ts";
    std::random_device rd;
    std::mt19937 gen(rd());
    const int NUM_QUERIES = 5;
    const int QUERY_LENGTH = 100;

    vector<string> raw_data_lines = loadRawDataset(path);

    vector<TimeSeries> datasetAoS = loadDatasetAoS(raw_data_lines);
    TimeSeries_SoA datasetSoA = loadDatasetSoA(raw_data_lines);
    cout << "Dataset loaded in both AoS and SoA formats." << endl;

    vector<vector<double>> all_queries;
    for(int i = 0; i < NUM_QUERIES; ++i) {
        all_queries.push_back(RandomQuery(datasetSoA, QUERY_LENGTH, gen));
    }
    cout << "Create " << all_queries.size() << " query." << endl;

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


    /// Print times
    cout << "Single Search AoS Time: " << time_SingleAoS_search.count() << " ms" << endl;
    cout << "Single Search SoA Time: " << time_SingleSoA_search.count() << " ms" << endl;

    cout << "Multi Search AoS Time: " << time_MultiAoS_search.count() << " ms" << endl;
    cout << "Multi Search SoA Time: " << time_MultiSoA_search.count() << " ms" << endl;   
    return 0;
}