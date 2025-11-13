#include "timeseries.h"
#include <fstream>
#include <sstream>


#include <omp.h>
#include <iostream>
#include <algorithm>
#include <limits>

#include <random> 
using namespace std;

static vector<double> DataParse(const string& s){
    vector<double> value;
    stringstream ss(s);
    string value_str;
    while(getline(ss, value_str, ',')){
        try {
            value.push_back(std::stod(value_str));
        } catch (...) { }
    }
    return value;
}

std::vector<TimeSeries> chooseClassAoS(const std::vector<TimeSeries>& input_dataset, int cls_index) 
{
    std::vector<TimeSeries> filtered_dataset;

    #pragma omp parallel for
    for (int i = 0; i < input_dataset.size(); ++i) {
        
        if (input_dataset[i].cls == cls_index) {

            #pragma omp critical
            {
                filtered_dataset.push_back(input_dataset[i]);
            }
        }
    }

    std::cout << "Filtered " << filtered_dataset.size() 
              << " raws for class " << cls_index << std::endl;
              
    return filtered_dataset;
}


vector<string> loadRawDataset(const string& filename){
    
    ifstream file(filename);
    string line;
    vector<string> rawDataset;

    bool starting = false; // controllo per non includere metadati e righe vuote

    if (!file.is_open()){
        throw runtime_error("Could not open file: " + filename);
    }

    // loading each line of the dataset (serially)

    while(getline(file, line)){
        if(!starting){
            if(line.find("@data") != string::npos){
                starting = true;
            }
            continue;
        }
        if(line.empty()){
            continue;
        }
        rawDataset.push_back(line);
    }
    file.close();
    std::cout<<"loaded "<<rawDataset.size()<<" raw time series from file  "<<endl;

    return rawDataset;
}

vector<TimeSeries> loadDatasetAoS(const vector<string>& rawDataset){
    
    // parsing each line of the dataset (in parallel)
    vector<TimeSeries> dataset(rawDataset.size());

    string delimiter = ":";

    #pragma omp parallel for 
    for (int i=0 ; i< rawDataset.size();  ++i){
        string _line = rawDataset[i];

        //string token = _line.substr(_line.find(delimiter), _line.length());
        size_t pos_delim = _line.find_last_of(delimiter); //size_t per evitare warning in caso npos
        string token = _line.substr(pos_delim + 1);
        
        string data_str = _line.substr(0, pos_delim);
        dataset[i].data = DataParse(data_str);
        dataset[i].cls = stoi(token);
        //AoS in questo caso
    }
    cout<<"loaded "<<dataset.size()<<" time series (AoS) from file  "<<endl;
    return dataset;
}

TimeSeries_SoA loadDatasetSoA(const vector<string>& rawDataset){
    
    vector<TimeSeries_SoA> partial_datasets; // per ogni thread
    TimeSeries_SoA dataset ;

    string delimiter = ":";

    #pragma omp parallel
    {
        TimeSeries_SoA local_dataset;

        #pragma omp for nowait
        for (int i=0 ; i< rawDataset.size();  ++i){
            string _line = rawDataset[i];

            size_t pos_delim = _line.find_last_of(delimiter); //size_t per evitare warning in caso npos
            string token = _line.substr(pos_delim + 1);
            string data_str = _line.substr(0, pos_delim);
            

            local_dataset.all_data.push_back(DataParse(data_str));
            local_dataset.all_classes.push_back(stoi(token));
            
        }
        #pragma omp critical
        {
            partial_datasets.push_back(local_dataset);
        }
    }

    for (const auto& pd : partial_datasets) {
        dataset.all_data.insert(dataset.all_data.end(), pd.all_data.begin(), pd.all_data.end()); // appendi tutto il pd alla fine del dataset
        dataset.all_classes.insert(dataset.all_classes.end(), pd.all_classes.begin(), pd.all_classes.end());
    }

    cout<<"loaded "<<dataset.all_data.size()<<" time series (SoA) from file  "<<endl;
    return dataset;
}

/// query functions:


 
int SAD_Search(const vector<double>& T, const vector<double>& Q){ // T --> è la serie di 5120 misurazioi sulla quale confrontare la query Q
    if (Q.size()>T.size() || T.empty() || Q.empty()){
        throw runtime_error("Error in SAD_Search: query size greater than time series size or empty series");
    }

    double min_sad = std::numeric_limits<double>::max(); // val max di un double così da essere sicuri che la prima somma calcolata sia minore
    int best_index = -1;
    int N = T.size();
    int M = Q.size();
    for (int i = 0; i <= N - M; ++i) {
        double current_sad = 0.0;

        #pragma omp simd reduction(+:current_sad)
        for (int j = 0; j < M; ++j) {
            current_sad += std::abs(T[i + j] - Q[j]);
        }
        if (current_sad < min_sad) {
            min_sad = current_sad;
            best_index = i;
        }
    }
    return best_index;
}



// stessa query
vector<int> ParallelSearch_AoS(const vector<TimeSeries>& dataset, 
                                 const vector<double>& query) {
    vector<int> results(dataset.size()); // Preallocazione

    #pragma omp parallel for 
    for (int i = 0; i < dataset.size(); ++i) {
        results[i] = SAD_Search(dataset[i].data, query); 
    }
    return results;
}

vector<int> ParallelSearch_SoA(const TimeSeries_SoA& dataset, const vector<double>& query){
    vector<int> results(dataset.all_data.size());    //preallocazione per non dover usare critical

    #pragma omp parallel for 
    for (int i=0; i < dataset.all_data.size(); ++i){
        results[i] = SAD_Search(dataset.all_data[i], query);
    }
    return results;
}

// multiple queries
vector<vector<int>> MultiQueryParallelSearch_AoS(const vector<TimeSeries>& dataset, const vector<vector<double>>& queries) 
{
    int numQueries = queries.size();
    int numDataRows = dataset.size();

    vector<vector<int>> all_results(numQueries, vector<int>(numDataRows));

    #pragma omp parallel for collapse(2)
    for (int q_idx = 0; q_idx < numQueries; ++q_idx) {
        for (int i = 0; i < numDataRows; ++i) {
            // La differenza è qui:
            all_results[q_idx][i] = SAD_Search(dataset[i].data, queries[q_idx]);
        }
    } 
    return all_results;
}

vector<vector<int>> MultiQueryParallelSearch_SoA(const TimeSeries_SoA& dataset, const vector<vector<double>>& queries){
    int numQueries = queries.size();
    int numDataRows = dataset.all_data.size();

    vector<vector<int>> all_results(numQueries, vector<int>(numDataRows));

    #pragma omp parallel for collapse(2)  // Parallelizza su entrambe le dimensioni --> ( num_query x num_righe )/num_thread 
    for (int q_idx = 0; q_idx < numQueries; ++q_idx) {
        for (int i = 0; i < numDataRows; ++i) {
            all_results[q_idx][i] = SAD_Search(dataset.all_data[i], queries[q_idx]);
        }
    } 
    
    return all_results;
}

std::vector<double> RandomQuery(const TimeSeries_SoA& dataset, size_t lunghezzaQuery, std::mt19937& gen){
    if (dataset.all_data.empty()) {
        std::cerr << "ERROR: Dataset is empty." << std::endl;
        return {};
    }
    if (lunghezzaQuery == 0) {
        std::cerr << "ERROR: Query length cannot be zero." << std::endl;
        return {};
    }
    std::uniform_int_distribution<size_t> row_dist(0, dataset.all_data.size() - 1);
    size_t row = row_dist(gen);

    const auto& serie = dataset.all_data[row];
    if (lunghezzaQuery > serie.size()) {
        std::cerr << "ERROR: Query length > series size (" << serie.size() << ")" << std::endl;
        return {};
    }

    // scegli un indice di inizio casuale affinché la sottoserie rientri nella serie
    size_t max_start = serie.size() - lunghezzaQuery;
    std::uniform_int_distribution<size_t> start_dist(0, max_start);
    size_t start = start_dist(gen);

    return std::vector<double>(serie.begin() + start, serie.begin() + start + lunghezzaQuery);
}