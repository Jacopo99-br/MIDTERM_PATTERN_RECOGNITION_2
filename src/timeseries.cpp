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
    size_t series_size = rawDataset.size();

    size_t pos_first = rawDataset[0].find_last_of(delimiter); //lunghezza serie della prima riga per preallocare
    vector<double> first_serie = DataParse(rawDataset[0].substr(0, pos_first));
    dataset.serie_lenght = first_serie.size();

    //si può fare la pre-allocazione totale
    dataset.all_data_flat.resize(series_size * dataset.serie_lenght); 
    dataset.all_classes.resize(series_size);
    
    #pragma omp parallel for schedule(static) // Ogni thread scrive direttamente nella posizione corretta del vettore flat e classes, evitando così la necessità di un vettore di dataset parziali e della sezione critica
    for  (int i = 0; i < series_size; ++i ){
        string _line = rawDataset[i];
        size_t pos_delim = _line.find_last_of(delimiter); //size_t per evitare warning in caso npos
        
        // Estrazione classe
        dataset.all_classes[i] = stoi(_line.substr(pos_delim + 1)); //string to integer

        //Parsing dati 
        vector<double> serie = DataParse(_line.substr(0, pos_delim));

        // Copia veloce nel vettore flat
        size_t offset = i * dataset.serie_lenght;
        std::copy(serie.begin(), serie.end(), dataset.all_data_flat.begin() + offset); //sposta blocchi di dati. Begin punto di partenza, end punto fine sorgente + offset è il punto di partenza nel vettore flat
        // diventa per alcuni compilatori memcpy , sfrutta la larghezza di banda massima della memoria RAM
    }

    cout<<"loaded "<<dataset.all_data_flat.size()/dataset.serie_lenght<<" time series (SoA) from file  "<<endl;
    return dataset;
}

/// query functions:


 
int SAD_Search(const double* T,int N, const vector<double>& Q){ // T --> è il puntatore alla serie di 5120 misurazioi sulla quale confrontare la query Q
    double min_sad = std::numeric_limits<double>::max();
    int best_index = -1;
    int M = Q.size(); // lunghezza query
    const double* Q_ptr = Q.data(); // ottieni il puntatore alla query

    for (int i=0; i<= N-M; ++i){
        double current_sad = 0.0;

        #pragma omp simd reduction(+:current_sad) // SIMD per vettorizzare il calcolo del SAD, reduction per sommare in parallelo
        for (int j=0; j<M; ++j){
            current_sad += std::abs(T[i+j] - Q_ptr[j]);
        }
        if (current_sad < min_sad) {
            min_sad = current_sad;
            best_index = i;
        }

    }

    return best_index;
}

// stessa query
vector<int> ParallelSearch_AoS(const vector<TimeSeries>& dataset, const vector<double>& query) {
    
    vector<int> results(dataset.size()); // Preallocazione

    #pragma omp parallel for 
    for (int i = 0; i < dataset.size(); ++i) {
        results[i] = SAD_Search(dataset[i].data.data(), dataset[i].data.size(), query); 
    }
    return results;
}

vector<int> ParallelSearch_SoA(const TimeSeries_SoA& dataset, const vector<double>& query){
    int numDataRows = dataset.all_classes.size();
    int L = dataset.serie_lenght;
    vector<int> results(numDataRows);    //preallocazione per non dover usare critical

    #pragma omp parallel for 
    for (int i=0; i < numDataRows; ++i){
        const double* series_ptr = &dataset.all_data_flat[i * L]; // Calcola l'offset per accedere alla serie i-esima
        results[i] = SAD_Search(series_ptr, L, query);
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
            all_results[q_idx][i] = SAD_Search(dataset[i].data.data(), dataset[i].data.size(), queries[q_idx]);
        }
    } 
    return all_results;
}

vector<vector<int>> MultiQueryParallelSearch_SoA(const TimeSeries_SoA& dataset, const vector<vector<double>>& queries){
    int numQueries = queries.size();
    int numDataRows = dataset.all_classes.size();
    int L = dataset.serie_lenght;
    vector<vector<int>> all_results(numQueries, vector<int>(numDataRows));

    #pragma omp parallel for collapse(2)  // Parallelizza su entrambe le dimensioni --> ( num_query x num_righe )/num_thread 
    for (int q_idx = 0; q_idx < numQueries; ++q_idx) {
        for (int i = 0; i < numDataRows; ++i) {
            const double* series_ptr = &dataset.all_data_flat[i * L]; // Calcola l'offset per accedere alla serie i-esima
            all_results[q_idx][i] = SAD_Search(series_ptr, L, queries[q_idx]);
        }
    } 
    
    return all_results;
}

std::vector<double> RandomQuery(const TimeSeries_SoA& dataset, size_t lunghezzaQuery, std::mt19937& gen){
    size_t numSeries = dataset.all_classes.size();
    
    if (numSeries == 0) {
        std::cerr << "ERROR: Dataset is empty." << std::endl;
        return {};
    }
    if (lunghezzaQuery == 0 || lunghezzaQuery > dataset.serie_lenght) {
        std::cerr << "ERROR: Query length invalid." << std::endl;
        return {};
    }
    std::uniform_int_distribution<size_t> row_dist(0, numSeries - 1);
    size_t row = row_dist(gen);
    
    // 3. Calcoliamo l'indice di inizio massimo all'interno della riga scelta
    size_t max_start = dataset.serie_lenght - lunghezzaQuery;
    std::uniform_int_distribution<size_t> start_dist(0, max_start);
    size_t start_in_row = start_dist(gen);

    // 4. Calcoliamo l'offset globale nel vettore FLAT
    // L'inizio è: (indice_riga * lunghezza_riga) + punto_di_inizio_nella_riga
    size_t global_start_index = (row * dataset.serie_lenght) + start_in_row;

    // 5. Estraiamo la sottoserie usando gli iteratori del vettore piatto
    return std::vector<double>(
        dataset.all_data_flat.begin() + global_start_index, 
        dataset.all_data_flat.begin() + global_start_index + lunghezzaQuery
    );
}