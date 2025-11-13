#include <vector>
#include <string>
#include <random>

struct TimeSeries{
    std::vector<double> data;
    int cls;
};

struct TimeSeries_SoA{
    std::vector<std::vector<double>> all_data;
    std::vector<int> all_classes;
};

std::vector<std::string> loadRawDataset(const std::string& filename);

std::vector<TimeSeries> loadDatasetAoS(const std::vector<std::string>& rawDataset);
std::vector<TimeSeries> chooseClassAoS( const std::vector<TimeSeries>& input_dataset, int cls_index);
std::vector<int> ParallelSearch_AoS(const std::vector<TimeSeries>& dataset, const std::vector<double>& query);
std::vector<std::vector<int>> MultiQueryParallelSearch_AoS(const std::vector<TimeSeries>& dataset, const std::vector<std::vector<double>>& queries);

TimeSeries_SoA loadDatasetSoA(const std::vector<std::string>& rawDataset);
//void chooseClass_2(int cls_index, std::vector<TimeSeries>& dataset);


int SAD_Search(const std::vector<double>& T, const std::vector<double>& Q);
std::vector<int> ParallelSearch_SoA(const TimeSeries_SoA& dataset, const std::vector<double>& query);
std::vector<std::vector<int>> MultiQueryParallelSearch_SoA(const TimeSeries_SoA& dataset, const std::vector<std::vector<double>>& queries);
std::vector<double> RandomQuery(const TimeSeries_SoA& dataset, size_t lunghezzaQuery, std::mt19937& gen);
