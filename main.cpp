#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <stdatomic.h>
#include <unordered_map>
#include <random>
#include <atomic>
#include <string>
#include <omp.h>
#include <iostream>
#include <unordered_set>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <utility>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <iostream>
#include <filesystem>
#include <cstdio>

using namespace std;
namespace fs = std::filesystem;
int pre = 0;

int width1;
int height1;
int depth1;
int size2;
int un_sign_as;
int un_sign_ds;
int ite = 0;
std::vector<int> all_d_max, all_d_min;


std::vector<int> record;
std::vector<std::vector<float>> record1;
std::vector<std::vector<float>> record_ratio;

std::vector<std::vector<int>> directions2 = {{0,1,0},{0,-1,0},{1,0,0},{-1,0,0},{-1,1,0},{1,-1,0},{0,0, -1},{0,-1, 1}, {0,0, 1},  {0,1, -1},  {-1,0, 1},   {1, 0,-1}};
std::vector<std::vector<int>> directions3 = {{0,1,0},{0,-1,0},{1,0,0},{-1,0,0},{-1,1,0},{1,-1,0},{0,0, -1},{0,-1, 1}, {0,0, 1},  {0,1, -1},  {-1,0, 1},   {1, 0,-1}};

std::vector<double> getdata2(std::string filename){
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    std::vector<double> data;
    if (!file) {
        std::cerr << "can not open file" << std::endl;
        return data;
    }

    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    
    std::streamsize num_floats = size / sizeof(double);
    
    std::vector<double> buffer(num_floats);

    
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        
        return buffer;
    } else {
        std::cerr << "文件读取失败" << std::endl;
        return buffer;
    }
    
    return buffer;
}
std::string inputfilename;

std::string decompfilename;

std::vector<double> decp_data1;
std::vector<double> input_data1;

std::unordered_map<int, double> maxrecord;
std::unordered_map<int, double> minrecord;
double bound1;




std::vector<int> find_low(){
    std::vector<int> lowGradientIndices(size2, 0);
    
    const double threshold = 1e-16; 
    
    for (int i = 0; i < width1; ++i) {
        
        for (int j = 0; j < height1; ++j) {
            
            for (int k = 0; k < depth1; ++k) {
                
                int rm = i  + j * width1 + k * (height1 * width1);
                
                
                
                for (auto& dir : directions2) {
                    int newX = i + dir[0];
                    int newY = j + dir[1];
                    int newZ = k + dir[2];
                    int r = newX  + newY * width1 + newZ* (height1 * width1);
                    if(r>=0 and r<size2){
                        double gradZ3 = abs(input_data1[r] - input_data1[rm])/2;
                        if (gradZ3<=threshold) {
                            lowGradientIndices[rm]=0;
                            lowGradientIndices[r]=0;
                        }
                    // }
                }
                }
            }
        }
    }
    // cout<<"ok"<<endl;
    return lowGradientIndices;
}

std::vector<int> lowGradientIndices;



int a = 0;


std::map<int, std::tuple<int, int, int>> createReverseMapping(const std::map<std::tuple<int, int ,int>, int>& originalMap) {
    std::map<int, std::tuple<int, int, int>> reverseMap;
    for (const auto& pair : originalMap) {
        reverseMap[pair.second] = pair.first;
    }
    return reverseMap;
}




std::vector<int> wrong_maxi_cp;
std::vector<int> wrong_min_cp;
std::vector<int> wrong_index_as;
std::vector<int> wrong_index_ds;

std::vector<int> or_direction_as;
std::vector<int> or_direction_ds;
std::vector<int> de_direction_as1;
std::vector<int> de_direction_ds1;


std::vector<int> dec_label1;
std::vector<int> or_label1;



double calculateMSE(const std::vector<double>& original, const std::vector<double>& compressed) {
    if (original.size() != compressed.size()) {
        throw std::invalid_argument("The size of the two vectors must be the same.");
    }

    double mse = 0.0;
    for (size_t i = 0; i < original.size(); i++) {
        mse += std::pow(static_cast<double>(original[i]) - compressed[i], 2);
    }
    mse /= original.size();
    return mse;
}

double calculatePSNR(const std::vector<double>& original, const std::vector<double>& compressed, double maxValue) {
    double mse = calculateMSE(original, compressed);
    if (mse == 0) {
        return std::numeric_limits<double>::infinity(); // Perfect match
    }
    double psnr = -20.0*log10(sqrt(mse)/maxValue);
    return psnr;
}



extern void init_inputdata(std::vector<int> *a,std::vector<int> *b,std::vector<int> *c,std::vector<int> *d,std::vector<double> *input_data1,std::vector<double> *decp_data1,std::vector<int>* dec_label1,std::vector<int>* or_label1, int width1, int height1, int depth1, std::vector<int> *low,double bound1,float &datatransfer,float &finddirection, int preserve_min, int preserve_max, int preserve_path, int neighbor_number);

double maxAbsoluteDifference(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        std::cerr << "Vectors are of unequal size." << std::endl;
        return -1; // Or handle the error as per your need
    }

    double maxDiff = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        double diff = std::abs(vec1[i] - vec2[i]);
        if (diff < maxDiff) {
            maxDiff = diff;
        }
    }

    return maxDiff;
}


void print_help(const std::string& program_name) {
    std::cout << "Usage: " << program_name << " <path/to/data>,<width>,<height>,<depth> "
              << "<relative_error_bound> <compressor_type> <connection_type> <preserve_min> <preserve_max> <preserve_integral_lines>\n\n"
              
              << "Arguments:\n"
              << "  <path/to/your/data>,<width>,<height>,<depth>\n"
              << "      Path to the input data file followed by its dimensions. Dimensions should be specified as width, height, and depth separated by commas.\n"
              << "      Example: path/to/your/data.bin,256,256,128\n\n"

              << "  <relative_error_bound>\n"
              << "      Relative error bound as a floating-point number.\n"
              << "      Example: 0.01\n\n"

              << "  <compressor_type>\n"
              << "      Compression library to use. Supported values are:\n"
              << "      - sz3\n"
              << "      - zfp\n"
              << "      Example: sz3\n\n"

              << "  <connection type>\n"
              << "      Connectivity type:\n"
              << "      - 0: Piecewise linear connectivity (e.g., 2D case: connects only up, down, left, right, up-right, and bottom-left).\n"
              << "      - 1: Full connectivity (e.g., 2D case: also connects diagonally).\n"
              << "      Example: 0\n\n"

              << "  <preserve_min>\n"
              << "      Whether to preserve local minima (0 for no, 1 for yes).\n"
              << "      Example: 1\n\n"
              << "  <preserve_max>\n"
              << "      Whether to preserve local maxima (0 for no, 1 for yes).\n"
              << "      Example: 0\n\n"

              << "  <preserve_integral_lines>\n"
              << "      Whether to preserve integral lines (0 for no, 1 for yes).\n"
              << "      DO NOT use this option if:\n"
              << "      - NOT both <preserve_min> and <preserve_max> are set to 1.\n"
              << "      - <connection_type> is set to 1 (full connectivity).\n"
              << "      Example: 1\n\n"

              
              
              << "Options:\n"
              << "  --help, -h           Show this help message and exit.\n\n"
              << "Example usage:\n"
              << "  " << program_name << " path/to/your/data.bin,256,256,128 0.01 sz3 1 0 1 0\n";
}

int main(int argc, char** argv){

    if (argc < 2) {
        std::cerr << "Error: Missing arguments. Use --help for usage information.\n";
        return 1;
    }

    std::string first_arg = argv[1];
    if (first_arg == "--help" || first_arg == "-h") {
        print_help(argv[0]);
        return 0;
    }
    std::string dimension = argv[1];
    
    double range = std::stod(argv[2]);
    std::string compressor_id = argv[3];
    int neighbor_number = std::stoi(argv[4]);
    int preserve_min = std::stoi(argv[5]);
    int preserve_max = std::stoi(argv[6]);
    int preserve_path = std::stoi(argv[7]);
    



    double right_labeled_ratio;
    double target_br;
    float datatransfer = 0.0;
    float mappath_path = 0.0;
    float getfpath = 0.0;
    float fixtime_path = 0.0;
    float finddirection = 0.0;
    float getfcp = 0.0;
    float fixtime_cp = 0.0;
    
    std::istringstream iss(dimension);
    char delimiter;
    std::string file_path;
    

    
    std::string filename;
    if (std::getline(iss, file_path, ',')) {
        
        if (iss >> width1 >> delimiter && delimiter == ',' &&
            iss >> height1 >> delimiter && delimiter == ',' &&
            iss >> depth1) {
            std::cout << "Filename: " << file_path << std::endl;
            std::cout << "Width: " << width1 << std::endl;
            std::cout << "Height: " << height1 << std::endl;
            std::cout << "Depth: " << depth1 << std::endl;
        } else {
            std::cerr << "Parsing error for dimensions" << std::endl;
        }
    } else {
        std::cerr << "Parsing error for file" << std::endl;
    }

    std::filesystem::path path(file_path);
    filename = path.stem().string();
    std::cout << "Extracted file name: " << filename<< std::endl;
    
    inputfilename = filename+".bin";
    
    

    
    
    
    auto start = std::chrono::high_resolution_clock::now();
    input_data1 = getdata2(inputfilename);
    auto min_it = std::min_element(input_data1.begin(), input_data1.end());
    auto max_it = std::max_element(input_data1.begin(), input_data1.end());
    double minValue = *min_it;
    double maxValue = *max_it;
    bound1 = (maxValue-minValue)*range;
    
    
    std::ostringstream stream;
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    
    stream << std::setprecision(std::numeric_limits<double>::max_digits10);
    stream << std::defaultfloat << bound1;  
    std::string valueStr = stream.str();
    size2 = input_data1.size();
    std::string cpfilename = "../compressed_"+filename+"_"+compressor_id+'_'+std::to_string(bound1)+".sz";
    std::string decpfilename = "../decp_"+filename+"_"+compressor_id+'_'+std::to_string(bound1)+".bin";
    std::string fix_path = "../fixed_decp_"+filename+"_"+compressor_id+'_'+std::to_string(bound1)+".bin";
    std::string command;
    cout<<decpfilename<<endl;
    cout<<bound1<<", "<<std::to_string(bound1)<<endl;
    
    int result;
    if(compressor_id=="sz3"){
        
        command = "sz3 -i " + inputfilename + " -z " + cpfilename +" -o "+decpfilename + " -d " + " -1 " + std::to_string(size2)+" -M "+"REL "+std::to_string(range)+" -a";
        std::cout << "Executing command: " << command << std::endl;
        result = std::system(command.c_str());
        if (result == 0) {
            std::cout << "Compression successful." << std::endl;
        } else {
            std::cout << "Compression failed." << std::endl;
        }
    }
    else if(compressor_id=="zfp"){
        cpfilename = "compressed_"+filename+"_"+std::to_string(bound1)+".zfp";
        // zfp -i ~/msz/experiment_data/finger.bin -z compressed.zfp -d -r 0.001
        command = "zfp -i " + inputfilename + " -z " + cpfilename +" -o "+decpfilename + " -d " + " -1 " + std::to_string(size2)+" -a "+std::to_string(bound1)+" -s";
        std::cout << "Executing command: " << command << std::endl;
        result = std::system(command.c_str());
        if (result == 0) {
            std::cout << "Compression successful." << std::endl;
        } else {
            std::cout << "Compression failed." << std::endl;
        }
    }
    
    
    decp_data1 = getdata2(decpfilename);
    cout<<"Preservation started "<<endl;
    auto ends = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration = ends - start;
    double compression_time = duration.count();
    std::vector<double> decp_data_copy(decp_data1);
    
    or_direction_as.resize(size2);
    or_direction_ds.resize(size2);
    de_direction_as1.resize(size2);
    de_direction_ds1.resize(size2);
    or_label1.resize(size2*2, -1);
    dec_label1.resize(size2*2, -1);
    std::vector<std::vector<double>> time_counter;
    
    lowGradientIndices.resize(size2, 0);
    
    

    int cnt=0;

    std::vector<int>* dev_a = &or_direction_as;
    std::vector<int>* dev_b = &or_direction_ds;
    std::vector<int>* dev_c = &de_direction_as1;
    std::vector<int>* dev_d = &de_direction_ds1;
    std::vector<double>* dev_e = &input_data1;
    std::vector<double>* dev_f = &decp_data1;
    std::vector<int>* dev_g = &lowGradientIndices;
    
    
    std::vector<int>* dev_q = &dec_label1;
    std::vector<int>* dev_m = &or_label1;
    
    
    start = std::chrono::high_resolution_clock::now();
    
    double compressed_dataSize = fs::file_size(cpfilename);
    double br = (compressed_dataSize*8)/size2;
    start = std::chrono::high_resolution_clock::now();
    
    init_inputdata(dev_a, dev_b, dev_c, dev_d, dev_e, dev_f, dev_m,dev_q,width1, height1, depth1, dev_g, bound1, datatransfer,finddirection, preserve_min, preserve_max, preserve_path, neighbor_number);
    ends = std::chrono::high_resolution_clock::now();
    duration = ends - start;
    double additional_time = duration.count();
    
    
    std::vector<double> indexs;
    std::vector<double> edits;
    for (int i=0;i<input_data1.size();i++){
        
        if (decp_data_copy[i]!=decp_data1[i]){
            indexs.push_back(i);
            edits.push_back(decp_data1[i]-decp_data_copy[i]);
            cnt++;
        }
    }
    std::vector<int> diffs; 
    std::string indexfilename = "../index_"+filename+".bin";
    std::string editsfilename = "../edits_"+filename+".bin";
    std::string compressedindex = "../index_"+filename+".bin.zst";
    std::string compressededits = "../edits_"+filename+".bin.zst";
    
    if (!indexs.empty()) {
        diffs.push_back(indexs[0]);
    }
    for (size_t i = 1; i < indexs.size(); ++i) {
        diffs.push_back(indexs[i] - indexs[i - 1]);
    }
    double ratio = double(cnt)/(decp_data_copy.size());
    cout<<cnt<<","<<ratio<<endl;
    std::ofstream file(indexfilename, std::ios::binary | std::ios::out);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(diffs.data()), diffs.size() * sizeof(int));
    }
    file.close();

    command = "zstd -f " + indexfilename;
    std::cout << "Executing command: " << command << std::endl;
    result = std::system(command.c_str());
    if (result == 0) {
        
        std::cout << "Compression successful." << std::endl;
    } else {
        std::cout << "Compression failed." << std::endl;
    }
    
    std::ofstream file1(editsfilename, std::ios::binary | std::ios::out);
    if (file1.is_open()) {
        file1.write(reinterpret_cast<const char*>(edits.data()), edits.size() * sizeof(double));
    }
    file1.close();
    command = "zstd -f " + editsfilename;
    std::cout << "Executing command: " << command << std::endl;
    result = std::system(command.c_str());
    if (result == 0) {
        std::cout << "Compression successful." << std::endl;
    } else {
        std::cout << "Compression failed." << std::endl;
    }
    

    double compressed_indexSize = fs::file_size(compressedindex);
    double compressed_editSize = fs::file_size(compressededits);
    double original_indexSize = fs::file_size(indexfilename);
    double original_editSize = fs::file_size(editsfilename);
    double original_dataSize = fs::file_size(inputfilename);
    
    
    double overall_ratio = (original_indexSize+original_editSize+original_dataSize)/(compressed_dataSize+compressed_editSize+compressed_indexSize);
    double bitRate = 64/overall_ratio; 

    double psnr = calculatePSNR(input_data1, decp_data_copy, maxValue-minValue);
    double fixed_psnr = calculatePSNR(input_data1, decp_data1, maxValue-minValue);

    std::ofstream outFile3("../result_"+filename+"_"+compressor_id+"_detailed.txt", std::ios::app);

    
    if (!outFile3) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return 0; 
    }

    
    outFile3 << std::to_string(bound1)<<":" << std::endl;
    outFile3 << std::setprecision(17)<< "related_error: "<<range << std::endl;
    outFile3 << std::setprecision(17)<< "Overall Compression Ratio: "<<overall_ratio << std::endl;
    outFile3 << std::setprecision(17)<<"Original Compression Ratio: "<<original_dataSize/compressed_dataSize << std::endl;
    outFile3 << std::setprecision(17)<<"Overall Bitrate: "<<bitRate << std::endl;
    outFile3 << std::setprecision(17)<<"Original Bitrate: "<< (compressed_dataSize*8)/size2 << std::endl;
    outFile3 << std::setprecision(17)<<"Original PSNR: "<<psnr << std::endl;
    outFile3 << std::setprecision(17)<<"After Preservation PSNR: "<<fixed_psnr << std::endl;
    

    // outFile3 << std::setprecision(17)<<"right_labeled_ratio: "<<right_labeled_ratio << std::endl;
    outFile3 << std::setprecision(17)<<"edit_ratio: "<<ratio << std::endl;
    outFile3 << std::setprecision(17)<<"compression_time: "<<compression_time<< std::endl;
    outFile3 << std::setprecision(17)<<"additional_time: "<<additional_time<< std::endl;
    outFile3 << "\n" << std::endl;
   
    outFile3.close();

    std::cout << "Variables have been appended to output.txt" << std::endl;
    
    
    command = "rm " + indexfilename;
    result = std::system(command.c_str());
    

    command = "rm " + editsfilename;
    result = std::system(command.c_str());
    

    command = "rm " + decpfilename;
    result = std::system(command.c_str());
    
    
    
    
    
    
    return 0;
}
