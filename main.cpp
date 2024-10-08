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
#include <parallel/algorithm>  
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
// g++ -std=c++17 -O3 -g hello.cpp -o helloworld
// g++ -std=c++17 -O3 -g -fopenmp -c preserve3d.cpp -o hello2.o
// g++-12 -fopenmp hello2.o kernel.o -lcudart -o helloworld
// g++-12 -fopenmp -std=c++17 -O3 -g hello2.cpp -o helloworld2
// 4.5177893520000003
// g++ -fopenmp hello2.o kernel.o -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64 -lcudart -o helloworld

int width;
int height;
int depth;
int size2;
int un_sign_as;
int un_sign_ds;
int ite = 0;
std::vector<int> all_max, all_min, all_d_max, all_d_min;
atomic_int count_max = 0;
atomic_int count_min = 0;
atomic_int count_f_max = 0;
atomic_int count_f_min = 0;

std::vector<int> record;
std::vector<std::vector<float>> record1;
std::vector<std::vector<float>> record_ratio;

std::vector<std::vector<int>> directions1 = {{0,1,0},{0,-1,0},{1,0,0},{-1,0,0},{-1,1,0},{1,-1,0},{0,0, -1},  {0,-1, 1}, {0,0, 1},  {0,1, -1},  {-1,0, 1},   {1, 0,-1}};


std::vector<double> getdata2(std::string filename){
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    std::vector<double> data;
    if (!file) {
        std::cerr << "无法打开文件" << std::endl;
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

std::vector<double> decp_data;
std::vector<double> input_data;

std::unordered_map<int, double> maxrecord;
std::unordered_map<int, double> minrecord;
double bound;




std::vector<int> find_low(){
    std::vector<int> lowGradientIndices(size2, 0);
    
    const double threshold = 1e-16; 
    
    for (int i = 0; i < width; ++i) {
        
        for (int j = 0; j < height; ++j) {
            
            for (int k = 0; k < depth; ++k) {
                
                int rm = i  + j * width + k * (height * width);
                
                
                
                for (auto& dir : directions1) {
                    int newX = i + dir[0];
                    int newY = j + dir[1];
                    int newZ = k + dir[2];
                    int r = newX  + newY * width + newZ* (height * width);
                    if(r>=0 and r<size2){
                        double gradZ3 = abs(input_data[r] - input_data[rm])/2;
                        if (gradZ3<=threshold) {
                            lowGradientIndices[rm]=1;
                            lowGradientIndices[r]=1;
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

std::vector<std::vector<int>> _compute_adjacency(){
    std::vector<std::vector<int>> adjacency;
    for (int i = 0; i < size2; ++i) {
            int y = (i / (width)) % height; // Get the x coordinate
            int x = i % width; // Get the y coordinate
            int z = (i / (width * height)) % depth;
            std::vector<int> adjacency_temp;
            for (auto& dir : directions1) {
                int newX = x + dir[0];
                int newY = y + dir[1];
                int newZ = z + dir[2];
                int r = newX + newY  * width + newZ* (height * width); // Calculate the index of the adjacent vertex
                
                
                // Check if the new coordinates are within the bounds of the mesh
                if (newX >= 0 && newX < width && newY >= 0 && newY < height && r < width*height*depth && newZ<depth && newZ>=0 && lowGradientIndices[r] != 1) {
                    
                    adjacency_temp.push_back(r);
                }
                // if(input_data[r]-input_data[i]==0 and input_data[r]==0){
                //     continue;
                // }
            }
            adjacency.push_back(adjacency_temp);
        }
    return adjacency;
}
std::vector<std::vector<int>> adjacency;



std::map<std::tuple<int, int, int>, int> createDirectionMapping() {
    std::map<std::tuple<int, int, int>, int> direction_mapping_3d;
    direction_mapping_3d[std::make_tuple(0,1,0)] = 1;
    direction_mapping_3d[std::make_tuple(0,-1,0)] = 2;
    direction_mapping_3d[std::make_tuple(1,0,0)] = 3;
    direction_mapping_3d[std::make_tuple(-1,0,0)] = 4;
    direction_mapping_3d[std::make_tuple(-1,1,0)] = 5;
    direction_mapping_3d[std::make_tuple(1,-1,0)] = 6;

    // Additional 3D directions
    direction_mapping_3d[std::make_tuple(0, 0, -1)] = 7;   // down in Z
    direction_mapping_3d[std::make_tuple(0,-1, 1)] = 8;   // down-left in Z
    direction_mapping_3d[std::make_tuple(0, 0, 1)] = 9;    // up in Z
    direction_mapping_3d[std::make_tuple(0, 1, -1)] = 10;  // up-right in Z
    direction_mapping_3d[std::make_tuple(-1, 0, 1)] = 11;  // left-up in Z
    direction_mapping_3d[std::make_tuple(1, 0, -1)] = 12; 

    return direction_mapping_3d;
};

int a = 0;


std::map<int, std::tuple<int, int, int>> createReverseMapping(const std::map<std::tuple<int, int ,int>, int>& originalMap) {
    std::map<int, std::tuple<int, int, int>> reverseMap;
    for (const auto& pair : originalMap) {
        reverseMap[pair.second] = pair.first;
    }
    return reverseMap;
}
std::map<std::tuple<int, int ,int>, int> direction_mapping = createDirectionMapping();

std::map<int, std::tuple<int, int ,int>> reverse_direction_mapping = createReverseMapping(direction_mapping);





std::vector<int> wrong_maxi_cp;
std::vector<int> wrong_min_cp;
std::vector<int> wrong_index_as;
std::vector<int> wrong_index_ds;
int direction_to_index_mapping[12][3] = {{0,1,0},{0,-1,0},{1,0,0},{-1,0,0},{-1,1,0},{1,-1,0},{0,0, -1},  {0,-1, 1}, {0,0, 1},  {0,1, -1},  {-1,0, 1},   {1, 0,-1}};

int getDirection(const std::map<std::tuple<int, int, int>, int>& direction_mapping, int row_diff, int col_diff, int dep_diff) {
    auto it = direction_mapping.find(std::make_tuple(row_diff, col_diff,dep_diff));
    
    if (it != direction_mapping.end()) {
        return it->second;
    } else {
        return -1; 
    }
}


int from_direction_to_index(int cur, int direc){
    
    if (direc==-1) return cur;
    int x = cur % width;
    int y = (cur / width) % height;
    int z = (cur/(width * height))%depth;
    // printf("%d %d\n", row, rank1);
    if (direc >= 1 && direc <= 12) {
        int delta_row = direction_to_index_mapping[direc-1][0];
        int delta_col = direction_to_index_mapping[direc-1][1];
        int delta_dep = direction_to_index_mapping[direc-1][2];
        
        
        int next_row = x + delta_row;
        int next_col = y + delta_col;
        int next_dep = z + delta_dep;
        // printf("%d \n", next_row * width + next_col);
        // return next_row * width + next_col + next_dep* (height * width);
        return next_row + next_col * width + next_dep* (height * width);
    }
    else {
        return -1;
    }
    // return 0;
};

std::vector<int> or_direction_as;
std::vector<int> or_direction_ds;
std::vector<int> de_direction_as;
std::vector<int> de_direction_ds;


std::vector<int> dec_label;
std::vector<int> or_label;



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



extern void init_inputdata(std::vector<int> *a,std::vector<int> *b,std::vector<int> *c,std::vector<int> *d,std::vector<double> *input_data1,std::vector<double> *decp_data1,std::vector<int> *dec_label1,std::vector<int> *or_label1,int width1, int height1, int depth1, std::vector<int> *low,double bound1,float &datatransfer,float &finddirection,double &right);

extern void fix_process(std::vector<int> *c,std::vector<int> *d, std::vector<double> *decp_data1, float &datatransfer, float &finddirection, float &getfcp, float &fixtime_cp,int &cpite);
extern void mappath1(std::vector<int> *label, std::vector<int> *direction_as, std::vector<int> *direction_ds, float &finddirection, float &mappath_path, float &datatransfer,int type=0);

void getlabel(int i){
    
    
    
    int cur = dec_label[i*2+1];
    int next_vertex;
    // cout<<cur<<endl;
    if (cur==-1){
        
        return;
    }
    else if (de_direction_as[cur]!=-1){
        
        // cout<<cur<<" "<<de_direction_as[cur]<<endl;
        int direc = de_direction_as[cur];
        int row = cur/width;
        int rank1 = cur%width;
        
        switch (direc) {
            case 1:
                next_vertex = (row)*width + (rank1-1);
                break;
            case 2:
                next_vertex = (row-1)*width + (rank1);
                break;
            case 3:
                next_vertex = (row-1)*width + (rank1+1);
                break;
            case 4:
                next_vertex = (row)*width + (rank1+1);
                break;
            case 5:
                next_vertex = (row+1)*width + (rank1);
                break;
            case 6:
                next_vertex = (row+1)*width + (rank1-1);
                break;
        };

        cur = next_vertex;
        
        if (de_direction_as[cur] != -1){
            
            un_sign_as+=1;
        }

        if(de_direction_as[i]!=-1){
            dec_label[i*2+1] = cur;
            
        }
        else{
            dec_label[i*2+1] = -1;
        };
        
    }

    
    
    cur = dec_label[i*2];
    int next_vertex1;
    if(cur==-1){
        return;
    }
    if (de_direction_as[cur]!=-1){
        // printf("%d\n", cur);
        int direc = de_direction_ds[cur];
        int row = cur/width;
        int rank1 = cur%width;
        
        switch (direc) {
            case 1:
                next_vertex1 = (row)*width + (rank1-1);
                break;
            case 2:
                next_vertex1 = (row-1)*width + (rank1);
                break;
            case 3:
                next_vertex1 = (row-1)*width + (rank1+1);
                break;
            case 4:
                next_vertex1 = (row)*width + (rank1+1);
                break;
            case 5:
                next_vertex1 = (row+1)*width + (rank1);
                break;
            case 6:
                next_vertex1 = (row+1)*width + (rank1-1);
                break;
        };

        cur = next_vertex1;
        
        if (de_direction_ds[cur] != -1){
            un_sign_ds+=1;
            // printf("%d \n",i);
        }

        if(de_direction_ds[i]!=-1){
            dec_label[i*2] = cur;
            
        }
        else{
            dec_label[i*2] = -1;
        };
        
    }

    

}
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
int main(int argc, char** argv){
    omp_set_num_threads(44);
    std::string dimension = argv[1];
    double range = std::stod(argv[2]);
    std::string compressor_id = argv[3];
    int mode = std::stoi(argv[4]);
    double right_labeled_ratio;
    double target_br;
    float datatransfer = 0.0;
    float mappath_path = 0.0;
    float getfpath = 0.0;
    float fixtime_path = 0.0;
    float finddirection = 0.0;
    float getfcp = 0.0;
    float fixtime_cp = 0.0;
    if(mode==1){
        target_br = std::stod(argv[5]);
    }
    std::istringstream iss(dimension);
    char delimiter;
    std::string filename;
    if (std::getline(iss, filename, ',')) {
        // 接下来读取整数值
        if (iss >> width >> delimiter && delimiter == ',' &&
            iss >> height >> delimiter && delimiter == ',' &&
            iss >> depth) {
            std::cout << "Filename: " << filename << std::endl;
            std::cout << "Width: " << width << std::endl;
            std::cout << "Height: " << height << std::endl;
            std::cout << "Depth: " << depth << std::endl;
        } else {
            std::cerr << "Parsing error for dimensions" << std::endl;
        }
    } else {
        std::cerr << "Parsing error for filename" << std::endl;
    }

    
    inputfilename = filename+".bin";
    
    

    
    
    
    auto start = std::chrono::high_resolution_clock::now();
    input_data = getdata2(inputfilename);
    auto min_it = std::min_element(input_data.begin(), input_data.end());
    auto max_it = std::max_element(input_data.begin(), input_data.end());
    double minValue = *min_it;
    double maxValue = *max_it;
    bound = (maxValue-minValue)*range;
    
    
    std::ostringstream stream;
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    
    stream << std::setprecision(std::numeric_limits<double>::max_digits10);
    stream << std::defaultfloat << bound;  
    std::string valueStr = stream.str();
    size2 = input_data.size();
    std::string cpfilename = "compressed_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".sz";
    std::string decpfilename = "decp_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".bin";
    std::string fix_path = "fixed_decp_"+filename+"_"+compressor_id+'_'+std::to_string(bound)+".bin";
    std::string command;
    cout<<decpfilename<<endl;
    cout<<bound<<", "<<std::to_string(bound)<<endl;
    // exit(0);
    int result;
    
    if(compressor_id=="sz3"){
        
        command = "sz3 -i " + inputfilename + " -z " + cpfilename +" -o "+decpfilename + " -d " + " -1 " + std::to_string(size2)+" -M "+"ABS "+std::to_string(bound)+" -a";
        std::cout << "Executing command: " << command << std::endl;
        result = std::system(command.c_str());
        if (result == 0) {
            std::cout << "Compression successful." << std::endl;
        } else {
            std::cout << "Compression failed." << std::endl;
        }
    }
    else if(compressor_id=="zfp"){
        cpfilename = "compressed_data/compressed_"+filename+"_"+std::to_string(bound)+".zfp";
        // zfp -i ~/msz/experiment_data/finger.bin -z compressed.zfp -d -r 0.001
        command = "zfp -i " + inputfilename + " -z " + cpfilename +" -o "+decpfilename + " -d " + " -1 " + std::to_string(size2)+" -a "+std::to_string(bound)+" -s";
        std::cout << "Executing command: " << command << std::endl;
        result = std::system(command.c_str());
        if (result == 0) {
            std::cout << "Compression successful." << std::endl;
        } else {
            std::cout << "Compression failed." << std::endl;
        }
    }
    
    
    decp_data = getdata2(decpfilename);
    auto ends = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration = ends - start;
    double compression_time = duration.count();
    std::vector<double> decp_data_copy(decp_data);
    
    or_direction_as.resize(size2);
    or_direction_ds.resize(size2);
    de_direction_as.resize(size2);
    de_direction_ds.resize(size2);
    or_label.resize(size2*2, -1);
    dec_label.resize(size2*2, -1);
    std::vector<std::vector<double>> time_counter;
    lowGradientIndices = find_low();
    lowGradientIndices.resize(size2, 0);
    
    adjacency = _compute_adjacency();
    
    all_max.resize(size2);
    all_min.resize(size2);
    
    std::random_device rd;  /
    std::mt19937 gen(rd()); 
    
    
    

    int cnt=0;

    std::vector<int>* dev_a = &or_direction_as;
    std::vector<int>* dev_b = &or_direction_ds;
    std::vector<int>* dev_c = &de_direction_as;
    std::vector<int>* dev_d = &de_direction_ds;
    std::vector<double>* dev_e = &input_data;
    std::vector<double>* dev_f = &decp_data;
    std::vector<int>* dev_g = &lowGradientIndices;
    
    
    std::vector<int>* dev_q = &dec_label;
    std::vector<int>* dev_m = &or_label;
    
    
    start = std::chrono::high_resolution_clock::now();
    
    double compressed_dataSize = fs::file_size(cpfilename);
    double br = (compressed_dataSize*8)/size2;
    start = std::chrono::high_resolution_clock::now();
    init_inputdata(dev_a, dev_b, dev_c, dev_d, dev_e, dev_f, dev_m,dev_q,width, height, depth, dev_g, bound,datatransfer,finddirection, right_labeled_ratio);
    ends = std::chrono::high_resolution_clock::now();
    duration = ends - start;
    double additional_time = duration.count();
    
    
    std::vector<double> indexs;
    std::vector<double> edits;
    for (int i=0;i<input_data.size();i++){
        
        if (decp_data_copy[i]!=decp_data[i]){
            indexs.push_back(i);
            edits.push_back(decp_data[i]-decp_data_copy[i]);
            cnt++;
        }
    }
    std::vector<int> diffs;  // 存储差值的 vector
    std::string indexfilename = "data"+filename+".bin";
    std::string editsfilename = "data_edits"+filename+".bin";
    std::string compressedindex = "data"+filename+".bin.zst";
    std::string compressededits = "data_edits"+filename+".bin.zst";
    
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
    
    
    

    
    
    

    
    
    
    return 0;
}
