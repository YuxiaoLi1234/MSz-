#include <iostream>
#include <float.h> 
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <stdio.h>
#include <parallel/algorithm>  
#include <unordered_map>
#include <random>
#include <iostream>
#include <cstring> 
#include <chrono> 
#include <cuda_runtime.h>
#include <string>
#include <omp.h>
#include <unordered_set>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <utility>
#include <iomanip>
#include <chrono>
#include <thrust/device_vector.h>
using std::count;
using std::cout;
using std::endl;


__device__ double* decp_data;
__device__ double* decp_data_copy ;
__device__ int directions1[36] =  {0,1,0,0,-1,0,1,0,0,-1,0,0,-1,1,0,1,-1,0,0,0, -1,  0,-1, 1, 0,0, 1,  0,1, -1,  -1,0, 1,   1, 0,-1};
__device__ int width;
__device__ int height;
__device__ int depth;
__device__ int num;
__device__ int* adjacency;
__device__ double* d_deltaBuffer1;
__device__ int* number_array;
__device__ int* all_max; 
__device__ int* all_min;
__device__ int* all_p_max; 
__device__ int* all_p_min;
__device__ int* unsigned_n;
__device__ int count_max;
__device__ int count_min;
__device__ int count_f_max;
__device__ int count_f_min;
__device__ int count_p_max;
__device__ int count_p_min;
__device__ int* maxi;

__device__ int* mini;
__device__ double bound;
__device__ int edit_count;
__device__ int* or_maxi;
__device__ int* or_mini;
__device__ double* d_deltaBuffer;
__device__ int* id_array;
__device__ int* or_label;
__device__ int* dec_label;
__device__ int* lowgradientindices;
__device__ double* input_data;
__device__ int* de_direction_as;
__device__ int* de_direction_ds;
__device__ int maxNeighbors = 12;

__device__ int direction_to_index_mapping[12][3] = {{0,1,0},{0,-1,0},{1,0,0},{-1,0,0},{-1,1,0},{1,-1,0},{0,0, -1},  {0,-1, 1}, {0,0, 1},  {0,1, -1},  {-1,0, 1},   {1, 0,-1}};   



template<typename T>
class LockFreeStack {
public:

    __device__ void push(const T& value) {
        Node* new_node = (Node*)malloc(sizeof(Node));
        new_node->value = value;
        Node* old_head = head;
        do {
            new_node->next = old_head;
        } while (atomicCAS(reinterpret_cast<unsigned long long*>(&head),
                           reinterpret_cast<unsigned long long>(old_head),
                           reinterpret_cast<unsigned long long>(new_node)) !=
                 reinterpret_cast<unsigned long long>(old_head));
        
    }

    __device__ bool pop(T& value) {
        Node* old_head = head;
        if (old_head == nullptr) {
            return false;
        }
        Node* new_head;
        do {
            new_head = old_head->next;
        } while (atomicCAS(reinterpret_cast<unsigned long long*>(&head),
                           reinterpret_cast<unsigned long long>(old_head),
                           reinterpret_cast<unsigned long long>(new_head)) !=
                 reinterpret_cast<unsigned long long>(old_head));
        value = old_head->value;
        free(old_head);
        return true;
    }
    __device__ void clear() {
        Node* current = head;
        while (current != nullptr) {
            Node* next = current->next;
            delete current;
            current = next;
        }
        head = nullptr;
    }
    __device__ int size() const {
        int count = 0;
        Node* current = head;
        while (current != nullptr) {
            count++;
            
            current = current->next;
        }
        return count;
    }

    __device__ bool isEmpty() const {
        return head == nullptr;
    }

private:
    struct Node {
        T value;
        Node* next;
    };

    Node* head = nullptr;
};

__device__ LockFreeStack<double> d_stacks;
__device__ LockFreeStack<int> id_stacks;
__device__ int getDirection(int x, int y, int z){
    
    for (int i = 0; i < 12; ++i) {
        if (direction_to_index_mapping[i][0] == x && direction_to_index_mapping[i][1] == y && direction_to_index_mapping[i][2] == z) {
            return i+1;  
        }
    }
    return -1;  


}


__device__ int from_direction_to_index1(int cur, int direc){
    
    if (direc==-1) return cur;
    int x = cur % width;
    int y = (cur / width) % height;
    int z = (cur/(width * height))%depth;
    
    if (direc >= 1 && direc <= 12) {
        int delta_row = direction_to_index_mapping[direc-1][0];
        int delta_col = direction_to_index_mapping[direc-1][1];
        int delta_dep = direction_to_index_mapping[direc-1][2];
        
        
        int next_row = x + delta_row;
        int next_col = y + delta_col;
        int next_dep = z + delta_dep;
        
        return next_row + next_col * width + next_dep* (height * width);
    }
    else {
        return -1;
    }
    // return 0;
};



__global__ void copy_array_to_stack(int* index_array, double* edit_array, int size) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < size; i++) {
            // id_stacks.push(index_array[i]);
            // d_stacks.push(edit_array[i]);
        }
    }
}

__device__ void find_direction2 (int type, int index){
    double *data;
    int *direction_as;
    int *direction_ds;
    if(type==0){
        data = decp_data;
        direction_as = de_direction_as;
        direction_ds = de_direction_ds;
    }
    else{
        data = input_data;
        direction_as = or_maxi;
        direction_ds = or_mini;
    }
    
    double mini = 0;
    
    
    // std::vector<int> indexs = adjacency[index];
    int largetst_index = index;
    
    
        
    for(int j =0;j<12;++j){
        int i = adjacency[index*12+j];
        
        if(i==-1){
            break;
        }
        if(lowgradientindices[i]==1){
            continue;
        }
        if((data[i]>data[largetst_index] or (data[i]==data[largetst_index] and i>largetst_index))){
            mini = data[i]-data[index];
            
            largetst_index = i;
            // }
            
        };
    };
    int row_l = (largetst_index / (height)) % width;
    int row_i = (index / (height)) % width;
    
    int col_diff = row_l - row_i;
    int row_diff = (largetst_index % height) - (index % height);

    int dep_diff = (largetst_index /(width * height))%depth - (index /(width * height))%depth;
    direction_as[index] = getDirection(row_diff, col_diff,dep_diff);
    
    

    mini = 0;
    largetst_index = index;
    for(int j =0;j<12;++j){
        int i = adjacency[index*12+j];
        
        if(i==-1){
            break;
        }
        if(lowgradientindices[i]==1){
            continue;
        }
        
        if((data[i]<data[largetst_index] or (data[i]==data[largetst_index] and i<largetst_index))){
            mini = data[i]-data[index];
            
            largetst_index = i;

            
        };
    };
    
    row_l = (largetst_index / (height)) % width;
    row_i = (index / (height)) % width;
    
    col_diff = row_l - row_i;
    row_diff = (largetst_index % height) - (index % height);

    dep_diff = (largetst_index /(width * height))%depth - (index /(width * height))%depth;
    
    
    direction_ds[index] = getDirection(row_diff, col_diff,dep_diff);
    
    
    
}


__global__ void find_direction (int type=0){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index>=num or lowgradientindices[index]==1){
        return;
    }
    
    double *data;
    int *direction_as;
    int *direction_ds;
    if(type==0){
        data = decp_data;
        direction_as = de_direction_as;
        direction_ds = de_direction_ds;
    }
    else{
        data = input_data;
        direction_as = or_maxi;
        direction_ds = or_mini;
    }
    
    double mini = 0;
        
        
    int largetst_index = index;

    
        
    for(int j =0;j<12;++j){
        int i = adjacency[index*12+j];
        
        if(i==-1){
            continue;
        }
        if(lowgradientindices[i]==1){
            continue;
        }
        if((data[i]>data[largetst_index] or (data[i]==data[largetst_index] and i>largetst_index))){
            mini = data[i]-data[index];
            
            largetst_index = i;
            // }
            
        };
    };
    // int row_l = (largetst_index % (height * width)) / width;
    // int row_i = (index % (height * width)) / width;
    
    // int row_diff = row_l - row_i;
    // int col_diff = (largetst_index % width) - (index % width);

    // int dep_diff = (largetst_index /(width * height)) - (index /(width * height));
    // int x_l = (largetst_index / (height)) % width;
    // int x_i = (index / (height)) % width;
    int y_diff = (largetst_index / (width)) % height - (index / (width)) % height;
    // int y_diff = row_l - row_i;
    int x_diff = (largetst_index % width) - (index % width);

    int z_diff = (largetst_index /(width * height)) % depth - (index /(width * height)) % depth;
    direction_as[index] = getDirection(x_diff, y_diff,z_diff);
    
    // if(index==24654784 and type==0){
        
    //     printf("值：");
    //     printf("%d %d %d\n",row_diff, col_diff,dep_diff);
    //     printf("%d %d \n", largetst_index % 750, index % 750);
    //     // printf("%f %f \n" ,decp_data[index],input_data[index]);
    //     // for(int i=0;i<12;i++){
    //     //     int j = adjacency[index*12+i];
    //     //     if(j==-1){
    //     //         break;
    //     //     }
    //     //     printf("%f %f \n" ,decp_data[j],input_data[j]);
    //     // }
        
    // }
    
    

    mini = 0;
    largetst_index = index;
    for(int j =0;j<12;++j){
        int i = adjacency[index*12+j];
        
        if(i==-1){
            break;
        }
        if(lowgradientindices[i]==1){
            continue;
        }
        // if(i==8186 and index==8058 and type==0){
        //     printf("%.20f %.20f\n",data[i]-data[index],data[8057]-data[index]);
        //     // cout<<data[i]<<", "<<data[index]<<", "<<data[8057]<<endl;
        // }
        if((data[i]<data[largetst_index] or (data[i]==data[largetst_index] and i<largetst_index))){
            mini = data[i]-data[index];
            
            largetst_index = i;

            
        };
    };
    
    
    // row_l = (largetst_index % (height * width)) / width;
    // row_i = (index % (height * width)) / width;
    
    // row_diff = row_l - row_i;
    // col_diff = (largetst_index % width) - (index % width);

    // dep_diff = (largetst_index /(width * height)) - (index /(width * height));
    y_diff = (largetst_index / (width)) % height - (index / (width)) % height;
    // int y_diff = row_l - row_i;
    x_diff = (largetst_index % width) - (index % width);

    z_diff = (largetst_index /(width * height)) % depth - (index /(width * height)) % depth;
    // direction_as[index] = getDirection(x_diff, y_diff,z_diff);
    direction_ds[index] = getDirection(x_diff, y_diff,z_diff);
    
    
    
    
    
    return;

};
__global__ void checkElementKernel(int* array, int size, int target, bool* result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        if (array[idx] == target) {
            *result = true;
        }
    }
}

__global__ void iscriticle(){
        
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        
        if(i>=num){
            
            return;
        }
        
        bool is_maxima = true;
        bool is_minima = true;
        
        for (int index=0;index<12;index++) {
            int j = adjacency[i*12+index];
            if(j==-1){
                break;
            }
            if(lowgradientindices[j]==1){
                continue;
            }
            
                
            if (decp_data[j] > decp_data[i]) {
                
                is_maxima = false;
                
                break;
            }
            else if(decp_data[j] == decp_data[i] and j>i){
                is_maxima = false;
                break;
            }
        }
        for (int index=0;index< 12;index++) {
            int j = adjacency[i*12+index];
            if(j==-1){
                break;
            }
            if(lowgradientindices[j]==1){
                    continue;
            }
            
            if (decp_data[j] < decp_data[i]) {
                is_minima = false;
                break;
            }
            else if(decp_data[j] == decp_data[i] and j<i){
                is_minima = false;
                break;
            }
        }
        
        
        if((is_maxima && or_maxi[i]!=-1) or (!is_maxima && or_maxi[i]==-1)){
            int idx_fp_max = atomicAdd(&count_f_max, 1);
            
            all_max[idx_fp_max] = i;
            
        }
        
        else if ((is_minima && or_mini[i]!=-1) or (!is_minima && or_mini[i]==-1)) {
            int idx_fp_min = atomicAdd(&count_f_min, 1);// in one instruction
            
            all_min[idx_fp_min] = i;
            
        } 
        
       
        
}

__global__ void get_wrong_index_path1(){

    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
        
    if(i>=num or lowgradientindices[i]==1){
        
        return;
    }
    
    if (or_label[i * 2 + 1] != dec_label[i * 2 + 1]) {
        int idx_fp_max = atomicAdd(&count_p_max, 1);
        // printf("%d %d %d\n",i,or_label[i * 2 + 1],dec_label[i * 2 + 1]);
        all_p_max[idx_fp_max] = i;
            
    }
    if (or_label[i * 2] != dec_label[i * 2]) {
        int idx_fp_min = atomicAdd(&count_p_min, 1);
        all_p_min[idx_fp_min] = i;
        
    }
    
    

    return;
};

__global__ void freeDeviceMemory() {
    // 释放 decp_data 指向的内存
    if (decp_data != nullptr) {
        delete[] decp_data;
        decp_data = nullptr;  // 避免野指针
    }
} 
__global__ void freeDeviceMemory1() {
    // 释放 decp_data 指向的内存
    if (de_direction_as != nullptr) {
        delete[] de_direction_as;
        de_direction_as = nullptr;  // 避免野指针
    }
    if (de_direction_ds != nullptr) {
        delete[] de_direction_ds;
        de_direction_ds = nullptr;  // 避免野指针
    }
}
__global__ void computeAdjacency() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num and lowgradientindices[i]==0) {
        
        int y = (i / (width)) % height; // Get the x coordinate
        int x = i % width; // Get the y coordinate
        int z = (i / (width * height)) % depth;
        int neighborIdx = 0;
        
        for (int d = 0; d < 12; d++) {
            
            int dirX = directions1[d * 3];     
            int dirY = directions1[d * 3 + 1]; 
            int dirZ = directions1[d * 3 + 2]; 
            int newX = x + dirX;
            int newY = y + dirY;
            int newZ = z + dirZ;
            int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
            // if(lowgradientindices[r]==1){
            //     continue;
            // }
            if (newX >= 0 && newX < width && newY >= 0 && newY < height && r < width*height*depth && newZ<depth && newZ>=0 && lowgradientindices[r]==0) {
                
                adjacency[i * maxNeighbors + neighborIdx] = r;
                neighborIdx++;

            }
        }

        // Fill the remaining slots with -1 or another placeholder value
        
        for (int j = neighborIdx; j < maxNeighbors; ++j) {
            adjacency[i * maxNeighbors + j] = -1;
        }
    }
}

__device__ unsigned long long doubleToULL(double value) {
    return *reinterpret_cast<unsigned long long*>(&value);
}

__device__ double ULLToDouble(unsigned long long value) {
    return *reinterpret_cast<double*>(&value);
}



__device__ double atomicCASDouble(double* address, double val) {
   
    uint64_t* address_as_ull = (uint64_t*)address;
    uint64_t old_val_as_ull = *address_as_ull;
    uint64_t new_val_as_ull = __double_as_longlong(val);
    uint64_t assumed;


    assumed = old_val_as_ull;
    
    old_val_as_ull = atomicCAS((unsigned long long int*)address_as_ull, (unsigned long long int)assumed, (unsigned long long int)new_val_as_ull);
    return __longlong_as_double(old_val_as_ull);
}
void saveVectorToBinFile(const std::vector<int>* vecPtr, const std::string& filename) {
    if (vecPtr == nullptr) {
        std::cerr << "pointer empty" << std::endl;
        return;
    }

    
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "can not open file " << filename << " to write" << std::endl;
        return;
    }

    
    size_t size = vecPtr->size();
    outfile.write(reinterpret_cast<const char*>(&size), sizeof(size));

    
    for (size_t i = 0; i < size; ++i) {
        int value = (*vecPtr)[i];
        if (value == -1) {
            int index = static_cast<int>(i/2);
            outfile.write(reinterpret_cast<const char*>(&index), sizeof(index));
            
        } else {
            outfile.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
    }


    
    outfile.close();
}
__device__ int swap(int index, double delta){
    int update_successful = 0;
    double oldValue = d_deltaBuffer[index];
    while (update_successful==0) {
        double current_value = d_deltaBuffer[index];
        if (-delta > current_value) {
            double swapped = atomicCASDouble(&d_deltaBuffer[index], delta);
            if (swapped == current_value) {
                update_successful = 1;
                
            } else {
                oldValue = swapped;
            }
        } else {
            update_successful = 1; 
    }
    }
}




__global__ void clearStacksKernel(LockFreeStack<double> stacks) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num) {
        stacks.clear();
    }
}



__global__ void fix_maxi_critical1(int direction, int cnt){
    int index_f = blockIdx.x * blockDim.x + threadIdx.x;
    
    
        
    double delta;
    
    int index;
    int next_vertex;
   
    if (direction == 0 && index_f<count_f_max){
        
        index = all_max[index_f];
        // if vertex is a regular point.
        if (or_maxi[index]!=-1){
            
            // find its largest neighbor
            
            next_vertex = from_direction_to_index1(index,or_maxi[index]);
            
    
            double d = ((input_data[index] - bound) + decp_data[index]) / 2.0 - decp_data[index];
            
            if(decp_data[index]<decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index<next_vertex)){
                return;
            }

            
            // printf("%.17f %.17f %.17f %.17f\n", decp_data[next_vertex], decp_data[index], input_data[index] - bound, input_data[next_vertex] - bound);
            double oldValue = d_deltaBuffer[index];
            if (d > oldValue) {
                swap(index, d);
            }  

            return;
            
            
            
        
        }
        else{
            // if is a maximum in the original data;
            
            int largest_index = from_direction_to_index1(index, de_direction_as[index]);
            
            if(decp_data[index]>decp_data[largest_index] or(decp_data[index]==decp_data[largest_index] and index>largest_index)){
                return;
            }

            double d = ((input_data[largest_index] - bound) + decp_data[largest_index]) / 2.0 - decp_data[largest_index];
            
            double oldValue = d_deltaBuffer[largest_index];
            if (d > oldValue) {
                swap(largest_index, d);
            }  

            return;
        }
        
        
    
    }
    
    else if (direction != 0 && index_f<count_f_min && lowgradientindices[all_min[index_f]]==0){
        index = all_min[index_f];
        
        if (or_mini[index]!=-1){
           
            
            int next_vertex= from_direction_to_index1(index,or_mini[index]);
            

            double d = ((input_data[next_vertex] - bound) + decp_data[index]) / 2.0 - decp_data[next_vertex];
            
            if(decp_data[index]>decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index>next_vertex)){
                return;
            }

            
            // printf("%.17f %.17f %.17f %.17f\n", decp_data[next_vertex], decp_data[index], input_data[index] - bound, input_data[next_vertex] - bound);
            double oldValue = d_deltaBuffer[next_vertex];
            if (d > oldValue) {
                swap(next_vertex, d);
            }  

            return;

            
            
            
            // if(diff>=1e-16){
                
            //     if(abs(decp_data[index]-decp_data[next_vertex])<1e-16){
                    
                      
                    
            //             while(abs(input_data[next_vertex]-decp_data[index]-diff)>bound and diff>=2e-16){
            //                 diff/=2;
            //             }
                        
            //             if(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and diff>=1e-16){
                           
            //                 delta = (decp_data[index]-diff) - decp_data[next_vertex];
                            
                            
            //                 double oldValue = d_deltaBuffer[next_vertex];
                        
            //             if (delta > oldValue) {
                                
                                
            //                     swap(next_vertex, delta);
                                
            //                 }
                            
            //             }
            //             else if(d1>=1e-16){
                            
            //                 delta = -d1;
                    
                            
            //                 double oldValue = d_deltaBuffer[next_vertex];
                    
            //             if (delta > oldValue) {
                                
                                
            //                     swap(next_vertex,delta);
            //                 }
            //             }
            //             else if(d>=1e-16){

                            
            //                 delta = d;
                            
            //                 double oldValue = d_deltaBuffer[index];
                      
            //             if (delta > oldValue) {
                                
                                
            //                     swap(index, delta);
            //                 }
                            
            //             }

                    
                    
            //     }
            //     else{
            //         if(decp_data[index]<=decp_data[next_vertex]){
                        
            //                 while(abs(input_data[next_vertex]-decp_data[index]+diff)>bound and diff >= 2e-16){
            //                         diff/=2;
            //                 }
                            
                            
            //                 if (abs(input_data[next_vertex]-decp_data[index]+diff)<=bound and decp_data[index]<=decp_data[next_vertex] and diff>=1e-16){
                               
            //                     while(abs(input_data[next_vertex]-decp_data[index]+diff)<bound and diff <= 1e-17){
            //                         diff*=2;
            //                     }
            //                     if(abs(input_data[next_vertex]-decp_data[index]+diff)<=bound){
                                    
            //                         delta = (decp_data[index]-diff) - decp_data[next_vertex];
                                    
                                    
                                    
            //                         double oldValue = d_deltaBuffer[next_vertex];
                        
            //             if (delta > oldValue) {
                               
                                
            //                     swap(next_vertex, delta);
                                
                                
            //                 }
            //                     }
                                
            //                 }
                            
            //                 else if(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<=bound and decp_data[index]<=decp_data[next_vertex] and d1>=1e-16){
            //                     while(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<bound and d1<=1e-16){
            //                         d1*=2;
            //                     }
                                
            //                     if(abs(input_data[next_vertex]-decp_data[next_vertex]+d1)<=bound and d1>=1e-16){
                                    
            //                         delta = -d1;
                                    
                            
                            
            //                 double oldValue = d_deltaBuffer[next_vertex];
                        
                        
            //             if (delta > oldValue) {
                                
            //                     swap(next_vertex, delta);
                                
            //                 }
            //                     }
                                
                                
            //                 }
            //                 else{
                               
            //                     delta = (input_data[next_vertex] - bound)- decp_data[next_vertex];
                                
                            
                            
            //                 double oldValue = d_deltaBuffer[next_vertex];
                        
            //             if (delta > oldValue) {
                               
            //                     swap(next_vertex, delta);
                                
            //                 }
                               
            //                 }
                            
                            
                        
                        
            //     };

            //     }
                
                

                
            // }

            // else{
                
            //     if(decp_data[index]<decp_data[next_vertex]){
                    
            //             double t = (decp_data[index]-(input_data[index]-bound))/2.0;
            //             if(abs(input_data[next_vertex]-decp_data[index]+t)<bound and t>=1e-16){
                            
                            
            //                 delta = (decp_data[index]-t) - decp_data[next_vertex];
                            
                            
                            
            //                 double oldValue = d_deltaBuffer[next_vertex];
                            
                       
            //             if (delta > oldValue) {
                                
            //                     swap(next_vertex, delta);
                                
            //                 }
                            
            //             }
            //             else{
                            
            //                 delta = (input_data[index] + bound) - decp_data[index];
                            
            //                 double oldValue = d_deltaBuffer[index];
                        
            //             if (delta > oldValue) {
                               
            //                     swap(index, delta);
                                
            //                 }
                            
            //             }
            //     }
                
            //     else if(decp_data[index]==decp_data[next_vertex]){
            //         double d = (bound - (input_data[index]-decp_data[index]))/2.0;
                    
            //         if(abs(input_data[index]-decp_data[index]-d)<=bound){
                        
            //             delta = d;
                        
                        
            //                 double oldValue = d_deltaBuffer[index];
                        
            //             if (delta > oldValue) {
                                
            //                     swap(index, delta);
                                
            //                 }
                        
            //         }
            //         else if(abs(input_data[next_vertex]-decp_data[next_vertex]+d)<=bound){
                        
            //             delta = -d;
                        
                        
            //             double oldValue = d_deltaBuffer[next_vertex];
                        
            //             if (delta > oldValue) {
                                
            //                     swap(next_vertex, delta);
                                
            //                 }
            //         }
            //     }
            // }
            

            
            
            
       
        
        }
    
        else{
            
            int largest_index = from_direction_to_index1(index,de_direction_ds[index]);
            double diff = (bound-(input_data[index]-decp_data[index]))/2.0;
            
            
            if(decp_data[index]<decp_data[largest_index] or (decp_data[index]==decp_data[largest_index] and index<largest_index)){
                // de_direction_ds[index] = -1;
                return;
            }
            
            double d = ((input_data[index] - bound) + decp_data[index]) / 2.0 - decp_data[index];
            
            

            
            // printf("%.17f %.17f %.17f %.17f\n", decp_data[next_vertex], decp_data[index], input_data[index] - bound, input_data[next_vertex] - bound);
            double oldValue = d_deltaBuffer[index];
            if (d > oldValue) {
                swap(index, d);
            }  

            return;

            if (diff>=1e-16){
                if (decp_data[index]>=decp_data[largest_index]){
                    while(abs(input_data[index]-decp_data[index]+diff)>bound and diff>=2e-16){
                        diff/=2;
                    }
                    
                    
                    if(abs(input_data[index]-decp_data[index]+diff)<=bound){
                       
               
                        delta = -diff;
                        
                        
                        double oldValue = d_deltaBuffer[index];
                        
                        if (delta > oldValue) {
                               
                                swap(index, delta);
                                
                            }
                        
                    }
                    
                    
                }                    
            }
            
                    
            else{
                if (decp_data[index]>=decp_data[largest_index]){
                    
                   
                    delta = ((input_data[index] - bound) - decp_data[index]);
                    
                   
                            double oldValue = d_deltaBuffer[index];
                        
                        if (delta > oldValue) {
                               
                                swap(index, delta);
                                
                            }
                    
                }   
                
    
            }


               
        }

        
    }    
    

    

    return;
}



__global__ void initializeKernel(double value) {
    
    if (threadIdx.x == 0) {
        d_stacks.clear();
        id_stacks.clear();
    }


}

__global__ void initializeKernel1(double value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<num){
        d_deltaBuffer[tid] = -2000.0;
    }

}



__global__ void fixpath11(int direction){
    int index_f = blockIdx.x * blockDim.x + threadIdx.x;
    double delta;
    if(direction == 0){
        if(index_f<count_p_max && lowgradientindices[all_p_max[index_f]]==0){

        
        int index = all_p_max[index_f];
        int cur = index;
        while (or_maxi[cur] == de_direction_as[cur]){
            int next_vertex =  from_direction_to_index1(cur,de_direction_as[cur]);
            
            if(de_direction_as[cur]==-1 && next_vertex == cur){
                cur = -1;
                break;
            }
            if(next_vertex == cur){
                cur = next_vertex;
                break;
            };
            
            cur = next_vertex;
        }

        int start_vertex = cur;
        
        
        if (start_vertex==-1) return;
        else{
            
            int false_index= from_direction_to_index1(cur,de_direction_as[cur]);
            int true_index= from_direction_to_index1(cur, or_maxi[cur]);
            if(false_index==true_index) return;

            double d = ((input_data[false_index] - bound) + decp_data[false_index]) / 2.0 - decp_data[false_index];
            
            

            
            // printf("%.17f %.17f %.17f %.17f\n", decp_data[next_vertex], decp_data[index], input_data[index] - bound, input_data[next_vertex] - bound);
            double oldValue = d_deltaBuffer[false_index];
            if (d > oldValue) {
                swap(false_index, d);
            }  

            return;

            
            
            double diff = (input_data[true_index]-bound-decp_data[false_index])/2.0;
            
            // double d = (decp_data[false_index]-input_data[false_index]+bound)/2.0;
            
            if(decp_data[false_index]<decp_data[true_index]){
                de_direction_as[cur]=or_maxi[cur];
                return;
            }
            
            double threshold = -DBL_MAX;;
            int smallest_vertex = false_index;
            
            for(int j=0;j<12;j++){
                int i = adjacency[12*false_index+j];
                if(i==-1) continue;
                if(input_data[i]<input_data[false_index] and input_data[i]>threshold and i!=false_index){
                    smallest_vertex = i;
                    threshold = input_data[i];
                }
            }
            
            threshold = decp_data[smallest_vertex];

            double threshold1 = DBL_MAX;;
            int smallest_vertex1 = true_index;
            
            for(int j=0;j<12;j++){
                int i = adjacency[12*true_index+j];
                if(i==-1) continue;
                if(input_data[i]>input_data[true_index] and input_data[i]<threshold1 and i!=true_index){
                    smallest_vertex1 = i;
                    threshold = input_data[i];
                }
            }
            
            threshold1 = decp_data[smallest_vertex1];

            if (diff>=1e-16 or d>=1e-16){
                if (decp_data[false_index]>=decp_data[true_index]){

                    
                    
                    while(abs(input_data[false_index]-decp_data[false_index] + d)>bound and d>2e-16){
                                d/=2;
                    }
                    
                    
                   
                    while(abs(input_data[true_index]-(decp_data[false_index] + diff))>bound and diff>2e-16){
                                diff/=2;
                    }
                    if(decp_data[true_index]<=threshold and threshold>=decp_data[false_index]){
                            
                            while(decp_data[false_index] + diff > threshold and diff>=2e-16)
                            {
                                diff/=2;
                            }
                            
                            
                    }
                    
                    if (abs(input_data[true_index]-(decp_data[false_index] + diff))<=bound and decp_data[false_index]>=decp_data[true_index]){
                        
                        delta = decp_data[false_index] + diff - decp_data[true_index];
                            
                        
                        double oldValue = d_deltaBuffer[true_index];
                        
                        if (delta > oldValue) {
                                swap(true_index, delta);
                                
                            }
                    }
                    if (abs(input_data[false_index]-decp_data[false_index] + d)<=bound){
                        
                        // decp_data[false_index] -=d;
                        delta = -d;
                        
                        
                        
                        double oldValue = d_deltaBuffer[false_index];
                        
                        if (delta > oldValue) {
                              
                                swap(false_index, delta);
                                
                            }
                    }
                    
                    
                        
                }

                else{
                    de_direction_as[cur] = or_maxi[cur];
                }
                    
            }
            
            else{
               
                double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
                
                if (decp_data[false_index]>=decp_data[true_index]){
                    if(abs(input_data[false_index]-((decp_data[false_index]+input_data[true_index]-bound)/2.0))<=bound){
                        
                        delta = (decp_data[false_index]+input_data[true_index]-bound)/2.0-decp_data[false_index];
                        
                            
                            
                        
                        
                        double oldValue = d_deltaBuffer[false_index];
                        
                        if (delta > oldValue) {
                                
                                swap(false_index, delta);
                                
                            }
                    }
                        
                    else{
                        
                        delta =  input_data[false_index] - bound-decp_data[false_index];
                        
                        
                        double oldValue = d_deltaBuffer[false_index];
                        
                        if (delta > oldValue) {
                                
                                swap(false_index, delta);
                                
                            }
                    }
                    
                }
                else{
                    de_direction_as[cur] = or_maxi[cur];
                };        
            }
            
        }
        }
    }

    else 
    {
        if(index_f<count_p_min && lowgradientindices[all_p_min[index_f]]==0){
            
        int index = all_p_min[index_f];
        int cur = index;
        
        
        while (or_mini[cur] == de_direction_ds[cur]){
            
            int next_vertex = from_direction_to_index1(cur,de_direction_ds[cur]);
            if (next_vertex == cur){
                cur = next_vertex;
                break;
            }
            cur = next_vertex;

            
                
        }
    
        int start_vertex = cur;
        
        if (start_vertex==-1) return;
        
        else{
            
            int false_index= from_direction_to_index1(cur,de_direction_ds[cur]);
            int true_index= from_direction_to_index1(cur, or_mini[cur]);
            if(false_index==true_index) return;

            double d = ((input_data[true_index] - bound) + decp_data[true_index]) / 2.0 - decp_data[true_index];
            
            
            // printf("%.17f %.17f %.17f %.17f\n", decp_data[next_vertex], decp_data[index], input_data[index] - bound, input_data[next_vertex] - bound);
            double oldValue = d_deltaBuffer[true_index];
            if (d > oldValue) {
                swap(true_index, d);
            }  

            return;

            
            double diff = (bound-(input_data[true_index]-decp_data[false_index]))/2.0;
            
            // double d = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
            
            if(decp_data[false_index]>decp_data[true_index]){
                de_direction_ds[cur]=or_mini[cur];
                return;
            }
            
            if(diff>=1e-16 or d>=1e-16){
                if(decp_data[false_index]<=decp_data[true_index]){
                    
                   
                        while(abs(input_data[false_index]-decp_data[false_index] - d)>bound and d>=2e-17){
                            d/=2;
                        }
                        while(abs(input_data[true_index]-(decp_data[false_index] - diff))>bound and diff>=2e-17){
                                    diff/=2;
                        }
                        if(abs(input_data[true_index]-(decp_data[false_index] - diff))<=bound and decp_data[false_index]<=decp_data[true_index]){
                            
                            delta =  decp_data[false_index] - diff- decp_data[true_index];
                            
                           
                            
                            double oldValue = d_deltaBuffer[true_index];
                        
                        if (delta > oldValue) {
                                
                                swap(true_index, delta);
                                
                            }
                        }
                        if(abs(input_data[false_index]-decp_data[false_index] - d)<=bound){
                            
                            delta =  d;
                            
                            
                            
                        double oldValue = d_deltaBuffer[false_index];
                       
                        if (delta > oldValue) {

                                swap(false_index, delta);
                                
                            }
                        }
                        
                        
                        if (decp_data[false_index]==decp_data[true_index]){
                            if(abs(input_data[false_index]-decp_data[false_index] - d)<=bound){
                    
                                delta =  d;
                                
                                
                                double oldValue = d_deltaBuffer[false_index];
                       
                        if (delta > oldValue) {
                               
                                swap(false_index, delta);
                                
                            }
                        }
                       
                    }
                    // }
                    
                }
            
                else{
                    de_direction_ds[cur] = or_mini[cur];
                }
            }

            else{
                
                double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
                
                if(decp_data[false_index]<=decp_data[true_index]){
                    if(abs(input_data[false_index]-((decp_data[true_index]+input_data[true_index]+bound)/2.0))<=bound){
                       
                        delta =  (decp_data[true_index]+input_data[true_index]+bound)/2.0 - decp_data[false_index];
                            
                            double oldValue = d_deltaBuffer[false_index];
                        
                        if (delta > oldValue) {
                                
                                swap(false_index, delta);
                                
                            }
                    }
                    else{
                        
                        delta =  input_data[false_index] + bound - decp_data[false_index];
                           
                            
                            double oldValue = d_deltaBuffer[false_index];
                        
                        if (delta > oldValue) {
                                
                                swap(false_index, delta);
                                
                            }
                    }
                    while(abs(input_data[false_index]-(decp_data[false_index] + diff))>bound and diff>=2e-17){
                        diff/=2;
                    }
                    if (decp_data[false_index]==decp_data[true_index]){
                            double diff = (bound-(input_data[false_index]-decp_data[false_index]))/2.0;
                           
                            delta =  diff;
                            
                            
                            double oldValue = d_deltaBuffer[false_index];
                       
                        if (delta > oldValue) {
                                
                                swap(false_index, delta);
                                
                            }
                    }
                
                }
            
                else{
                    de_direction_ds[cur] = or_mini[cur];
                }
            }
        }
    }
    }
    return;
};

__global__ void initialize2DArray(double** d_array, int* sizes, int rows) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < sizes[row]) {
        d_array[row][col] = row * 10 + col; // Example initialization
    }
}




void resizeArray(double** d_array, int* sizes, int row, int new_size) {
    double* d_subarray;
    cudaMalloc(&d_subarray, new_size * sizeof(double));
    cudaMemcpy(d_subarray, d_array[row], sizes[row] * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaFree(d_array[row]);
    d_array[row] = d_subarray;
    sizes[row] = new_size;
}





__global__ void applyDeltaBuffer1() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num) {
        if(d_deltaBuffer[tid] != -2000){
            if(abs(d_deltaBuffer[tid]) > 1e-15) decp_data[tid] += d_deltaBuffer[tid];
            else decp_data[tid] = input_data[tid] - bound;
        }

        
    }
    
}


__global__ void getlabel(int *un_sign_ds, int *un_sign_as, int type=0){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int *direction_as;
    int *direction_ds;
    int *label;
    
    if(i>=num){
        return;
    }
    
    if(type==0){
        direction_as = de_direction_as;
        direction_ds = de_direction_ds;
        label = dec_label;
    }
    else{
        direction_as = or_maxi;
        direction_ds = or_mini;
        label = or_label;
    }
    
    int cur = label[i*2+1];
    
    
        int next_vertex;
        
        if (cur!=-1 and direction_as[cur]!=-1){
            
            int direc = direction_as[cur];
            
            
            next_vertex = from_direction_to_index1(cur, direc);
            
            
            if(label[next_vertex*2+1] == -1){
                label[i*2+1] = next_vertex;
                
            }
            
            else{
                
                label[i*2+1] = label[next_vertex*2+1];
                
                
            }
            
            if (direction_as[label[i*2+1]] != -1){
                
                *un_sign_as+=1;  
                
            }
            
        }
    
    
    
    
        cur = label[i*2];
        int next_vertex1;
        
        
        if (cur!=-1 and label[cur*2]!=-1){
            
            int direc = direction_ds[cur];
            
            next_vertex1 = from_direction_to_index1(cur, direc);
            
            if(label[next_vertex1*2] == -1){
                label[i*2] = next_vertex1;
                
            }
            
            else if(label[label[next_vertex1*2]*2] == -1){
                label[i*2] = label[next_vertex1*2];  
            }
            
            else if(direction_ds[i]!=-1){
               
                if(label[next_vertex1*2]!=-1){
                    label[i*2] = label[next_vertex1*2];
                }
                
                else{

                    label[i*2] = next_vertex1;
                }
                
                
            }
            
            if (direction_ds[label[i*2]]!=-1){
                *un_sign_ds+=1;
                }
            } 
        
        
    return;

}


__global__ void initializeWithIndex(int size, int type=0) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int* label;
    if (index < size) {
        int *direction_ds;
        int *direction_as;
        if(type==0){
            direction_ds = de_direction_ds;
            direction_as = de_direction_as;
            label = dec_label;
        }
        else{
            direction_ds = or_mini;
            direction_as = or_maxi;
            label = or_label;
        }

        if(direction_ds[index]!=-1){
            label[index*2] = index;
            
        }
        else{
            label[index*2] = -1;
        }

        if(direction_as[index]!=-1){
            label[index*2+1] = index;
        }
        else{
            label[index*2+1] = -1;
        }
    }
}


void init_inputdata(std::vector<int> *a,std::vector<int> *b,std::vector<int> *c,std::vector<int> *d,std::vector<double> *input_data1,std::vector<double> *decp_data1,std::vector<int>* dec_label1,std::vector<int>* or_label1, int width1, int height1, int depth1, std::vector<int> *low,double bound1,float &datatransfer,float &finddirection, int preserve_min, int preserve_max, int preserve_path){
    int* temp;
    
    int* temp1;
    int* d_data;
    int* or_l;
    int* dec_l;
    
    

    float mappath_path = 0.0;
    float getfpath = 0.0;
    float fixtime_path = 0.0;
    float finddirection1 = 0.0;
    float getfcp = 0.0;
    float fixtime_cp = 0.0;
    double* temp3;
    double* temp4;
    
    LockFreeStack<double> stack_temp;
    LockFreeStack<int> id_stack_temp;

    std::vector<std::vector<float>> time_counter;
    int total_cnt;
    int sub_cnt;
    int num1 = width1*height1*depth1;
    int h_un_sign_as = num1;
    int h_un_sign_ds = num1;
    
    float elapsedTime;
    // int initialValue = 0;
    cout<<bound1<<endl;
    
    
    cout<<num1<<endl;
    
    int *un_sign_as;
    cudaMalloc((void**)&un_sign_as, sizeof(int));
    cudaMemset(un_sign_as, 0, sizeof(int));

    int *un_sign_ds;
    cudaMalloc((void**)&un_sign_ds, sizeof(int));
    cudaMemset(un_sign_ds, 0, sizeof(int));

    cout<<width1<<endl;
    
    std::vector<int> h_all_p_max(num1);
    std::vector<int> h_all_p_min(num1);


    cudaError_t cudaStatus= cudaMemcpyToSymbol(width, &width1, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed101: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
    cudaMemcpyToSymbol(height, &height1, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(depth, &depth1, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(num, &num1, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpyToSymbol(bound, &bound1, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed91: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
    
    cudaMalloc(&temp, num1 * sizeof(int));
    cudaMalloc(&temp1, num1 * sizeof(int));
    cudaStatus = cudaMalloc(&temp3, num1  * sizeof(double));
    cudaMalloc(&temp4, num1  * sizeof(double));

    
    



   if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed89: " << cudaGetErrorString(cudaStatus) << std::endl;
        }

    

    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed89: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
    
    cudaMalloc(&d_data, num1 * sizeof(int));
    cudaMalloc(&or_l, num1 * 2  * sizeof(int));
    cudaMalloc(&dec_l, num1 * 2 * sizeof(int));
    
    
    cudaEvent_t start, stop;

    cudaEventCreate(&start);

    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    

    cudaStatus = cudaMemcpy(temp3, input_data1->data(), num1 * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed89: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
    cudaStatus = cudaMemcpy(temp4, decp_data1->data(), num1 * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed17: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
    cudaStatus = cudaMemcpy(d_data, low->data(), num1 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed27: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    datatransfer+=elapsedTime;
    
    int *d_temp;  
    size_t size = num1 * sizeof(int);

    
    cudaMalloc(&d_temp, size);

    
    
    cudaEventRecord(start, 0);

    cudaMemcpyToSymbol(all_max, &d_temp, sizeof(int*));
    
    cudaMemcpyToSymbol(lowgradientindices, &d_data, sizeof(int*));
    
    int *d_temp1;  
    size_t size1 = num1 * sizeof(int);

    
    cudaMalloc(&d_temp1, size1);

    
    cudaMemcpyToSymbol(all_min, &d_temp1, sizeof(int*));

    int *p_temp; 
   

    
    cudaMalloc(&p_temp, size1);

    
    cudaMemcpyToSymbol(all_p_min, &p_temp, sizeof(int*));

    int *p_temp1;  
    
    cudaMalloc(&p_temp1, size1);

    
    cudaMemcpyToSymbol(all_p_max, &p_temp1, sizeof(int*));

    int *d_temp2;  
    size_t size4 = num1  * sizeof(int);
    
    cudaMalloc(&d_temp2, size4);

    
    cudaMemcpyToSymbol(de_direction_as, &d_temp2, sizeof(int*));
    cudaMemcpyToSymbol(or_label, &or_l, sizeof(int*));
    cudaMemcpyToSymbol(dec_label, &dec_l, sizeof(int*));

    int *d_temp3;  
    size_t size3 = num1 * sizeof(int);

    
    cudaMalloc(&d_temp3, size3);

    
    cudaStatus = cudaMemcpyToSymbol(de_direction_ds, &d_temp3, sizeof(int*));
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed87: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
    cudaStatus = cudaMemcpyToSymbol(or_maxi, &temp, sizeof(int*));
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed83: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
    cudaStatus = cudaMemcpyToSymbol(or_mini, &temp1, sizeof(int*));
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed84: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
    cudaMemcpyToSymbol(input_data, &temp3, sizeof(double*));
    cudaMemcpyToSymbol(decp_data, &temp4, sizeof(double*));

    
    
    dim3 blockSize(256);
    
    dim3 gridSize((num1 + blockSize.x - 1) / blockSize.x);
    
    int* tempDevicePtr = nullptr;
    size_t arraySize = num1*12; 
    cudaStatus = cudaMalloc(&tempDevicePtr, arraySize * sizeof(int));
    
    cudaStatus = cudaMemcpyToSymbol(adjacency, &tempDevicePtr, sizeof(tempDevicePtr));
   

    
    

    

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    datatransfer+=elapsedTime;
    cudaEventRecord(start, 0);
    computeAdjacency<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout<<"comupte_adjacency: "<<elapsedTime<<endl;
   
    cudaEventRecord(start, 0);

    find_direction<<<gridSize, blockSize>>>(1);
    
    
    
   
    double init_value = -2*bound1;
    double* buffer_temp;
    cudaMalloc(&buffer_temp, num1  * sizeof(double));
    cudaMemcpyToSymbol(d_deltaBuffer, &buffer_temp, sizeof(double*));

    double* array_temp;
    cudaMalloc(&array_temp, num1  * sizeof(int));
    cudaMemcpyToSymbol(id_array, &array_temp, sizeof(int*));

    // initializeKernel1<<<gridSize, blockSize>>>(init_value);

    
    cudaEventRecord(start, 0);
   
    find_direction<<<gridSize, blockSize>>>();
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    
    cudaEventRecord(start, 0);
    
    int initialValue = 0;
    cudaStatus = cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
    
    cudaStatus = cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
    iscriticle<<<gridSize,blockSize>>>();
    
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    // double h_s[num1];
    int host_count_f_max;
    cudaStatus = cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpyToSymbol failed11: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    int host_count_f_min;
    cudaStatus = cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed12: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    
    int cnt  = 0;
    
    std::vector<int> h_all_max(num1);
    int h_count_f_max = 0;
    
    if(preserve_max == 0) host_count_f_max = 0;
    if(preserve_min == 0) host_count_f_min = 0;
    
    
    while(host_count_f_min>0 || host_count_f_max>0){
            cout<<host_count_f_min<<", "<<host_count_f_max<<endl;
            
            initializeKernel1<<<gridSize, blockSize>>>(init_value);
            
            
            cudaDeviceSynchronize();
            
            dim3 blockSize1(256);
            
            dim3 gridSize1((host_count_f_max + blockSize1.x - 1) / blockSize1.x);
            cudaEventRecord(start, 0);
            int threads_per_block = 256;
            int num_blocks = (num1+threads_per_block-1)/threads_per_block;
            cudaEventRecord(start, 0);
            if(preserve_max == 1)
            {
                fix_maxi_critical1<<<gridSize1, blockSize1>>>(0,cnt);
            }   
 
            dim3 blocknum(256);
            dim3 gridnum((host_count_f_min + blocknum.x - 1) / blocknum.x);
            if(preserve_min == 1)
            {
                fix_maxi_critical1<<<gridnum, blocknum>>>(1,cnt);
            
            }
            
            applyDeltaBuffer1<<<gridSize, blockSize>>>();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);

            cudaDeviceSynchronize();
            
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            
            cudaEventElapsedTime(&elapsedTime, start, stop);
            cudaStatus = cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
            if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed11: " << cudaGetErrorString(cudaStatus) << std::endl;
            }
            cudaStatus = cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
            
            if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed11: " << cudaGetErrorString(cudaStatus) << std::endl;
            }

            cudaDeviceSynchronize();
            iscriticle<<<gridSize, blockSize>>>();
            find_direction<<<gridSize,blockSize>>>();
            cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
            
            cudaDeviceSynchronize();
            if(preserve_max == 0) host_count_f_max = 0;
            if(preserve_min == 0) host_count_f_min = 0;
            
    }

    
    if(preserve_path ==0 || preserve_max == 0 || preserve_min == 0) 
    {
        cudaMemcpy(decp_data1->data(), temp4, num1 * sizeof(double), cudaMemcpyDeviceToHost);
        return;
    }
    cudaEventRecord(start, 0);

    initializeWithIndex<<<gridSize, blockSize>>>(num1,0);
    initializeWithIndex<<<gridSize, blockSize>>>(num1,1);
    
    cudaEventRecord(start, 0);
   
    while(h_un_sign_as>0 or h_un_sign_ds>0){
        
        int zero = 0;
        int zero1 = 0;

        // cout<<"找path"<<h_un_sign_as<<", "<<h_un_sign_ds<<endl;
        cudaMemcpy(un_sign_as, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(un_sign_ds, &zero1, sizeof(int), cudaMemcpyHostToDevice);
        getlabel<<<gridSize, blockSize>>>(un_sign_as,un_sign_ds,0);
        
        cudaMemcpy(&h_un_sign_as, un_sign_as, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_un_sign_ds, un_sign_ds, sizeof(int), cudaMemcpyDeviceToHost);
        
        
    }   
    
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    
    h_un_sign_as = num1;
    h_un_sign_ds = num1;
    while(h_un_sign_as>0 or h_un_sign_ds>0){
        
        int zero = 0;
        int zero1 = 0;


        cudaMemcpy(un_sign_as, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(un_sign_ds, &zero1, sizeof(int), cudaMemcpyHostToDevice);
        getlabel<<<gridSize, blockSize>>>(un_sign_as,un_sign_ds,1);
        
        cudaMemcpy(&h_un_sign_as, un_sign_as, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_un_sign_ds, un_sign_ds, sizeof(int), cudaMemcpyDeviceToHost);
        
        
        
    }
    
    
    cudaMemcpy(dec_label1->data(), dec_l, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(or_label1->data(), or_l, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    
    saveVectorToBinFile(dec_label1, "dec_jet_"+std::to_string(bound)+".bin");
    saveVectorToBinFile(or_label1, "or_jet_"+std::to_string(bound)+".bin");
    
    cudaMemcpyToSymbol(count_p_max, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(count_p_min, &initialValue, sizeof(int));
    get_wrong_index_path1<<<gridSize, blockSize>>>();

    int host_count_p_max;
    
    cudaStatus = cudaMemcpyFromSymbol(&host_count_p_max, count_p_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpyToSymbol failed11: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    int host_count_p_min;
    cudaStatus = cudaMemcpyFromSymbol(&host_count_p_min, count_p_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed12: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    
    while(host_count_p_min>0 or host_count_p_max>0 or host_count_f_min>0 or host_count_f_max>0){
        cout<<host_count_f_max<<", "<<host_count_f_min<<","<< host_count_p_max<<","<<host_count_p_min<<endl;
        datatransfer = 0.0;
        mappath_path = 0.0;
        getfpath = 0.0;
        fixtime_path = 0.0;
        finddirection1 = 0.0;
        getfcp = 0.0;
        fixtime_cp = 0.0;
        sub_cnt = 0;
        total_cnt+=1;

        initializeKernel1<<<gridSize, blockSize>>>(init_value);
        dim3 blockSize2(256);
        dim3 gridSize2((host_count_p_max + blockSize2.x - 1) / blockSize2.x);


        cudaEventRecord(start, 0);
        fixpath11<<<gridSize2, blockSize2>>>(0 );
        cudaDeviceSynchronize();

        
        
        cudaDeviceSynchronize();
        
        
        
        dim3 blockSize3(256);
        dim3 gridSize3((host_count_p_min + blockSize3.x - 1) / blockSize3.x);
        fixpath11<<<gridSize3, blockSize3>>>(1);
        cudaDeviceSynchronize();


        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        fixtime_path+=elapsedTime;
       
    
        applyDeltaBuffer1<<<gridSize, blockSize>>>();
        cudaDeviceSynchronize();
        cudaEventRecord(start, 0);

        // clearStacksKernel<<<gridSize, blockSize>>>();
        cudaDeviceSynchronize();

        cudaEventRecord(start, 0);
        find_direction<<<gridSize, blockSize>>>();
        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        finddirection1+=elapsedTime;

        cudaStatus = cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
        
        cudaStatus = cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
            


        cudaEventRecord(start, 0);
        iscriticle<<<gridSize, blockSize>>>();
        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        getfcp+=elapsedTime;
        
        cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
        
        cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
        // cout<<host_count_f_max<<", "<<host_count_f_min<<endl;
        while(host_count_f_max>0 or host_count_f_min>0){
        
            cout<<host_count_f_max<<", "<<host_count_f_min<<endl;
            sub_cnt+=1;
            
            dim3 blockSize1(256);
            dim3 gridSize1((host_count_f_max + blockSize1.x - 1) / blockSize1.x);
            // cudaEventRecord(start, 0);
            initializeKernel1<<<gridSize, blockSize>>>(init_value);
            cudaEventRecord(start, 0);
            
            fix_maxi_critical1<<<gridSize1, blockSize1>>>(0,cnt);
            
            cudaDeviceSynchronize();
            
            
            cudaDeviceSynchronize();
            // cudaDeviceSynchronize();
            
            
            dim3 blocknum(256);
            dim3 gridnum((host_count_f_min + blocknum.x - 1) / blocknum.x);
            
            
            fix_maxi_critical1<<<gridnum, blocknum>>>(1,cnt);
            
            cudaDeviceSynchronize();
            
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            
            cudaEventElapsedTime(&elapsedTime, start, stop);
            fixtime_cp+=elapsedTime;
            
            
            cudaStatus = cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
            
            cudaStatus = cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
            


            
            
            applyDeltaBuffer1<<<gridSize, blockSize>>>();
            

            cudaEventRecord(start, 0);
            find_direction<<<gridSize,blockSize>>>();
            
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedTime, start, stop);
            finddirection1+=elapsedTime;
           
            cudaEventRecord(start, 0);
            iscriticle<<<gridSize, blockSize>>>();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            
            cudaEventElapsedTime(&elapsedTime, start, stop);
            getfcp+=elapsedTime;
            
            cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
            
            cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
            
            cudaDeviceSynchronize();
            
           
        }
        
        initializeWithIndex<<<gridSize, blockSize>>>(num1,0);
        
        h_un_sign_as = num1;
        h_un_sign_ds = num1;
        cudaEventRecord(start, 0);
        while(h_un_sign_as>0 or h_un_sign_ds>0){
        
            int zero = 0;
            int zero1 = 0;

            
            cudaMemcpy(un_sign_as, &zero, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(un_sign_ds, &zero1, sizeof(int), cudaMemcpyHostToDevice);
            getlabel<<<gridSize, blockSize>>>(un_sign_as,un_sign_ds,0);
            
            cudaMemcpy(&h_un_sign_as, un_sign_as, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_un_sign_ds, un_sign_ds, sizeof(int), cudaMemcpyDeviceToHost);
           
            
            
        } 
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        mappath_path+=elapsedTime;
        
        cudaMemcpyToSymbol(count_p_max, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(count_p_min, &initialValue, sizeof(int));

        cudaEventRecord(start, 0);
        get_wrong_index_path1<<<gridSize, blockSize>>>();
        
        
    

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        getfpath+=elapsedTime;

        cudaStatus = cudaMemcpyFromSymbol(&host_count_p_max, count_p_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed11: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
        cudaStatus = cudaMemcpyFromSymbol(&host_count_p_min, count_p_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
                std::cerr << "cudaMemcpyToSymbol failed12: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
        cudaStatus = cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
        cudaStatus = cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));

        cudaEventRecord(start, 0);
        iscriticle<<<gridSize, blockSize>>>();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        getfcp+=elapsedTime;


        cudaStatus = cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed11: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
        cudaStatus = cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
                std::cerr << "cudaMemcpyToSymbol failed12: " << cudaGetErrorString(cudaStatus) << std::endl;
        }

        std::vector<float> temp;
        temp.push_back(finddirection1);
        temp.push_back(getfcp);
        temp.push_back(fixtime_cp);
        temp.push_back(mappath_path);
        temp.push_back(getfpath);
        temp.push_back(fixtime_path);
        temp.push_back(sub_cnt);
        time_counter.push_back(temp);
    
    }
    
    
    
    
    cudaMemcpy(a->data(), temp, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(b->data(), temp1, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(c->data(), d_temp2, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(d->data(), d_temp3, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(decp_data1->data(), temp4, num1 * sizeof(double), cudaMemcpyDeviceToHost);
    

    return;
}
__global__ void copyDeviceVarToDeviceMem(int *deviceMem,int *deviceMem1) {
    if (threadIdx.x == 0) {  
        *deviceMem = *de_direction_as;
        *deviceMem1 = *de_direction_ds;
    }
}




void fix_process(std::vector<int> *c,std::vector<int> *d,std::vector<double> *decp_data1,float &datatransfer, float &finddirection, float &getfcp, float &fixtime_cp, int &cpite){
    auto total_start2 = std::chrono::high_resolution_clock::now();
    int num1;
    cudaMemcpyFromSymbol(&num1, num, sizeof(int), 0, cudaMemcpyDeviceToHost);
    
    double* temp5;
    float elapsedTime;
    
    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // memory for deltaBuffer
    double* d_deltaBuffer;
    cudaMalloc(&d_deltaBuffer, num1 * sizeof(double));
    // initialization of deltaBuffer
    cudaMemset(d_deltaBuffer, 0.0, num1 * sizeof(double));
    cudaError_t cudaStatus = cudaMalloc((void**)&temp5, num1 * sizeof(double));
    
    cudaStatus = cudaMemcpy(temp5, decp_data1->data(), num1 * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed7: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    
    
    cudaStatus = cudaMemcpyToSymbol(decp_data, &temp5, sizeof(double*));
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed73: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
   
    
    
    
    

    cudaDeviceSynchronize();
    

    
    
    int* hostArray;
    cudaStatus = cudaMalloc((void**)&hostArray, num1 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed70: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
    
   
    cudaMemcpyToSymbol(de_direction_as, &hostArray, sizeof(int*));
    
    int* hostArray1;

    
    cudaStatus = cudaMalloc((void**)&hostArray1, num1 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed71: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
    cudaStatus =  cudaMemcpyToSymbol(de_direction_ds, &hostArray1, sizeof(int*));
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed72: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    datatransfer+=elapsedTime;

    dim3 blockSize(256);
    dim3 gridSize((num1 + blockSize.x - 1) / blockSize.x);
    cudaEventRecord(start, 0);

    find_direction<<<gridSize,blockSize>>>();
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // cout<<"1000次finddirection:"<<elapsedTime<<endl;
    
    finddirection+=elapsedTime;

    cudaEventRecord(start, 0);
    
    iscriticle<<<gridSize,blockSize>>>();
    
    
    
    cudaDeviceSynchronize();

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // cout<<"100cigetfcp: "<<elapsedTime;
    getfcp+=elapsedTime;
    
    
    int host_count_f_max;
    cudaStatus = cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpyToSymbol failed11: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    int host_count_f_min;
    cudaStatus = cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed12: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    while(host_count_f_max>0 or host_count_f_min>0){
        
        // cout<<host_count_f_max<<", "<<host_count_f_min<<endl;

        cpite+=1;
        dim3 blockSize1(256);
        dim3 gridSize1((host_count_f_max + blockSize1.x - 1) / blockSize1.x);
        // cudaEventRecord(start, 0);
        cudaEventRecord(start, 0);
        // fix_maxi_critical1<<<gridSize1, blockSize1>>>(0,d_deltaBuffer,id_array);
        
        // cudaDeviceSynchronize();

        dim3 blocknum(256);
        dim3 gridnum((host_count_f_min + blocknum.x - 1) / blocknum.x);
        
        
        //fix_maxi_critical1<<<gridnum, blocknum>>>(1,d_deltaBuffer,id_array);
        // cout<<"wanc"<<endl;
        cudaDeviceSynchronize();
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        
        cudaEventElapsedTime(&elapsedTime, start, stop);
        fixtime_cp+=elapsedTime;
        
        int initialValue = 0;
        cudaStatus = cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
        // if (cudaStatus != cudaSuccess) {
        //     std::cerr << "cudaMemcpyToSymbol failed4: " << cudaGetErrorString(cudaStatus) << std::endl;
        // }
        // int initialValue = 0;
        cudaStatus = cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));

        // if (cudaStatus != cudaSuccess) {
         //     std::cerr << "cudaMemcpyToSymbol failed5: " << cudaGetErrorString(cudaStatus) << std::endl;
        // }
        
        // std::cout << "Average Time Per Iteration = " << elapsedTime << " ms" << std::endl;
        cudaEventRecord(start, 0);

        iscriticle<<<gridSize, blockSize>>>();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        
        cudaEventElapsedTime(&elapsedTime, start, stop);
        getfcp+=elapsedTime;

        cudaEventRecord(start, 0);
        find_direction<<<gridSize,blockSize>>>();
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        finddirection+=elapsedTime;
        
        
        
        cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
        
        cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
        // cout<<host_count_f_max<<", "<<host_count_f_min<<endl;
        cudaDeviceSynchronize();
        
        // exit(0);
    }
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    
    cudaEventRecord(start, 0);
    find_direction<<<gridSize,blockSize>>>();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // finddirection1+=elapsedTime;
    // cudaEventElapsedTime(&wholeTime, start1, stop);
    // cout<<"["<<totalElapsedTime/wholeTime<<", "<<totalElapsedTime_fcp/wholeTime<<", "<<totalElapsedTime_fd/wholeTime<<"],"<<endl;;
    // start2 = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start, 0);
    cudaStatus = cudaMemcpy(decp_data1->data(), temp5, num1 * sizeof(double), cudaMemcpyDeviceToHost);
    

    


    

    
    // cudaMemcpy(hostArray1, de_direction_ds, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(c->data(), hostArray, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(d->data(), hostArray1, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(decp_data1->data(), temp4, num1 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    datatransfer+=elapsedTime;
    
    cudaDeviceSynchronize();
    
    
    cudaFree(temp5);
    cudaFree(hostArray);
    cudaFree(hostArray1);
    
    
    
   
    

    return;
    
}

__global__ void copyDeviceToArray(int* hostArray,int* hostArray1) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num) {
        
        hostArray[index] = de_direction_as[index];
        
        hostArray1[index] = de_direction_ds[index];
    }
    
}



void mappath1(std::vector<int> *label, std::vector<int> *direction_as, std::vector<int> *direction_ds, float &finddirection, float &mappath_path, float &datatransfer,int type=0){
    int num1;
    cudaMemcpyFromSymbol(&num1, num, sizeof(int), 0, cudaMemcpyDeviceToHost);
    
    int *un_sign_as;
    cudaMalloc((void**)&un_sign_as, sizeof(int));
    cudaMemset(un_sign_as, 0, sizeof(int));

    int *un_sign_ds;
    cudaMalloc((void**)&un_sign_ds, sizeof(int));
    cudaMemset(un_sign_ds, 0, sizeof(int));

    
    
    
    dim3 blockSize1(256);
    dim3 gridSize1((num1 + blockSize1.x - 1) / blockSize1.x);

    float elapsedTime;
    
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    int* label_temp;
    cudaError_t cudaStatus = cudaMalloc((void**)&label_temp, num1*2 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed60: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    
    
    
    int h_un_sign_as = num1;
    int h_un_sign_ds = num1;
    // int *un_sign_as = 0;
    // int *un_sign_ds = 0;
    int* hostArray;
    cudaStatus = cudaMalloc((void**)&hostArray, num1 * sizeof(int));
    
    // cudaMemcpy(decp_data1->data(), temp5, num1 * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaStatus = cudaMemcpy(hostArray,direction_as->data(), num1 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed76: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    int* hostArray1;
    cudaStatus = cudaMalloc((void**)&hostArray1, num1 * sizeof(int));
    cudaStatus = cudaMemcpy(hostArray1,direction_ds->data(),  num1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    datatransfer+=elapsedTime;

    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpyToSymbol failed78: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    if(type==0){
        
        cudaEventRecord(start, 0);
        cudaMemcpyToSymbol(de_direction_as, &hostArray, sizeof(int*));
        
        
        cudaStatus =  cudaMemcpyToSymbol(de_direction_ds, &hostArray1, sizeof(int*));
        if (cudaStatus != cudaSuccess) {
                std::cerr << "cudaMemcpyToSymbol failed72: " << cudaGetErrorString(cudaStatus) << std::endl;
        }
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        datatransfer+=elapsedTime;
        
    }
    cudaEventRecord(start, 0);
    
    initializeWithIndex<<<gridSize1, blockSize1>>>(num1,type);
    cudaDeviceSynchronize();
    
    
    while(h_un_sign_as>0 or h_un_sign_ds>0){
        
        int zero = 0;
        int zero1 = 0;

       
        cudaMemcpy(un_sign_as, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(un_sign_ds, &zero1, sizeof(int), cudaMemcpyHostToDevice);
        getlabel<<<gridSize1,blockSize1>>>(un_sign_as,un_sign_ds,0);
        
        cudaMemcpy(&h_un_sign_as, un_sign_as, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_un_sign_ds, un_sign_ds, sizeof(int), cudaMemcpyDeviceToHost);
       
        
        
    }   
        


    //     cudaDeviceSynchronize();
    // }
    cudaDeviceSynchronize();
    

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    mappath_path+=elapsedTime;

    cudaEventRecord(start, 0);
    cudaStatus = cudaMemcpy(label->data(), label_temp, num1 *2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    datatransfer+=elapsedTime;
    if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpyToSymbol failed61: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    if(type==0){
        cudaFree(label_temp);
        
    }
    
    cudaFree(hostArray1);
    cudaFree(hostArray);
    
    
    return;
};
