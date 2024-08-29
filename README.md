# MSz-
MSz source code

# use the following command to compile:
nvcc -c main.cu -o main.o
g++ -O3 -g -fopenmp -c main.cpp -o main.o
g++ -fopenmp main.o main.o -lcudart -o main

# use the command:
./main filename,width,height,depth bound compressor
to run the code

filename is your input data's name without the suffix (e.g., input filename = NYX.bin, filename = NYX), width, height, depth is the dimentsion of 
your dataset, bound is your relative error bound (e.g. 1e-6), compressor is the compressor you use (e.g. sz3)

# an example: ./main NYX,512,512,512 1e-6 sz3
