# MSz: An Efficient Parallel Algorithm for Correcting Morse-Smale Segmentations in Error-Bounded Lossy Compressors

MSz is designed to preserve topological features such as local minima, maxima, and integral paths with user-defined connectivity modes for 2D/3D datasets.

## Requirements
- C++17 or later
- CMake 3.18 or later
- SZ3, ZFP and ZSTD libraries (if using these compression methods)

## Installation
### 1. Clone the Repository
```bash
git clone --recursive https://github.com/YuxiaoLi1234/MSz.git
cd MSz
```
### 2. Build the Program
```bash
mkdir build
cd build
cmake ..
make
```

### 3. Usage
```bash
./MSz <path/to/data>,<width>,<height>,<depth> <relative_error_bound> <compressor_type> <connection_type> <preserve_min> <preserve_max> <preserve_integral_lines>
```

### 4. Parameters
1. path/to/data,width,height,depth
   - Description: Path to the input dataset, followed by its dimensions (width, height, depth).
   - Example: path/to/your/data.bin,256,256,128
2. relative_error_bound
   - Description: Relative error bound for processing as a floating-point value.
   - Example: 0.01
3. compressor_type
   - Description: Compression library to use. Supported values:
       - sz3
       - zfp
   - Example: sz3
4. connection_type
   - Description: Connectivity type for the dataset:
       - 0: 0: Piecewise linear connectivity (e.g., 2D case: connects only up, down, left, right, up-right, and bottom-left).
       - 1: Full connectivity (e.g., 2D: also all diagonal connections).
   - Example: 0
5. preserve_min
   - Description: Whether to preserve local minima (0 for no, 1 for yes).
   - Example: 1
6. preserve_max
   - Description: Whether to preserve local maxima (0 for no, 1 for yes).
   - Example: 0
7. preserve_integral_lines
   - Description: Whether to preserve integral lines (0 for no, 1 for yes).
       - DO NOT use this option if:
           - Both <preserve_min> and <preserve_max> are set to 1.
           - <neighbor_number> is set to 1.
   - Example: 0

### 5. Get the decompressed data with topology preservation
```bash
python3 get_fix_decp.py -e path/to/your/compressed/edits -i path/to/your/compressed/index -d path/to/your/compressed/data
example: python3 get_fix_decp.py -e edits_at.bin.zst -i index_at.bin.zst -d ../compressed_at_sz3_0.105184.sz3
```
### 6. Output
The decompressed data with topology preservation will be stored in the `result` folder. 

Additionally, a detailed report file named `result_<filename>_<compressor>_detailed.txt` will be generated. This file includes evaluation metrics such as Peak Signal-to-Noise Ratio (PSNR) and Compression Ratio (CR).





