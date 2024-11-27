import numpy as np
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="A script to process and execute commands.")

    parser.add_argument("-e", "--edits", type=str, required=True,
                        help="path/to/compressed/edits.")
    parser.add_argument("-i", "--index", type=str, 
                        help="path/to/compressed/index.")
    parser.add_argument("-d", "--cp", type=str, 
                        help="path/to/compressed/data.")

    args = parser.parse_args()
    command = ["zstd", "-d", args.edits,"-o", "edits.bin"]
    print("Executing command:", " ".join(command))
    try:
        result = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error occurred:\n", e.stderr)

    
    command = ["zstd", "-d",  args.index, "-o", "index.bin"]
    print("Executing command:", " ".join(command))
    try:
        result = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error occurred:\n", e.stderr)

    print(args.edits, args.index, args.cp)
    command = ["sz3", "-d", "-z", args.cp, "-o", "decp.bin"]
    print("Executing command:", " ".join(command))
    try:
        result = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error occurred:\n", e.stderr)

    edits = np.fromfile("edits.bin", dtype=np.float64)
    index = np.fromfile("index.bin", dtype=np.int32)
    decp = np.fromfile("decp.bin", dtype=np.float64)
    diffs = []
    for i in range(len(edits)):
        if(i==0):
            index_ = index[i]
            diffs.append(index_)
        else:
            index_ = index[i] + diffs[-1]
            diffs.append(index_)
        edit_ = edits[i]
        decp[index_] += edit_

    decp.tofile("./result/fixed_decompressed_data.bin")

    command = ["rm", "index.bin"]
    print("Executing command:", " ".join(command))
    try:
        result = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error occurred:\n", e.stderr)
    
    command = ["rm", "edits.bin"]
    print("Executing command:", " ".join(command))
    try:
        result = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error occurred:\n", e.stderr)
if __name__ == "__main__":
    main()
