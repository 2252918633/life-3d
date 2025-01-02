import random
import sys

def generate_file(filename, N):
    total_bytes = N ** 3  
    
    with open(filename, "wb") as file:
        for _ in range(total_bytes):
            byte = 1 if random.random() < 0.23 else 0
            file.write(byte.to_bytes(1, byteorder="little"))  

if len(sys.argv) < 2:
    print("usage: python3 RandGen.py N")
    sys.exit(1)
N = int(sys.argv[1])
output_file = "data/data.in"
generate_file(output_file, N)

