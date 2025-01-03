/*-----------------------------------------------
 * 请在此处填写你的个人信息
 * 学号:SA24218163
 * 姓名:李金浩
 * 邮箱:lijinhao@mail.ustc.edu.cn
 ------------------------------------------------*/

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <cuda_runtime.h>

#define AT(x, y, z) universe[(x) * N * N + (y) * N + z]
#define AT_NEXT(x, y, z) next_universe[(x) * N * N + (y) * N + z]

using namespace std;
__global__ void life3d_run_kernel(int N, char *universe, char *next_universe) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // 确保计算在宇宙范围内
    if (x >= N || y >= N || z >= N) return;

    int alive = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = (x + dx + N) % N;
                int ny = (y + dy + N) % N;
                int nz = (z + dz + N) % N;
                alive += AT(nx, ny, nz);
            }
        }
    }

    char current_state = AT(x, y, z);
    char next_state = current_state;
    if (current_state && (alive < 5 || alive > 7)) {
        next_state = 0;
    } else if (!current_state && alive == 6) {
        next_state = 1;
    }

    AT_NEXT(x, y, z) = next_state;
}

// 存活细胞数
int population(int N, char *universe) {
    int result = 0;
    for (int i = 0; i < N * N * N; i++)
        result += universe[i];
    return result;
}

// 打印世界状态
void print_universe(int N, char *universe) {
    // 仅在N较小(<= 32)时用于Debug
    if (N > 32)
        return;
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            for (int z = 0; z < N; z++) {
                if (AT(x, y, z))
                    cout << "O ";
                else
                    cout << "* ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << "population: " << population(N, universe) << endl;
}

// 使用 CUDA 核函数计算下一个宇宙状态
void life3d_run_cuda(int N, char *universe, int T) {
    char *d_universe, *d_next_universe;
    size_t size = N * N * N * sizeof(char);

    // 分配设备内存
    cudaMalloc(&d_universe, size);
    cudaMalloc(&d_next_universe, size);

    // 将宇宙状态从主机复制到设备
    cudaMemcpy(d_universe, universe, size, cudaMemcpyHostToDevice);

    // 设置网格和线程块大小
    dim3 threadsPerBlock(8, 8, 8); // 修订线程块大小以适应更大的 N
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // 临时数组用于调试
    char *h_debug_universe = (char *)malloc(N * N * N * sizeof(char));

    // 迭代 T 次
    for (int t = 0; t < T; t++) {
        life3d_run_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, d_universe, d_next_universe);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }

        // 等待所有 CUDA 操作完成
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA sync error: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }

        // 使用设备内存交换
        cudaMemcpy(d_universe, d_next_universe, size, cudaMemcpyDeviceToDevice);

        // 若 N <= 32，则输出前几个细胞状态和存活细胞数，用于调试
        if (N <= 32 && t < 5) { // 只检查前5次迭代（视需要调整）
            cudaMemcpy(h_debug_universe, d_universe, size, cudaMemcpyDeviceToHost);

            cout << "Iteration " << t + 1 << " - First few cells (N=" << N << "): ";
            for (int i = 0; i < std::min(10, N * N * N); ++i) {
                cout << (int)h_debug_universe[i] << " ";
            }
            cout << endl;
            cout << "Iteration " << t + 1 << " - Population: " << population(N, h_debug_universe) << endl;
        }
    }

    // 从设备将计算好的宇宙状态复制回主机
    cudaMemcpy(universe, d_universe, size, cudaMemcpyDeviceToHost);

    // 释放所有内存
    free(h_debug_universe);
    cudaFree(d_universe);
    cudaFree(d_next_universe);
}

// 读取输入文件
void read_file(char *input_file, char *buffer, int N) {
    ifstream file(input_file, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        cout << "Error: Could not open file " << input_file << std::endl;
        exit(1);
    }
    std::streamsize file_size = file.tellg();
    if (file_size != N * N * N) {
        cout << "Error: File size does not match the expected size " << N * N * N << std::endl;
        exit(1);
    }
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(buffer), file_size)) {
        std::cerr << "Error: Could not read file " << input_file << std::endl;
        exit(1);
    }
    file.close();

    // 调试信息：检查输入文件的前几个细胞状态
    cout << "First few cells in input file (N=" << N << "): ";
    for (int i = 0; i < std::min(10, N * N * N); ++i) {
        cout << (int)buffer[i] << " ";
    }
    cout << endl;
}

// 写入输出文件
void write_file(char *output_file, char *buffer, int N) {
    ofstream file(output_file, std::ios::binary | std::ios::trunc);
    if (!file) {
        cout << "Error: Could not open file " << output_file << endl;
        exit(1);
    }
    file.write(reinterpret_cast<char*>(buffer), N * N * N);
    file.close();
}

int main(int argc, char **argv) {
    // 命令行参数检查
    if (argc < 5) {
        cout << "usage: ./life3d N T input output" << endl;
        return 1;
    }
    int N = std::stoi(argv[1]);
    int T = std::stoi(argv[2]);
    char *input_file = argv[3];
    char *output_file = argv[4];

    char *universe = (char *)malloc(N * N * N * sizeof(char));
    read_file(input_file, universe, N);

    // 执行 CUDA 核函数模拟
    int start_pop = population(N, universe);
    auto start_time = std::chrono::high_resolution_clock::now();
    life3d_run_cuda(N, universe, T);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    int final_pop = population(N, universe);
    write_file(output_file, universe, N);

    // 输出结果
    cout << "start population: " << start_pop << endl;
    cout << "final population: " << final_pop << endl;
    double time = duration.count();
    cout << "time: " << time << "s" << endl;
    if (time > 0)
        cout << "cell per sec: " << T * N * N * N / time << endl;

    // 打印最终状态（仅在N较小且T较小时）
    if (N <= 16) {
        print_universe(N, universe);
    }

    free(universe);
    return 0;
}