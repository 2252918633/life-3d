/*-----------------------------------------------
 * 请在此处填写你的个人信息
 * 学号:SA24218163
 * 姓名:李金浩
 * 邮箱:
 ------------------------------------------------*/
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#define AT(x, y, z) universe[(x) * N * N + (y) * N + z]

using std::cin, std::cout, std::endl;
using std::ifstream, std::ofstream;

// 存活细胞数
__global__ void population(int N, char *universe, int *result)
{
    __shared__ int partialSum[256];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    int alive = 0;
    if (index < N * N * N)
        alive = universe[index];

    __syncthreads();

    // Reducing in shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            partialSum[tid] += partialSum[tid + offset];
        }
        __syncthreads();
    }

    // Write the result of this block to global memory
    if (tid == 0)
    {
        atomicAdd(result, partialSum[0]);
    }
}

// 3D 生命游戏核函数
__global__ void life3d_run(int N, char *universe, int T)
{
    extern __shared__ char shared_universe[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // 检查是否超出边界
    if (x >= N || y >= N || z >= N)
        return;

    int idx = x * N * N + y * N + z;

    // 将数据从全局内存复制到共享内存
    shared_universe[threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x] = universe[idx];
    __syncthreads();

    // 核心计算
    for (int t = 0; t < T; t++)
    {
        int alive = 0;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++)
                {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;
                    int nx = (x + dx + N) % N;
                    int ny = (y + dy + N) % N;
                    int nz = (z + dz + N) % N;

                    int nidx = nx * N * N + ny * N + nz;
                    alive += shared_universe[nidx];
                }

        char new_state = 0;
        if (shared_universe[idx] && (alive < 5 || alive > 7))
            new_state = 0;
        else if (!shared_universe[idx] && alive == 6)
            new_state = 1;
        else
            new_state = shared_universe[idx];

        // 同步线程，确保所有线程完成计算
        __syncthreads();

        // 更新共享内存中的状态
        shared_universe[threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x] = new_state;

        // 同步线程，确保所有线程完成更新
        __syncthreads();
    }

    // 将最终状态从共享内存复制回全局内存
    universe[idx] = shared_universe[threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x];
}

// 读取输入文件
void read_file(char *input_file, char *buffer)
{
    ifstream file(input_file, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        cout << "Error: Could not open file " << input_file << std::endl;
        exit(1);
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer, file_size))
    {
        std::cerr << "Error: Could not read file " << input_file << std::endl;
        exit(1);
    }
    file.close();
}

// 写入输出文件
void write_file(char *output_file, char *buffer, int N)
{
    ofstream file(output_file, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        cout << "Error: Could not open file " << output_file << std::endl;
        exit(1);
    }
    file.write(buffer, N * N * N);
    file.close();
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        cout << "usage: ./life3d N T input output" << endl;
        return 1;
    }

    int N = std::stoi(argv[1]);
    int T = std::stoi(argv[2]);
    char *input_file = argv[3];
    char *output_file = argv[4];

    char *host_universe = (char *)malloc(N * N * N);
    read_file(input_file, host_universe);

    char *device_universe;
    cudaMalloc(&device_universe, N * N * N);

    int *host_population, *device_population;
    host_population = (int *)malloc(sizeof(int));
    cudaMalloc(&device_population, sizeof(int));
    cudaMemset(device_population, 0, sizeof(int));

    cudaMemcpy(device_universe, host_universe, N * N * N, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y, (N + blockSize.z - 1) / blockSize.z);

    int start_pop;
    population<<<gridSize, blockSize>>>(N, device_universe, device_population);
    cudaMemcpy(&start_pop, device_population, sizeof(int), cudaMemcpyDeviceToHost);

    auto start_time = std::chrono::high_resolution_clock::now();
    life3d_run<<<gridSize, blockSize, blockSize.x * blockSize.y * blockSize.z>>>(N, device_universe, T);
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    int final_pop;
    cudaMemset(device_population, 0, sizeof(int));
    population<<<gridSize, blockSize>>>(N, device_universe, device_population);
    cudaMemcpy(&final_pop, device_population, sizeof(int), cudaMemcpyDeviceToHost);

    char *host_result = (char *)malloc(N * N * N);
    cudaMemcpy(host_result, device_universe, N * N * N, cudaMemcpyDeviceToHost);
    write_file(output_file, host_result, N);

    cout << "start population: " << start_pop << endl;
    cout << "final population: " << final_pop << endl;
    double time = duration.count();
    cout << "time: " << time << "s" << endl;
    cout << "cell per sec: " << T / time * N * N * N << endl;

    free(host_universe);
    free(host_result);
    free(host_population);
    cudaFree(device_universe);
    cudaFree(device_population);

    return 0;
}
