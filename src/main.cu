#include <vector>
#include <fstream>
#include <iostream>

#include <cmath>
#include <ctime>

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <mpi.h>

#define DEBUG
#define PRINT_TIME
#define HEAD_LENGTH 5

int rank, size;
int window_size, window_size_mm, min_dist;
int global_length;

// функция, которая ожидает завершения вызова ядра
void CudaSyncAndCheckErrors(void){
    // синхронизация потока выполнения CUDA
    cudaStreamSynchronize(0);
    
    // ловим ошибки и выходим, если они произошли
    cudaError_t x = cudaGetLastError();
    if ((x) != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(x)); fclose(stdout); exit(1); 
    }
}

// Определим два типа данных:
// временной ряд на CPU и GPU соответственно
typedef thrust::host_vector<double> Timeseries;
typedef thrust::device_vector<double> Timeseries_GPU;

// Функция, которая считывает временной ряд с файла специального формата
void ReadTimeseries(Timeseries &ts, const char* filename){
    std::ifstream fin(filename, std::ios::binary);

    // длина временного ряда
    unsigned int length;
    fin.read((char*)&length, sizeof(int));

    // сам временной ряд
    ts.resize(length);
    fin.read((char*)&ts[0], sizeof(double) * length);
    fin.close();

    MPI_File mpi_file;
    MPI_Status mpi_status;

    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);

    if(rank == 0)
        MPI_File_read_at(mpi_file, 0, &length, 1, MPI_UNSIGNED, &mpi_status);
    
    MPI_Bcast(&length, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    int part_length = length / size + (rank == size - 1? length % size : 0);
    int delta = (rank == size - 1? 0 : window_size - 1);

    MPI_Reduce(&part_length, &global_length, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);

    int offset = 4 + sizeof(double) * rank * part_length;

    ts.resize(part_length);
    std::cout << "rank=" << rank << ", part_length=" << part_length << ", delta=" << delta << std::endl;
    MPI_File_read_at(mpi_file, offset, &ts[0], (part_length + delta), MPI_DOUBLE, &mpi_status);

    MPI_File_close(&mpi_file);
}

// Вводим две структуры для оконных статистик
// на CPU
struct RollingStat{
    Timeseries mean;
    Timeseries var;
};

// на GPU
struct RollingStat_GPU{
    Timeseries_GPU mean;
    Timeseries_GPU var;
};

// Функция, которая считает оконные статистики 
void ComputeRollingStat_CPU(int window_size, Timeseries &ts, RollingStat &rolling){
    int rolling_length = ts.size() - (window_size - 1);

    // выделяем память
    rolling.mean.resize(rolling_length);
    rolling.var.resize(rolling_length);

    for(int i = 0; i < rolling_length; ++i){
        // считаем сначала среднее
        double loc = 0.0;
        for(int k = 0; k < window_size; ++k)
            loc += ts[i+k];
        rolling.mean[i] = loc / window_size;

        // теперь считаем дисперсию
        loc = 0;
        for(int k = 0; k < window_size; ++k)
            loc += pow(ts[i+k] - rolling.mean[i], 2);
        rolling.var[i] = loc / (window_size - 1);
    }
    return;
}

// CUDA-ядро для вычисления оконных статистик
__global__ void __kernel_ComputeRollingStat_GPU(int window_size, int N, double *ts, double *mean, double *var){
    // получаем номер нити в сетке
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    // если мы не вылезли за границы, то
    if(idx < N){
        // считаем среднее
        double loc_mean = 0.0;
        for(int k = 0; k < window_size; ++k)
            loc_mean += ts[idx + k];
        mean[idx] = loc_mean / window_size;
        
        // считаем дисперсию
        double loc = 0.0;
        for(int k = 0; k < window_size; ++k)
            loc += pow(ts[idx + k] - mean[idx], 2);
        var[idx] = loc / (window_size - 1);
    }
}

// Функция, которая вычисляет оконные статистики на GPU
void ComputeRollingStat_GPU(int window_size, Timeseries_GPU &ts, RollingStat_GPU &rolling){
    int rolling_length = ts.size() - (window_size - 1);
    rolling.mean.resize(rolling_length);
    rolling.var.resize(rolling_length);

    // задаем параметры сетки
    // размер блока 256 -- дань традициям
    dim3 block_spec(256); 
    dim3 grid_spec(rolling_length / 256 + (rolling_length % 256? 1 : 0));

    // вызываем CUDA-ядро
    __kernel_ComputeRollingStat_GPU<<< grid_spec, block_spec >>> (window_size, rolling_length, 
        thrust::raw_pointer_cast(&ts[0]),
        thrust::raw_pointer_cast(&rolling.mean[0]),
        thrust::raw_pointer_cast(&rolling.var[0]));
    
    // ждем завершения программы
    CudaSyncAndCheckErrors();

    return;
}

// Для вычисления квантили использовалась библиотека Thrust
// функция, которая вычисляет квантиль на CPU
double ComputeQuantile_CPU(double alpha, Timeseries &ts){
    Timeseries loc(ts);

    // номер искомой порядковой статистики
    int k_alpha = (int)(alpha * ts.size());

    // вычисление вариационного ряда
    thrust::sort(thrust::host, loc.begin(), loc.end());
    return loc[k_alpha];
}

// функция, которая вычисляет квантиль на GPU
double ComputeQuantile_GPU(double alpha, Timeseries &ts){
    Timeseries_GPU loc(ts);

    // номер искомой порядковой статистики
    int k_alpha = (int)(alpha * ts.size());

    // вычисление вариационного ряда
    thrust::sort(thrust::device, loc.begin(), loc.end());
    return loc[k_alpha];
}

// функция, которая посчитает среднее для окна
inline double ComputeMean_CPU(double *ts, int window_size){
    double loc = 0;
    for(int i = 0; i < window_size; ++i)
        loc += ts[i];
    return loc / window_size;
}

// функция, вычисляющая точки начала движения
void DetectMovement_CPU(int window_size, int min_dist, double q_alpha,
    Timeseries &ts, Timeseries &movement, std::vector<double> &points
){
    // инициализируем параметры
    double prev_mean = ComputeMean_CPU(&ts[0], window_size);

    bool prev_movement = false;
    int prev_point = -2 * min_dist;

    // начинаем со второго окна и сравниваем его с первым
    for(int left = window_size; left < ts.size(); left += window_size){
        double cur_mean = ComputeMean_CPU(&ts[left], window_size);

        // узнаем, значимо ли отличие средних в смежных окнах
        bool cur_movement = (fabs(cur_mean - prev_mean) > q_alpha);

        // если до этого движения не было, а сейчас есть и расстояние
        // от предыдущей точки начала движения больше, чем выбранный
        // порог фильтрации, то добавляем точку в список
        if(cur_movement && (!prev_movement) && (left - prev_point > min_dist)){
            points.push_back(left);
            prev_point = left;
        }

        // переходим к следующему шагу
        prev_movement = cur_movement;
        prev_mean = cur_mean;
    }
}

// основная программа
int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // засекаем глобальное время
    double global_start, global_stop;
    global_start = MPI_Wtime();

    // приветственная фраза
    if(rank == 0)
        std::cout << "Hello, MPI and CUDA!" << std::endl;

    // инициализация параметров
    // размер окна для вычисления оконной дисперсии
    window_size = 32;
    // размер окна для метода сравнения средних
    window_size_mm = 64;
    // минимальное расстояние между точками начала движения
    min_dist = window_size_mm * 4;

    // для замера времени работы отдельных кусков программы
    double start, stop;

    // Считываем временной ряд миограммы
    // в качестве аргумента можно подать путь к файлу с данными
    Timeseries local_ts;
    start = MPI_Wtime();
    ReadTimeseries(local_ts,(argc > 1? argv[1] : "data/miog.bin"));
    stop = MPI_Wtime();
    #ifdef PRINT_TIME
    if(rank == 0)
        std::cerr << "ReadTimeseries time: " << (stop - start) << std::endl;
    #endif

    // копируем считанный ряд на GPU
    start = MPI_Wtime();
    Timeseries_GPU local_ts_gpu(local_ts);
    stop = MPI_Wtime();
    #ifdef PRINT_TIME
    if(rank == 0)
        std::cerr << "HtoD time: " << (stop - start) << std::endl;
    #endif

    // Считаем оконные статистики на GPU
    RollingStat_GPU rolling_gpu;
    start = MPI_Wtime();
    ComputeRollingStat_GPU(window_size, local_ts_gpu, rolling_gpu);
    stop = MPI_Wtime();
    #ifdef PRINT_TIME
    if(rank == 0)
        std::cerr << "ComputeRollingStat_GPU time: " << (stop - start) << std::endl;
    #endif

    // копируем результаты на CPU
    start = MPI_Wtime();
    Timeseries local_var(rolling_gpu.var);
    stop = MPI_Wtime();
    #ifdef PRINT_TIME
    if(rank == 0)
        std::cerr << "DtoH time: " << (stop - start) << std::endl;
    #endif

    ///////////////////////////////////// TODO /////////////////////////////////////
    // собираем все данные на нулевой узел
    Timeseries var;
    if(rank == 0){
        std::cout << "global_length=" << global_length << std::endl;
        var.resize(global_length);
    }

    std::vector<int> displs;
    std::vector<int> recvcounts;
    
    if(rank == 0){
        for(int i = 0; i < size; ++i){
            int o = i * (global_length / size);
            int l = (global_length / size) + (i == size - 1? global_length % size : 0);

            displs.push_back(o);
            recvcounts.push_back(l);
            std::cout << "rank=" << i << ", offset=" << o << ", will send=" << l << std::endl;
        }
    }

    MPI_Gatherv(&local_var[0], local_var.size(), MPI_DOUBLE, &var[0], &recvcounts[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank == 0){
        // Вычисляем квантиль на GPU
        start = MPI_Wtime();
        double q_gpu = ComputeQuantile_GPU(0.7, var);
        stop = MPI_Wtime();
        #ifdef PRINT_TIME
        std::cerr << "ComputeQuantile_GPU time: " << (stop - start) << std::endl;
        #endif

        std::cout << "q_alpha=" << q_gpu << std::endl;

        // ищем точки начала движения
        Timeseries movement;
        std::vector<double> points;

        start = MPI_Wtime();
        DetectMovement_CPU(window_size_mm, min_dist, q_gpu, var, movement, points);
        stop = MPI_Wtime();
        #ifdef PRINT_TIME
        std::cerr << "DetectMovement_CPU time: " << (stop - start) << std::endl;
        #endif

        // сохраняем найденные точки в текстовый файл
        // можно передать название файла вторым аргументом командной строки
        start = MPI_Wtime();
        std::ofstream fout((argc > 2? argv[2] : "result.txt"));
        for(int i = 0; i < points.size(); ++i)
            fout << points[i] << " ";
        fout.close();
        stop = MPI_Wtime();
        #ifdef PRINT_TIME
        std::cerr << "SavingResults time: " << (stop - start) << std::endl;
        #endif

        std::cout << "Found points: " << points.size() << std::endl;
        // печатаем время работы программы
        global_stop = MPI_Wtime();
        std::cerr << "Total time: " << (global_stop - global_start) << std::endl;
    }

    MPI_Finalize();
    return 0;
}