#include "dist_objects.h"

#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>

__global__ void _construct_rows_per_neighbour(int *row_ptr_d, int *col_indices_d, int *is_nnz_row_d, int *expanded_rows_d, int matrix_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        if (row_ptr_d[i] != row_ptr_d[i+1]){

            is_nnz_row_d[i] = 1;
            expanded_rows_d[i] = i;
        }
        
    }
}

void Distributed_matrix::construct_rows_per_neighbour()
{

    nnz_rows_per_neighbour = new int[number_of_neighbours];
    rows_per_neighbour_d = new int*[number_of_neighbours];  
    for(int i = 0; i < number_of_neighbours; i++){
        int *is_nnz_row_d;
        int *expanded_rows_d;

        int cols = counts[neighbours[i]];
        int rows = rows_this_rank;

        // malloc
        cudaErrchk(cudaMalloc(&is_nnz_row_d, sizeof(int) * rows));
        cudaErrchk(cudaMalloc(&expanded_rows_d, sizeof(int) * rows));

        // memset to zero
        cudaErrchk(cudaMemset(is_nnz_row_d, 0, sizeof(int) * rows));
        cudaErrchk(cudaMemset(expanded_rows_d, 0, sizeof(int) * rows));

        // kernel
        int threads = 1024;
        int blocks = (rows + threads - 1) / threads;   
        _construct_rows_per_neighbour<<<blocks, threads>>>(row_ptr_d[i], col_indices_d[i], is_nnz_row_d, expanded_rows_d, rows);

        // sum
        int nnz = thrust::reduce(thrust::device, is_nnz_row_d, is_nnz_row_d + rows, 0, thrust::plus<int>());
        nnz_rows_per_neighbour[i] = nnz;

        // malloc
        cudaErrchk(cudaMalloc(&rows_per_neighbour_d[i], sizeof(int) * nnz));

        // copyif into rows_per_neighbour_d
        thrust::copy_if(thrust::device, expanded_rows_d, expanded_rows_d + rows, rows_per_neighbour_d[i], thrust::identity<int>());
        // thrust::copy_if(thrust::device, is_tunnel_indices, is_tunnel_indices + gpubuf.N_, tunnel_indices, is_not_zero());


        cudaErrchk(cudaFree(is_nnz_row_d));
        cudaErrchk(cudaFree(expanded_rows_d));
    }

    send_buffer_h = new double*[number_of_neighbours];
    send_buffer_d = new double*[number_of_neighbours];

    for(int k = 1; k < number_of_neighbours; k++){
        cudaErrchk(cudaMallocHost(&send_buffer_h[k], nnz_rows_per_neighbour[k]*sizeof(double)));
        cudaErrchk(cudaMalloc(&send_buffer_d[k], nnz_rows_per_neighbour[k]*sizeof(double)));
    }

}


__global__ void _construct_cols_per_neighbour(
    int *row_ptr_d,
    int *col_indices_d,
    int *is_nnz_col_d,
    int *expanded_cols_d,
    int matrix_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // todo coo
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        for(int j = row_ptr_d[i]; j < row_ptr_d[i+1]; j++){
            int col = col_indices_d[j];
            int one = 1;
            atomicMax(&is_nnz_col_d[col], one);
            atomicMax(&expanded_cols_d[col], col);
        }
    }


}

void Distributed_matrix::construct_cols_per_neighbour()
{

    nnz_cols_per_neighbour = new int[number_of_neighbours];
    cols_per_neighbour_d = new int*[number_of_neighbours];

    for(int k = 0; k < number_of_neighbours; k++){
        int *is_nnz_col_d;
        int *expanded_cols_d;

        int cols = counts[neighbours[k]];
        int rows = rows_this_rank;

        // malloc
        cudaErrchk(cudaMalloc(&is_nnz_col_d, sizeof(int) * cols));
        cudaErrchk(cudaMalloc(&expanded_cols_d, sizeof(int) * cols));

        // memset to zero
        cudaErrchk(cudaMemset(is_nnz_col_d, 0, sizeof(int) * cols));
        cudaErrchk(cudaMemset(expanded_cols_d, 0, sizeof(int) * cols));

        // kernel
        int threads = 1024;
        int blocks = (rows + threads - 1) / threads;
        _construct_cols_per_neighbour<<<blocks, threads>>>(row_ptr_d[k], col_indices_d[k], is_nnz_col_d, expanded_cols_d, rows);

        // sum
        int nnz = thrust::reduce(thrust::device, is_nnz_col_d, is_nnz_col_d + cols, 0, thrust::plus<int>());
        nnz_cols_per_neighbour[k] = nnz;

        // malloc
        cudaErrchk(cudaMalloc(&cols_per_neighbour_d[k], sizeof(int) * nnz));

        // copyif into cols_per_neighbour_d
        thrust::copy_if(thrust::device, expanded_cols_d, expanded_cols_d + cols, cols_per_neighbour_d[k], thrust::identity<int>());

        cudaErrchk(cudaFree(is_nnz_col_d));
        cudaErrchk(cudaFree(expanded_cols_d));
    }

    recv_buffer_h = new double*[number_of_neighbours];
    recv_buffer_d = new double*[number_of_neighbours];

    for(int k = 1; k < number_of_neighbours; k++){
        cudaErrchk(cudaMallocHost(&recv_buffer_h[k], nnz_cols_per_neighbour[k]*sizeof(double)));
        cudaErrchk(cudaMalloc(&recv_buffer_d[k], nnz_cols_per_neighbour[k]*sizeof(double)));
    }
}