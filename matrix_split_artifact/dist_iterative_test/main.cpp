#include <iostream>
#include <string>
#include "utils.h"
#include <mpi.h>
#include <cuda_runtime.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "utils_gpu.h"
#include <cublas_v2.h>
#include "../dist_iterative/dist_conjugate_gradient.h"
#include "../dist_iterative/dist_spmv.h"
#include <pthread.h>

template <void (*distributed_spmv)(Distributed_matrix&, Distributed_vector&, cusparseDnVecDescr_t&, cudaStream_t&, cusparseHandle_t&)>
void get_solution(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *diag_inv_d,
    double *time_taken,
    int measurements)
{
    MPI_Barrier(comm);

    

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(rank == 0){
        std::cout << "PCG test starts tot" << std::endl;
    }
    // prepare for allgatherv
    int counts[size];
    int displacements[size];
    int rows_per_rank = matrix_size / size;    
    split_matrix(matrix_size, size, counts, displacements);

    int row_start_index = displacements[rank];
    rows_per_rank = counts[rank];

    int *row_indptr_local_h = new int[rows_per_rank+1];
    double *r_local_h = new double[rows_per_rank];
    #pragma omp parallel for
    for (int i = 0; i < rows_per_rank+1; ++i) {
        row_indptr_local_h[i] = row_indptr_h[i+row_start_index] - row_indptr_h[row_start_index];
    }
    #pragma omp parallel for
    for (int i = 0; i < rows_per_rank; ++i) {
        r_local_h[i] = r_h[i+row_start_index];
    }
    int nnz_local = row_indptr_local_h[rows_per_rank];
    int *col_indices_local_h = new int[nnz_local];
    double *data_local_h = new double[nnz_local];

    #pragma omp parallel for
    for (int i = 0; i < nnz_local; ++i) {
        col_indices_local_h[i] = col_indices_h[i+row_indptr_h[row_start_index]];
        data_local_h[i] = data_h[i+row_indptr_h[row_start_index]];
    }

    // create distributed matrix
    Distributed_matrix A_distributed(
        matrix_size,
        nnz_local,
        counts,
        displacements,
        col_indices_local_h,
        row_indptr_local_h,
        data_local_h,
        comm
    );
    Distributed_vector p_distributed(
        matrix_size,
        counts,
        displacements,
        A_distributed.number_of_neighbours,
        A_distributed.neighbours,
        comm
    );
    double *r_local_d;
    double *x_local_d;

    for(int i = 0; i < measurements; i++){
        cudaMalloc(&r_local_d, rows_per_rank * sizeof(double));
        cudaMalloc(&x_local_d, rows_per_rank * sizeof(double));
        cudaMemcpy(r_local_d, r_local_h, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(x_local_d, starting_guess_h + row_start_index,
            rows_per_rank * sizeof(double), cudaMemcpyHostToDevice);


        double *diag_inv_local_d = diag_inv_d + row_start_index;

        cudaDeviceSynchronize();
        MPI_Barrier(comm);
        time_taken[i] = -MPI_Wtime();
        iterative_solver::conjugate_gradient_jacobi<dspmv::gpu_packing>(
            A_distributed,
            p_distributed,
            r_local_d,
            x_local_d,
            diag_inv_local_d,
            relative_tolerance,
            max_iterations,
            comm);
        cudaDeviceSynchronize();
        MPI_Barrier(comm);
        time_taken[i] += MPI_Wtime();
        if(rank == 0){
            std::cout << "time_taken["<<i<<"] " << time_taken[i] << std::endl;
        }
    }
    //copy solution to host
    cudaErrchk(cudaMemcpy(reference_solution + row_start_index,
        x_local_d, rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));
    //MPI allgather in place
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, reference_solution, counts, displacements, MPI_DOUBLE, comm);

    delete[] row_indptr_local_h;
    delete[] r_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    cudaFree(r_local_d);
    cudaFree(x_local_d);

    MPI_Barrier(comm);
}

template 
void get_solution<dspmv::gpu_packing>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *diagonal_d,
    double *time_taken, int measurements);


template <void (*distributed_spmv_split_sparse)
    (Distributed_subblock_sparse &,
    Distributed_matrix &,    
    double *,
    double *,
    cusparseDnVecDescr_t &,
    Distributed_vector &,
    double *,
    cusparseDnVecDescr_t &,
    cusparseDnVecDescr_t &,
    double *,
    cudaStream_t &,
    cusparseHandle_t &)>
void test_preconditioned_split_sparse(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *A_subblock_data_h,
    int *A_subblock_col_indices_h,
    int *A_subblock_row_ptr_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *time_taken, int measurements)
{
    MPI_Barrier(comm);


    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(rank == 0){
        std::cout << "PCG test starts split" << std::endl;
    }
    
    // prepare for allgatherv
    int counts[size];
    int displacements[size];
    int rows_per_rank = matrix_size / size;    
    split_matrix(matrix_size, size, counts, displacements);

    int row_start_index = displacements[rank];
    rows_per_rank = counts[rank];

    int *row_indptr_local_h = new int[rows_per_rank+1];
    double *r_local_h = new double[rows_per_rank];
    #pragma omp parallel for
    for (int i = 0; i < rows_per_rank+1; ++i) {
        row_indptr_local_h[i] = row_indptr_h[i+row_start_index] - row_indptr_h[row_start_index];
    }
    #pragma omp parallel
    for (int i = 0; i < rows_per_rank; ++i) {
        r_local_h[i] = r_h[i+row_start_index];
    }
    int nnz_local = row_indptr_local_h[rows_per_rank];
    int *col_indices_local_h = new int[nnz_local];
    double *data_local_h = new double[nnz_local];
    #pragma omp parallel for
    for (int i = 0; i < nnz_local; ++i) {
        col_indices_local_h[i] = col_indices_h[i+row_indptr_h[row_start_index]];
        data_local_h[i] = data_h[i+row_indptr_h[row_start_index]];
    }

    // create distributed matrix
    Distributed_matrix A_distributed(
        matrix_size,
        nnz_local,
        counts,
        displacements,
        col_indices_local_h,
        row_indptr_local_h,
        data_local_h,
        comm
    );
    Distributed_vector p_distributed(
        matrix_size,
        counts,
        displacements,
        A_distributed.number_of_neighbours,
        A_distributed.neighbours,
        comm
    );
    double *r_local_d;
    double *x_local_d;
    cudaMalloc(&r_local_d, rows_per_rank * sizeof(double));
    cudaMalloc(&x_local_d, rows_per_rank * sizeof(double));

    int *dense_subblock_indices_d;
    cudaMalloc(&dense_subblock_indices_d, subblock_size * sizeof(int));
    cudaMemcpy(dense_subblock_indices_d, subblock_indices_h, subblock_size * sizeof(int), cudaMemcpyHostToDevice);
    MPI_Barrier(comm);
    cudaDeviceSynchronize();

    int *count_subblock = new int[size];
    int *displ_subblock = new int[size];
    double *diag_local_h = new double[rows_per_rank];


    for(int i = 0; i < size; i++){
        count_subblock[i] = 0;
    }
    #pragma omp parallel for
    for(int j = 0; j < size; j++){
        int tmp = 0;
        for(int i = 0; i < subblock_size; i++){
            if( subblock_indices_h[i] >= displacements[j] && subblock_indices_h[i] < displacements[j] + counts[j]){
                tmp++;
            }
        }
        count_subblock[j] = tmp;
    }
    displ_subblock[0] = 0;
    for(int i = 1; i < size; i++){
        displ_subblock[i] = displ_subblock[i-1] + count_subblock[i-1];
    }

    
    int *subblock_indices_local_h = new int[count_subblock[rank]];
    #pragma omp parallel for
    for(int i = 0; i < count_subblock[rank]; i++){
        subblock_indices_local_h[i] = subblock_indices_h[displ_subblock[rank] + i] - displacements[rank];
    }

    int *subblock_indices_local_d;
    cudaMalloc(&subblock_indices_local_d, count_subblock[rank] * sizeof(int));
    cudaMemcpy(subblock_indices_local_d, subblock_indices_local_h, count_subblock[rank] * sizeof(int), cudaMemcpyHostToDevice);



    int *A_subblock_row_ptr_local_h = new int[count_subblock[rank]+1];
    #pragma omp parallel for
    for (int i = 0; i < count_subblock[rank]+1; ++i) {
        A_subblock_row_ptr_local_h[i] = A_subblock_row_ptr_h[i+displ_subblock[rank]] - A_subblock_row_ptr_h[displ_subblock[rank]];
    }

    int nnz_local_subblock = A_subblock_row_ptr_local_h[count_subblock[rank]];
    double *A_subblock_data_local_h = new double[nnz_local_subblock];
    int *A_subblock_col_indices_local_h = new int[nnz_local_subblock];
    #pragma omp parallel for
    for (int i = 0; i < nnz_local_subblock; ++i) {
        A_subblock_col_indices_local_h[i] = A_subblock_col_indices_h[i+A_subblock_row_ptr_h[displ_subblock[rank]]];
        A_subblock_data_local_h[i] = A_subblock_data_h[i+A_subblock_row_ptr_h[displ_subblock[rank]]];
    }

    double *A_subblock_data_local_d;
    int *A_subblock_col_indices_local_d;
    int *A_subblock_row_ptr_local_d;
    cudaMalloc(&A_subblock_data_local_d, nnz_local_subblock * sizeof(double));
    cudaMalloc(&A_subblock_col_indices_local_d, nnz_local_subblock * sizeof(int));
    cudaMalloc(&A_subblock_row_ptr_local_d, (count_subblock[rank]+1) * sizeof(int));
    cudaMemcpy(A_subblock_data_local_d, A_subblock_data_local_h, nnz_local_subblock * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(A_subblock_col_indices_local_d, A_subblock_col_indices_local_h, nnz_local_subblock * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(A_subblock_row_ptr_local_d, A_subblock_row_ptr_local_h, (count_subblock[rank]+1) * sizeof(int), cudaMemcpyHostToDevice);

    cusparseSpMatDescr_t subblock_descriptor;
    cusparseCreateCsr(
        &subblock_descriptor,
        count_subblock[rank],
        subblock_size,
        nnz_local_subblock,
        A_subblock_row_ptr_local_d,
        A_subblock_col_indices_local_d,
        A_subblock_data_local_d,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F);

    double alpha = 1.0;
    double beta = 0.0;
    double *tmp_in_d;
    double *tmp_out_d;
    cudaMalloc(&tmp_in_d, subblock_size * sizeof(double));
    cudaMalloc(&tmp_out_d, count_subblock[rank] * sizeof(double));
    cusparseDnVecDescr_t subblock_vector_descriptor_in;
    cusparseCreateDnVec(
        &subblock_vector_descriptor_in,
        subblock_size,
        tmp_in_d,
        CUDA_R_64F);
    cusparseDnVecDescr_t subblock_vector_descriptor_out;
    cusparseCreateDnVec(
        &subblock_vector_descriptor_out,
        count_subblock[rank],
        tmp_out_d,
        CUDA_R_64F);

    size_t subblock_buffersize;

    cusparseHandle_t cusparse_handle;
    cusparseCreate(&cusparse_handle);

    cusparseSpMV_bufferSize(
        cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        subblock_descriptor,
        subblock_vector_descriptor_in,
        &beta,
        subblock_vector_descriptor_out,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &subblock_buffersize);

    cusparseDestroy(cusparse_handle);
    cudaFree(tmp_in_d);
    cudaFree(tmp_out_d);
    cusparseDestroyDnVec(subblock_vector_descriptor_in);
    cusparseDestroyDnVec(subblock_vector_descriptor_out);

    double *subblock_buffer_d;
    cudaMalloc(&subblock_buffer_d, subblock_buffersize);


    double *diag_inv_local_d;
    cudaMalloc(&diag_inv_local_d, rows_per_rank * sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < rows_per_rank; ++i) {
        diag_local_h[i] = 0.0;
    }
    #pragma omp parallel for
    for (int i = 0; i < rows_per_rank; ++i) {
        for (int j = row_indptr_local_h[i]; j < row_indptr_local_h[i+1]; ++j) {
            if (col_indices_local_h[j] == i + row_start_index) {
                diag_local_h[i] = data_local_h[j];
            }
        }
    }
    // only diagonal block matters for the preconditioner
    #pragma omp parallel for
    for(int i = 0; i < count_subblock[rank]; i++){
        for(int j = A_subblock_row_ptr_local_h[i]; j < A_subblock_row_ptr_local_h[i+1]; j++){
            int col = A_subblock_col_indices_local_h[j];
            if(i + displ_subblock[rank] == col){
                diag_local_h[subblock_indices_local_h[i]] += A_subblock_data_local_h[j];
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < rows_per_rank; ++i) {
        diag_local_h[i] = 1.0 / diag_local_h[i];
    }
    cudaMemcpy(diag_inv_local_d, diag_local_h, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice);    


    Distributed_subblock_sparse A_subblock;
    A_subblock.subblock_indices_local_d = subblock_indices_local_d;
    A_subblock.descriptor = &subblock_descriptor;
    A_subblock.buffer_d = subblock_buffer_d;
    A_subblock.subblock_size = subblock_size;
    A_subblock.count_subblock_h = count_subblock;
    A_subblock.displ_subblock_h = displ_subblock;
    A_subblock.send_subblock_requests = new MPI_Request[size-1];
    A_subblock.recv_subblock_requests = new MPI_Request[size-1];
    A_subblock.streams_recv_subblock = new cudaStream_t[size-1];



    for(int i = 0; i < size-1; i++){
        cudaStreamCreate(&A_subblock.streams_recv_subblock[i]);
    }
    A_subblock.events_recv_subblock = new cudaEvent_t[size];
    for(int i = 0; i < size; i++){
        cudaEventCreateWithFlags(&A_subblock.events_recv_subblock[i], cudaEventDisableTiming);
    }

    for(int i = 0; i < measurements; i++){

        cudaMemcpy(r_local_d, r_local_h, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(x_local_d, starting_guess_h + row_start_index,
            rows_per_rank * sizeof(double), cudaMemcpyHostToDevice);


        cudaDeviceSynchronize();
        MPI_Barrier(comm);
        time_taken[i] = -MPI_Wtime();
        
        iterative_solver::conjugate_gradient_jacobi_split_sparse<distributed_spmv_split_sparse>(
            A_subblock,
            A_distributed,
            p_distributed,
            r_local_d,
            x_local_d,
            diag_inv_local_d,
            relative_tolerance,
            max_iterations,
            comm);

        cudaDeviceSynchronize();
        MPI_Barrier(comm);
        time_taken[i] += MPI_Wtime();
        if(rank == 0){
            std::cout << "time_taken["<<i<<"] " << time_taken[i] << std::endl;
        }
    }


    for(int i = 0; i < size-1; i++){
        cudaStreamDestroy(A_subblock.streams_recv_subblock[i]);
    }


    // //copy solution to host
    double *solution = new double[rows_per_rank];
    cudaErrchk(cudaMemcpy(solution,
        x_local_d, rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));
    MPI_Allgatherv(solution, rows_per_rank, MPI_DOUBLE, reference_solution, counts, displacements, MPI_DOUBLE, comm);

    for(int i = 0; i < size; i++){
        cudaEventDestroy(A_subblock.events_recv_subblock[i]);
    }

    delete[] A_subblock.streams_recv_subblock;


    delete[] A_subblock.send_subblock_requests;
    delete[] A_subblock.recv_subblock_requests;    
    delete[] A_subblock.events_recv_subblock;

    delete[] solution;
    delete[] count_subblock;
    delete[] displ_subblock;
    delete[] diag_local_h;
    delete[] subblock_indices_local_h;
    cudaFree(subblock_indices_local_d);

    delete[] row_indptr_local_h;
    delete[] r_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    cudaFree(r_local_d);
    cudaFree(x_local_d);
    cudaFree(dense_subblock_indices_d);
    cudaFree(diag_inv_local_d);

    delete[] A_subblock_row_ptr_local_h;
    delete[] A_subblock_data_local_h;
    delete[] A_subblock_col_indices_local_h;
    cudaFree(A_subblock_data_local_d);
    cudaFree(A_subblock_col_indices_local_d);
    cudaFree(A_subblock_row_ptr_local_d);

    cudaFree(subblock_buffer_d);

    MPI_Barrier(comm);
}
template 
void test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse1>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *A_subblock_data_h,
    int *A_subblock_col_indices_h,
    int *A_subblock_row_ptr_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *time_taken, int measurements);


int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cudaError_t set_device_error = cudaSetDevice(0);
    std::cout << "rank " << rank << " set_device_error " << set_device_error << std::endl;

    std::string data_path = "/home/cc/100/";
    std::string save_path = "/home/cc/data/";


    int matrix_size = 102722;
    int nnz_sparse = 1707556;
    int nnz_tot = 95903772;
    int subblock_size = 14854;    
    int indices_offset = 2;
    int nnz_subblock = 94211070;

    int max_iterations = 10000;
    double relative_tolerance = 1e-18;

    int rows_per_rank = matrix_size / size;
    int remainder = matrix_size % size;
    int row_start_index = rank * rows_per_rank;
    int col_start_index = row_start_index + rows_per_rank;
    if (rank == size-1) {
        col_start_index += remainder;
        rows_per_rank += remainder;
    }

    double *data_sparse = new double[nnz_sparse];
    int *row_ptr_sparse = new int[matrix_size+1];
    int *col_indices_sparse = new int[nnz_sparse];

    double *data_tot = new double[nnz_tot];
    int *row_ptr_tot = new int[matrix_size+1];
    int *col_indices_tot = new int[nnz_tot];

    double *reference_solution = new double[matrix_size];
    double *rhs = new double[matrix_size];
    double *starting_guess = new double[matrix_size];
    for (int i = 0; i < matrix_size; ++i) {
        starting_guess[i] = 0.0;
    }

    
    std::string data_sparse_filename = data_path + "X_data_sparse.bin";
    std::string row_ptr_sparse_filename = data_path + "X_row_ptr_sparse.bin";
    std::string col_indices_sparse_filename = data_path + "X_col_indices_sparse.bin";

    std::string data_tot_filename = data_path + "X_data.bin";
    std::string row_ptr_tot_filename = data_path + "X_row_ptr.bin";
    std::string col_indices_tot_filename = data_path + "X_col_indices.bin";

    std::string rhs_filename = data_path + "X_rhs.bin";

    load_binary_array<double>(data_sparse_filename, data_sparse, nnz_sparse);
    load_binary_array<int>(row_ptr_sparse_filename, row_ptr_sparse, matrix_size+1);
    load_binary_array<int>(col_indices_sparse_filename, col_indices_sparse, nnz_sparse);

    load_binary_array<double>(data_tot_filename, data_tot, nnz_tot);
    load_binary_array<int>(row_ptr_tot_filename, row_ptr_tot, matrix_size+1);
    load_binary_array<int>(col_indices_tot_filename, col_indices_tot, nnz_tot);

    load_binary_array<double>(rhs_filename, rhs, matrix_size);

    int *dense_subblock_indices = new int[subblock_size];
    double *dense_subblock_data = new double[subblock_size * subblock_size];
    double *data_subblock = new double[nnz_subblock];
    int *row_ptr_subblock = new int[subblock_size+1];
    int *col_indices_subblock = new int[nnz_subblock];

    std::string dense_subblock_indices_filename = data_path + "tunnel_indices.bin";
    std::string dense_subblock_data_filename = data_path + "tunnel_matrix.bin";
    std::string data_subblock_filename = data_path + "tunnel_matrix_data.bin";
    std::string row_ptr_subblock_filename = data_path + "tunnel_matrix_row_ptr.bin";
    std::string col_indices_subblock_filename = data_path + "tunnel_matrix_col_indices.bin";

    load_binary_array<int>(dense_subblock_indices_filename, dense_subblock_indices, subblock_size);
    load_binary_array<double>(dense_subblock_data_filename, dense_subblock_data, subblock_size * subblock_size);
    load_binary_array<double>(data_subblock_filename, data_subblock, nnz_subblock);
    load_binary_array<int>(row_ptr_subblock_filename, row_ptr_subblock, subblock_size+1);
    load_binary_array<int>(col_indices_subblock_filename, col_indices_subblock, nnz_subblock);

    for(int i = 0; i < subblock_size; i++){
        dense_subblock_indices[i] += indices_offset;
    }

    // correct for wrong tot matrix
    double *diag = new double[matrix_size];
    for (int i = 0; i < matrix_size; ++i) {
        diag[i] = 0.0;
    }
    for (int i = 0; i < matrix_size; ++i) {
        for (int j = row_ptr_sparse[i]; j < row_ptr_sparse[i+1]; ++j) {
            if (col_indices_sparse[j] == i) {
                diag[i] = data_sparse[j];
            }
        }
    }
    for(int i = 0; i < subblock_size; i++){
        for(int j = 0; j < subblock_size; j++){
            if(dense_subblock_indices[i] == dense_subblock_indices[j]){
                diag[dense_subblock_indices[i]] += dense_subblock_data[i + j * subblock_size];
            }
        }
    }
    for (int i = 0; i < matrix_size; ++i) {
        for(int j = row_ptr_tot[i]; j < row_ptr_tot[i+1]; j++){
            if(col_indices_tot[j] == i){
                data_tot[j] = diag[i];
            }
        }
    }

    // extract diagonal
    double *diag_inv_h = new double[matrix_size];
    for(int i = 0; i < matrix_size; i++){
        diag_inv_h[i] = 1.0 / diag[i];
    }
    double *diag_inv_d;
    cudaMalloc(&diag_inv_d, matrix_size * sizeof(double));
    cudaMemcpy(diag_inv_d, diag_inv_h, matrix_size * sizeof(double), cudaMemcpyHostToDevice);


    int start_up_measurements = 2;
    int true_number_of_measurements = 20;
    int number_of_measurements = start_up_measurements + true_number_of_measurements;
    double time_tot[number_of_measurements];
    double time_split_sparse1[number_of_measurements];

    get_solution<dspmv::gpu_packing>(
        data_tot,
        col_indices_tot,
        row_ptr_tot,
        rhs,
        reference_solution,
        starting_guess,
        matrix_size,
        relative_tolerance,
        max_iterations,
        MPI_COMM_WORLD,
        diag_inv_d,
        time_tot,
        number_of_measurements
    );

    double *test_solution_split = new double[matrix_size];
    
    test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse1>(
            data_sparse,
            col_indices_sparse,
            row_ptr_sparse,
            dense_subblock_indices,
            data_subblock,
            col_indices_subblock,
            row_ptr_subblock,
            subblock_size,
            rhs,
            test_solution_split,
            starting_guess,
            matrix_size,
            relative_tolerance,
            max_iterations,
            MPI_COMM_WORLD,
            time_split_sparse1,
            number_of_measurements
    );


    double sum = 0.0;
    double diff_split = 0.0;
    #pragma omp parallel for reduction(+:sum, diff_split)
    for (int i = 0; i < matrix_size; ++i) {
        sum += std::abs(reference_solution[i]) * std::abs(reference_solution[i]);
        diff_split += std::abs(reference_solution[i] - test_solution_split[i]) * std::abs(reference_solution[i] - test_solution_split[i]);
    }
    if(rank == 0){
        std::cout << "relative error " << std::sqrt(diff_split / sum) << std::endl; 
    }

    cudaFree(diag_inv_d);
    delete[] diag_inv_h;
    delete[] data_sparse;
    delete[] row_ptr_sparse;
    delete[] col_indices_sparse;
    delete[] data_tot;
    delete[] row_ptr_tot;
    delete[] col_indices_tot;
    delete[] reference_solution;
    delete[] rhs;
    delete[] starting_guess;
    delete[] dense_subblock_indices;
    delete[] dense_subblock_data;
    delete[] diag;

    // std::string path_solve_tot = get_filename(save_path, "non_split", size, rank);
    // std::string path_solve_split_sparse1 = get_filename(save_path, "split", size, rank);

    // save_measurements(path_solve_tot,
    //     time_tot + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_solve_split_sparse1,
    //     time_split_sparse1 + start_up_measurements,
    //     true_number_of_measurements, true);

    MPI_Finalize();
    return 0;
}
