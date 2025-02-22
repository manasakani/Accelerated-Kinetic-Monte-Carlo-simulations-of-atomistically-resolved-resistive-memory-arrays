#include <iostream>
#include <string>
#include "utils.h"
#include <mpi.h>
#include <hip/hip_runtime.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include "utils_gpu.h"
#include <hipblas.h>
#include "../dist_iterative/dist_conjugate_gradient.h"
#include "../dist_iterative/dist_spmv.h"
#include <pthread.h>
#include "rocsparse.h"


template <void (*distributed_spmv)(
    Distributed_matrix&,
    Distributed_vector&,
    rocsparse_dnvec_descr&,
    hipStream_t&,
    rocsparse_handle &)>
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
        rocsparse_spmv_alg_csr_adaptive,
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
        hipMalloc(&r_local_d, rows_per_rank * sizeof(double));
        hipMalloc(&x_local_d, rows_per_rank * sizeof(double));
        hipMemcpy(r_local_d, r_local_h, rows_per_rank * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(x_local_d, starting_guess_h + row_start_index,
            rows_per_rank * sizeof(double), hipMemcpyHostToDevice);


        double *diag_inv_local_d = diag_inv_d + row_start_index;

        MPI_Barrier(comm);
        hipDeviceSynchronize();
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
        hipDeviceSynchronize();
        time_taken[i] += MPI_Wtime();
        if(rank == 0){
            std::cout << rank << " time_taken["<<i<<"] " << time_taken[i] << std::endl;
        }
    }
    //copy solution to host
    cudaErrchk(hipMemcpy(reference_solution + row_start_index,
        x_local_d, rows_per_rank * sizeof(double), hipMemcpyDeviceToHost));
    //MPI allgather in place
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, reference_solution, counts, displacements, MPI_DOUBLE, comm);

    delete[] row_indptr_local_h;
    delete[] r_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    hipFree(r_local_d);
    hipFree(x_local_d);

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

template <void (*distributed_spmv_split)
    (Distributed_subblock &,
    Distributed_matrix &,    
    double *,
    double *,
    Distributed_vector &,
    double *,
    rocsparse_dnvec_descr &,
    double *,
    hipStream_t &,
    rocblas_handle &,
    rocsparse_handle&)>
void test_preconditioned_split(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *A_subblock_h,
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
        rocsparse_spmv_alg_csr_adaptive,
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
    hipMalloc(&r_local_d, rows_per_rank * sizeof(double));
    hipMalloc(&x_local_d, rows_per_rank * sizeof(double));

    int *dense_subblock_indices_d;
    double *A_subblock_d;
    hipMalloc(&dense_subblock_indices_d, subblock_size * sizeof(int));
    hipMalloc(&A_subblock_d, subblock_size * subblock_size * sizeof(double));
    hipMemcpy(dense_subblock_indices_d, subblock_indices_h, subblock_size * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(A_subblock_d, A_subblock_h, subblock_size * subblock_size * sizeof(double), hipMemcpyHostToDevice);

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
    hipMalloc(&subblock_indices_local_d, count_subblock[rank] * sizeof(int));
    hipMemcpy(subblock_indices_local_d, subblock_indices_local_h, count_subblock[rank] * sizeof(int), hipMemcpyHostToDevice);


    double *diag_inv_local_d;
    hipMalloc(&diag_inv_local_d, rows_per_rank * sizeof(double));
    double *A_subblock_local_h = new double[count_subblock[rank] * subblock_size];
    double *A_subblock_local_d;
    hipMalloc(&A_subblock_local_d, count_subblock[rank] * subblock_size * sizeof(double));

    #pragma omp for
    for(int i = 0; i < count_subblock[rank]; i++){
        for(int j = 0; j < subblock_size; j++){
            A_subblock_local_h[i + j * count_subblock[rank]] = A_subblock_h[i + displ_subblock[rank] + j * subblock_size];
        }
    }
    hipMemcpy(A_subblock_local_d, A_subblock_local_h, count_subblock[rank] * subblock_size * sizeof(double), hipMemcpyHostToDevice);



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
        for(int j = 0; j < count_subblock[rank]; j++){
            if(subblock_indices_local_h[i] == subblock_indices_local_h[j]){
                diag_local_h[subblock_indices_local_h[i]] += A_subblock_local_h[i + j * count_subblock[rank] +
                    displ_subblock[rank] * count_subblock[rank]];
            }
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < rows_per_rank; ++i) {
        diag_local_h[i] = 1.0 / diag_local_h[i];
    }
    hipMemcpy(diag_inv_local_d, diag_local_h, rows_per_rank * sizeof(double), hipMemcpyHostToDevice);    




    Distributed_subblock A_subblock;
    A_subblock.subblock_indices_local_d = subblock_indices_local_d;
    A_subblock.A_subblock_local_d = A_subblock_local_d;
    A_subblock.subblock_size = subblock_size;
    A_subblock.count_subblock_h = count_subblock;
    A_subblock.displ_subblock_h = displ_subblock;
    A_subblock.send_subblock_requests = new MPI_Request[size-1];
    A_subblock.recv_subblock_requests = new MPI_Request[size-1];
    A_subblock.streams_recv_subblock = new hipStream_t[size-1];
    for(int i = 0; i < size-1; i++){
        hipStreamCreate(&A_subblock.streams_recv_subblock[i]);
    }
    A_subblock.events_recv_subblock = new hipEvent_t[size];
    for(int i = 0; i < size; i++){
        hipEventCreateWithFlags(&A_subblock.events_recv_subblock[i], hipEventDisableTiming);
    }
    for(int i = 0; i < measurements; i++){

        hipMemcpy(r_local_d, r_local_h, rows_per_rank * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(x_local_d, starting_guess_h + row_start_index,
            rows_per_rank * sizeof(double), hipMemcpyHostToDevice);


        MPI_Barrier(comm);
        hipDeviceSynchronize();
        time_taken[i] = -MPI_Wtime();
        
        iterative_solver::conjugate_gradient_jacobi_split<distributed_spmv_split>(
            A_subblock,
            A_distributed,
            p_distributed,
            r_local_d,
            x_local_d,
            diag_inv_local_d,
            relative_tolerance,
            max_iterations,
            comm);

        hipDeviceSynchronize();
        time_taken[i] += MPI_Wtime();
        if(rank == 0){
            std::cout << rank << " time_taken["<<i<<"] " << time_taken[i] << std::endl;
        }
    }


    for(int i = 0; i < size-1; i++){
        hipStreamDestroy(A_subblock.streams_recv_subblock[i]);
    }


    // //copy solution to host
    double *solution = new double[rows_per_rank];
    cudaErrchk(hipMemcpy(solution,
        x_local_d, rows_per_rank * sizeof(double), hipMemcpyDeviceToHost));
    MPI_Allgatherv(solution, rows_per_rank, MPI_DOUBLE, reference_solution, counts, displacements, MPI_DOUBLE, comm);

    for(int i = 0; i < size; i++){
        hipEventDestroy(A_subblock.events_recv_subblock[i]);
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
    delete[] A_subblock_local_h;
    hipFree(subblock_indices_local_d);
    hipFree(A_subblock_local_d);

    delete[] row_indptr_local_h;
    delete[] r_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    hipFree(r_local_d);
    hipFree(x_local_d);
    hipFree(dense_subblock_indices_d);
    hipFree(A_subblock_d);
    hipFree(diag_inv_local_d);
    MPI_Barrier(comm);
}
template 
void test_preconditioned_split<dspmv_split::spmm_split1>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *A_subblock_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *time_taken, int measurements);
template 
void test_preconditioned_split<dspmv_split::spmm_split2>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *A_subblock_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *time_taken, int measurements);
template 
void test_preconditioned_split<dspmv_split::spmm_split3>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *A_subblock_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *time_taken, int measurements);
template 
void test_preconditioned_split<dspmv_split::spmm_split4>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *A_subblock_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *time_taken, int measurements);
template 
void test_preconditioned_split<dspmv_split::spmm_split5>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *A_subblock_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *time_taken, int measurements);
template 
void test_preconditioned_split<dspmv_split::spmm_split6>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *A_subblock_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *time_taken, int measurements);



template <void (*distributed_spmv_split_sparse)
    (Distributed_subblock_sparse &,
    Distributed_matrix &,    
    double *,
    double *,
    rocsparse_dnvec_descr &,
    Distributed_vector &,
    double *,
    rocsparse_dnvec_descr &,
    rocsparse_dnvec_descr &,
    double *,
    hipStream_t &,
    rocsparse_handle &)>
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
        std::cout << "PCG test starts split sparse" << std::endl;
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
        rocsparse_spmv_alg_csr_adaptive,
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
    hipMalloc(&r_local_d, rows_per_rank * sizeof(double));
    hipMalloc(&x_local_d, rows_per_rank * sizeof(double));

    int *dense_subblock_indices_d;
    hipMalloc(&dense_subblock_indices_d, subblock_size * sizeof(int));
    hipMemcpy(dense_subblock_indices_d, subblock_indices_h, subblock_size * sizeof(int), hipMemcpyHostToDevice);
    MPI_Barrier(comm);
    hipDeviceSynchronize();

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
    hipMalloc(&subblock_indices_local_d, count_subblock[rank] * sizeof(int));
    hipMemcpy(subblock_indices_local_d, subblock_indices_local_h, count_subblock[rank] * sizeof(int), hipMemcpyHostToDevice);



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
    hipMalloc(&A_subblock_data_local_d, nnz_local_subblock * sizeof(double));
    hipMalloc(&A_subblock_col_indices_local_d, nnz_local_subblock * sizeof(int));
    hipMalloc(&A_subblock_row_ptr_local_d, (count_subblock[rank]+1) * sizeof(int));
    hipMemcpy(A_subblock_data_local_d, A_subblock_data_local_h, nnz_local_subblock * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(A_subblock_col_indices_local_d, A_subblock_col_indices_local_h, nnz_local_subblock * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(A_subblock_row_ptr_local_d, A_subblock_row_ptr_local_h, (count_subblock[rank]+1) * sizeof(int), hipMemcpyHostToDevice);

    rocsparse_spmat_descr subblock_descriptor;
    rocsparse_create_csr_descr(&subblock_descriptor,
                            count_subblock[rank],
                            subblock_size,
                            nnz_local_subblock,
                            A_subblock_row_ptr_local_d,
                            A_subblock_col_indices_local_d,
                            A_subblock_data_local_d,
                            rocsparse_indextype_i32,
                            rocsparse_indextype_i32,
                            rocsparse_index_base_zero,
                            rocsparse_datatype_f64_r);

    double alpha = 1.0;
    double beta = 0.0;
    double *tmp_in_d;
    double *tmp_out_d;
    hipMalloc(&tmp_in_d, subblock_size * sizeof(double));
    hipMemset(&tmp_in_d, 1, subblock_size * sizeof(double)); //
    hipMalloc(&tmp_out_d, count_subblock[rank] * sizeof(double));

    rocsparse_dnvec_descr subblock_vector_descriptor_in;
    rocsparse_dnvec_descr subblock_vector_descriptor_out;

    rocsparse_create_dnvec_descr(&subblock_vector_descriptor_in,
                                subblock_size,
                                tmp_in_d,
                                rocsparse_datatype_f64_r);

    // Create dense vector Y
    rocsparse_create_dnvec_descr(&subblock_vector_descriptor_out,
                                count_subblock[rank],
                                tmp_out_d,
                                rocsparse_datatype_f64_r);

    size_t subblock_buffersize;

    rocsparse_handle rocsparse_handle;
    rocsparse_create_handle(&rocsparse_handle);


    rocsparse_spmv_alg algo = rocsparse_spmv_alg_csr_adaptive;

    rocsparse_spmv(rocsparse_handle,
                rocsparse_operation_none,
                &alpha,
                subblock_descriptor,
                subblock_vector_descriptor_in,
                &beta,
                subblock_vector_descriptor_out,
                rocsparse_datatype_f64_r,
                algo,
                &subblock_buffersize,
                nullptr);
    
    // hipsparseDestroy(cusparse_handle);
    rocsparse_destroy_handle(rocsparse_handle);

    hipFree(tmp_in_d);
    hipFree(tmp_out_d);

    rocsparse_destroy_dnvec_descr(subblock_vector_descriptor_in);
    rocsparse_destroy_dnvec_descr(subblock_vector_descriptor_out);

    double *subblock_buffer_d;
    hipMalloc(&subblock_buffer_d, subblock_buffersize);


    double *diag_inv_local_d;
    hipMalloc(&diag_inv_local_d, rows_per_rank * sizeof(double));

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
    hipMemcpy(diag_inv_local_d, diag_local_h, rows_per_rank * sizeof(double), hipMemcpyHostToDevice);    


    Distributed_subblock_sparse A_subblock;
    A_subblock.subblock_indices_local_d = subblock_indices_local_d;
    A_subblock.descriptor = &subblock_descriptor;
    A_subblock.algo = algo;
    A_subblock.buffersize = &subblock_buffersize;
    A_subblock.buffer_d = subblock_buffer_d;
    A_subblock.subblock_size = subblock_size;
    A_subblock.count_subblock_h = count_subblock;
    A_subblock.displ_subblock_h = displ_subblock;
    A_subblock.send_subblock_requests = new MPI_Request[size-1];
    A_subblock.recv_subblock_requests = new MPI_Request[size-1];
    A_subblock.streams_recv_subblock = new hipStream_t[size-1];



    for(int i = 0; i < size-1; i++){
        hipStreamCreate(&A_subblock.streams_recv_subblock[i]);
    }
    A_subblock.events_recv_subblock = new hipEvent_t[size];
    for(int i = 0; i < size; i++){
        hipEventCreateWithFlags(&A_subblock.events_recv_subblock[i], hipEventDisableTiming);
    }

    if(rank == 0){
        std::cout << "Before loop" << std::endl;
    }

    for(int i = 0; i < measurements; i++){

        hipMemcpy(r_local_d, r_local_h, rows_per_rank * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(x_local_d, starting_guess_h + row_start_index,
            rows_per_rank * sizeof(double), hipMemcpyHostToDevice);


        MPI_Barrier(comm);
        hipDeviceSynchronize();
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

        hipDeviceSynchronize();
        time_taken[i] += MPI_Wtime();
        if(rank == 0){
            std::cout << rank << " time_taken["<<i<<"] " << time_taken[i] << std::endl;
        }
    }


    for(int i = 0; i < size-1; i++){
        hipStreamDestroy(A_subblock.streams_recv_subblock[i]);
    }


    // //copy solution to host
    double *solution = new double[rows_per_rank];
    cudaErrchk(hipMemcpy(solution,
        x_local_d, rows_per_rank * sizeof(double), hipMemcpyDeviceToHost));
    MPI_Allgatherv(solution, rows_per_rank, MPI_DOUBLE, reference_solution, counts, displacements, MPI_DOUBLE, comm);

    for(int i = 0; i < size; i++){
        hipEventDestroy(A_subblock.events_recv_subblock[i]);
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
    hipFree(subblock_indices_local_d);

    delete[] row_indptr_local_h;
    delete[] r_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    hipFree(r_local_d);
    hipFree(x_local_d);
    hipFree(dense_subblock_indices_d);
    hipFree(diag_inv_local_d);

    delete[] A_subblock_row_ptr_local_h;
    delete[] A_subblock_data_local_h;
    delete[] A_subblock_col_indices_local_h;
    hipFree(A_subblock_data_local_d);
    hipFree(A_subblock_col_indices_local_d);
    hipFree(A_subblock_row_ptr_local_d);

    hipFree(subblock_buffer_d);

    MPI_Barrier(comm);
    // DEBUG - hipsparse
    // exit(1);
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
template 
void test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse2>(
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
template 
void test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse3>(
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
template 
void test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse4>(
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
template 
void test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse5>(
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

    // MPI_Init(&argc, &argv);

    // Init thread multiple
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    std::cout << "provided " << provided << std::endl;

    if (provided != MPI_THREAD_MULTIPLE) {
        std::cout << "MPI_THREAD_MULTIPLE not supported by MPI, aborting" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char* slurm_localid = getenv("SLURM_LOCALID");
    int localid = atoi(slurm_localid);
    int device_id = localid; 
    std::cout << "rank " << rank << " device_id " << device_id << std::endl;

    hipError_t set_device_error = hipSetDevice(localid);
    std::cout << "rank " << rank << " set_device_error " << set_device_error << std::endl;

    char* rocr_visible_devices = getenv("ROCR_VISIBLE_DEVICES");
    if (rocr_visible_devices != nullptr) {
        std::cout << "Rank " << rank << " ROCR_VISIBLE_DEVICES: " << rocr_visible_devices << std::endl;
    } else {
        std::cerr << "Rank " << rank << " ROCR_VISIBLE_DEVICES not set" << std::endl;
        exit(1);
    }

    // std::string data_path = "/usr/scratch/mont-fort23/almaeder/kmc_split/";

    std::string data_path = "/scratch/project_465000929/maederal/100/";
    // std::string data_path = "/usr/scratch/mont-fort23/almaeder/kmc_split/100/";

    int matrix_size = 102722;
    int nnz_sparse = 1707556;
    int nnz_tot = 95903772;
    int subblock_size = 14854;    
    int indices_offset = 2;
    int nnz_subblock = 94211070;

    // std::string data_path = "/scratch/snx3000/amaeder/37/";
    // std::string data_path = "/usr/scratch/mont-fort23/almaeder/kmc_split/37/";

    // int matrix_size = 25682;
    // int nnz_sparse = 416286;
    // int nnz_tot = 6297966;
    // int subblock_size = 3713;    
    // int indices_offset = 2;
    // int nnz_subblock = 5885393;

    // std::string data_path = "/usr/scratch/mont-fort23/almaeder/kmc_split/";

    // int matrix_size = 25682;
    // int nnz_sparse = 1439928;
    // int nnz_tot = 15193074;
    // int subblock_size = 3713;
    // int indices_offset = 0;

    if(nnz_subblock + nnz_sparse - subblock_size != nnz_tot){
        std::cout << "nnz_subblock + nnz_sparse - subblock_size != nnz_tot" << std::endl;
        exit(1);
    }

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

    std::cout << "rank " << rank << " row_start_index " << row_start_index << " row_end_index " << col_start_index << std::endl;
    std::cout << "rank " << rank << " rows_per_rank " << rows_per_rank << std::endl;

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

    if(rank == 0){

        load_binary_array<double>(data_sparse_filename, data_sparse, nnz_sparse);
        load_binary_array<int>(row_ptr_sparse_filename, row_ptr_sparse, matrix_size+1);
        load_binary_array<int>(col_indices_sparse_filename, col_indices_sparse, nnz_sparse);

        load_binary_array<double>(data_tot_filename, data_tot, nnz_tot);
        load_binary_array<int>(row_ptr_tot_filename, row_ptr_tot, matrix_size+1);
        load_binary_array<int>(col_indices_tot_filename, col_indices_tot, nnz_tot);

        load_binary_array<double>(rhs_filename, rhs, matrix_size);

        load_binary_array<int>(dense_subblock_indices_filename, dense_subblock_indices, subblock_size);
        load_binary_array<double>(dense_subblock_data_filename, dense_subblock_data, subblock_size * subblock_size);
        load_binary_array<double>(data_subblock_filename, data_subblock, nnz_subblock);
        load_binary_array<int>(row_ptr_subblock_filename, row_ptr_subblock, subblock_size+1);
        load_binary_array<int>(col_indices_subblock_filename, col_indices_subblock, nnz_subblock);
    }
    // broadcast data
    std::cout << "broadcasting data" << std::endl;
    MPI_Bcast(data_sparse, nnz_sparse, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_ptr_sparse, matrix_size+1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_indices_sparse, nnz_sparse, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(data_tot, nnz_tot, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_ptr_tot, matrix_size+1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_indices_tot, nnz_tot, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(rhs, matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(dense_subblock_indices, subblock_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(dense_subblock_data, subblock_size * subblock_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(data_subblock, nnz_subblock, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_ptr_subblock, subblock_size+1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_indices_subblock, nnz_subblock, MPI_INT, 0, MPI_COMM_WORLD);


    std::cout << "loaded data" << std::endl;


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
    // only diagonal block matters for the preconditioner
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


    // // check matrices the same
    // std::cout << "rank " << rank << " data loaded" << std::endl;

    // double *dense_tot = new double[matrix_size * matrix_size];
    // double *dense_split = new double[matrix_size * matrix_size];
    // for (int i = 0; i < matrix_size; ++i) {
    //     for (int j = 0; j < matrix_size; ++j) {
    //         dense_tot[i * matrix_size + j] = 0.0;
    //         dense_split[i * matrix_size + j] = 0.0;
    //     }
    // }
    // for (int i = 0; i < matrix_size; ++i) {
    //     for (int j = row_ptr_tot[i]; j < row_ptr_tot[i+1]; ++j) {
    //         dense_tot[i * matrix_size + col_indices_tot[j]] = data_tot[j];
    //     }
    // }
    // for (int i = 0; i < matrix_size; ++i) {
    //     for (int j = row_ptr_sparse[i]; j < row_ptr_sparse[i+1]; ++j) {
    //         dense_split[i * matrix_size + col_indices_sparse[j]] = data_sparse[j];
    //     }
    // }
    // for (int i = 0; i < subblock_size; ++i) {
    //     for (int j = 0; j < subblock_size; ++j) {
    //         dense_split[dense_subblock_indices[i] * matrix_size + dense_subblock_indices[j]] +=
    //             dense_subblock_data[i * subblock_size + j];
    //     }
    // }


    // double sum_matrix = 0.0;
    // double diff_matrix = 0.0;
    // for (int i = 0; i < matrix_size; ++i) {
    //     for (int j = 0; j < matrix_size; ++j) {
    //         sum_matrix += std::abs(dense_tot[i * matrix_size + j]) * std::abs(dense_tot[i * matrix_size + j]);
    //         diff_matrix += std::abs(dense_tot[i * matrix_size + j] - dense_split[i * matrix_size + j]) *
    //             std::abs(dense_tot[i * matrix_size + j] - dense_split[i * matrix_size + j]);
    //     }
    // }
    // std::cout << "rank " << rank << " relative between matrices " << std::sqrt(diff_matrix / sum_matrix) << std::endl;
    // for (int i = 0; i < matrix_size; ++i) {
    //     for (int j = 0; j < matrix_size; ++j) {
    //         if(std::abs(dense_tot[i * matrix_size + j] - dense_split[i * matrix_size + j]) > 1e-10){
    //             std::cout << "rank " << rank << " i " << i << " j " << j << " dense_tot " << dense_tot[i * matrix_size + j]
    //                 << " dense_split " << dense_split[i * matrix_size + j] << std::endl;
    //         }
    //     }
    // }


    // extract diagonal
    double *diag_inv_h = new double[matrix_size];
    for(int i = 0; i < matrix_size; i++){
        diag_inv_h[i] = 1.0 / diag[i];
    }
    double *diag_inv_d;
    hipMalloc(&diag_inv_d, matrix_size * sizeof(double));
    hipMemcpy(diag_inv_d, diag_inv_h, matrix_size * sizeof(double), hipMemcpyHostToDevice);


    int start_up_measurements = 2;
    int true_number_of_measurements = 5;
    int number_of_measurements = start_up_measurements + true_number_of_measurements;
    double time_tot[number_of_measurements];
    double time_split1[number_of_measurements];
    double time_split2[number_of_measurements];
    double time_split3[number_of_measurements];
    double time_split4[number_of_measurements];
    double time_split5[number_of_measurements];
    double time_split6[number_of_measurements];
    double time_split_sparse1[number_of_measurements];
    double time_split_sparse2[number_of_measurements];
    double time_split_sparse3[number_of_measurements];
    double time_split_sparse4[number_of_measurements];
    double time_split_sparse5[number_of_measurements];

    // double *solution = new double[matrix_size];
    // std::string solution_path = data_path + "X_solution.bin";
    // load_binary_array<double>(solution_path, solution, matrix_size);

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
    double sum = 0.0;
    double diff_split = 0.0;

    // test_preconditioned_split<dspmv_split::spmm_split1>(
    //         data_sparse,
    //         col_indices_sparse,
    //         row_ptr_sparse,
    //         dense_subblock_indices,
    //         dense_subblock_data,
    //         subblock_size,
    //         rhs,
    //         test_solution_split,
    //         starting_guess,
    //         matrix_size,
    //         relative_tolerance,
    //         max_iterations,
    //         MPI_COMM_WORLD,
    //         time_split1,
    //         number_of_measurements
    // );



    // #pragma omp parallel for reduction(+:sum, diff_split)
    // for (int i = 0; i < matrix_size; ++i) {
    //     sum += std::abs(reference_solution[i]) * std::abs(reference_solution[i]);
    //     diff_split += std::abs(reference_solution[i] - test_solution_split[i]) * std::abs(reference_solution[i] - test_solution_split[i]);
    // }
    // if(rank == 0){
    //     std::cout << " relative error split 1 " << std::sqrt(diff_split / sum) << std::endl; 
    // }
    
    // test_preconditioned_split<dspmv_split::spmm_split2>(
    //         data_sparse,
    //         col_indices_sparse,
    //         row_ptr_sparse,
    //         dense_subblock_indices,
    //         dense_subblock_data,
    //         subblock_size,
    //         rhs,
    //         test_solution_split,
    //         starting_guess,
    //         matrix_size,
    //         relative_tolerance,
    //         max_iterations,
    //         MPI_COMM_WORLD,
    //         time_split2,
    //         number_of_measurements
    // );


    // sum = 0.0;
    // diff_split = 0.0;
    // #pragma omp parallel for reduction(+:sum, diff_split)
    // for (int i = 0; i < matrix_size; ++i) {
    //     sum += std::abs(reference_solution[i]) * std::abs(reference_solution[i]);
    //     diff_split += std::abs(reference_solution[i] - test_solution_split[i]) * std::abs(reference_solution[i] - test_solution_split[i]);
    // }
    // if(rank == 0){
    //     std::cout << " relative error split 2 " << std::sqrt(diff_split / sum) << std::endl; 
    // }
    
    // test_preconditioned_split<dspmv_split::spmm_split3>(
    //         data_sparse,
    //         col_indices_sparse,
    //         row_ptr_sparse,
    //         dense_subblock_indices,
    //         dense_subblock_data,
    //         subblock_size,
    //         rhs,
    //         test_solution_split,
    //         starting_guess,
    //         matrix_size,
    //         relative_tolerance,
    //         max_iterations,
    //         MPI_COMM_WORLD,
    //         time_split3,
    //         number_of_measurements
    // );


    // sum = 0.0;
    // diff_split = 0.0;
    // #pragma omp parallel for reduction(+:sum, diff_split)
    // for (int i = 0; i < matrix_size; ++i) {
    //     sum += std::abs(reference_solution[i]) * std::abs(reference_solution[i]);
    //     diff_split += std::abs(reference_solution[i] - test_solution_split[i]) * std::abs(reference_solution[i] - test_solution_split[i]);
    // }
    // if(rank == 0){
    //     std::cout << " relative error split 3 " << std::sqrt(diff_split / sum) << std::endl; 
    // }
    
    // test_preconditioned_split<dspmv_split::spmm_split4>(
    //         data_sparse,
    //         col_indices_sparse,
    //         row_ptr_sparse,
    //         dense_subblock_indices,
    //         dense_subblock_data,
    //         subblock_size,
    //         rhs,
    //         test_solution_split,
    //         starting_guess,
    //         matrix_size,
    //         relative_tolerance,
    //         max_iterations,
    //         MPI_COMM_WORLD,
    //         time_split4,
    //         number_of_measurements
    // );


    // sum = 0.0;
    // diff_split = 0.0;
    // #pragma omp parallel for reduction(+:sum, diff_split)
    // for (int i = 0; i < matrix_size; ++i) {
    //     sum += std::abs(reference_solution[i]) * std::abs(reference_solution[i]);
    //     diff_split += std::abs(reference_solution[i] - test_solution_split[i]) * std::abs(reference_solution[i] - test_solution_split[i]);
    // }
    // if(rank == 0){
    //     std::cout << " relative error split 4 " << std::sqrt(diff_split / sum) << std::endl; 
    // }
    

    test_preconditioned_split<dspmv_split::spmm_split5>(
            data_sparse,
            col_indices_sparse,
            row_ptr_sparse,
            dense_subblock_indices,
            dense_subblock_data,
            subblock_size,
            rhs,
            test_solution_split,
            starting_guess,
            matrix_size,
            relative_tolerance,
            max_iterations,
            MPI_COMM_WORLD,
            time_split5,
            number_of_measurements
    );


    sum = 0.0;
    diff_split = 0.0;
    #pragma omp parallel for reduction(+:sum, diff_split)
    for (int i = 0; i < matrix_size; ++i) {
        sum += std::abs(reference_solution[i]) * std::abs(reference_solution[i]);
        diff_split += std::abs(reference_solution[i] - test_solution_split[i]) * std::abs(reference_solution[i] - test_solution_split[i]);
    }
    if(rank == 0){
        std::cout << " relative error split 5 " << std::sqrt(diff_split / sum) << std::endl; 
    }
    
    
    // test_preconditioned_split<dspmv_split::spmm_split6>(
    //         data_sparse,
    //         col_indices_sparse,
    //         row_ptr_sparse,
    //         dense_subblock_indices,
    //         dense_subblock_data,
    //         subblock_size,
    //         rhs,
    //         test_solution_split,
    //         starting_guess,
    //         matrix_size,
    //         relative_tolerance,
    //         max_iterations,
    //         MPI_COMM_WORLD,
    //         time_split6,
    //         number_of_measurements
    // );


    // sum = 0.0;
    // diff_split = 0.0;
    // #pragma omp parallel for reduction(+:sum, diff_split)
    // for (int i = 0; i < matrix_size; ++i) {
    //     sum += std::abs(reference_solution[i]) * std::abs(reference_solution[i]);
    //     diff_split += std::abs(reference_solution[i] - test_solution_split[i]) * std::abs(reference_solution[i] - test_solution_split[i]);
    // }
    // if(rank == 0){
    //     std::cout << " relative error split 6 " << std::sqrt(diff_split / sum) << std::endl; 
    // }
    
    
    // test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse1>(
    //         data_sparse,
    //         col_indices_sparse,
    //         row_ptr_sparse,
    //         dense_subblock_indices,
    //         data_subblock,
    //         col_indices_subblock,
    //         row_ptr_subblock,
    //         subblock_size,
    //         rhs,
    //         test_solution_split,
    //         starting_guess,
    //         matrix_size,
    //         relative_tolerance,
    //         max_iterations,
    //         MPI_COMM_WORLD,
    //         time_split_sparse1,
    //         number_of_measurements
    // );


    // sum = 0.0;
    // diff_split = 0.0;
    // #pragma omp parallel for reduction(+:sum, diff_split)
    // for (int i = 0; i < matrix_size; ++i) {
    //     sum += std::abs(reference_solution[i]) * std::abs(reference_solution[i]);
    //     diff_split += std::abs(reference_solution[i] - test_solution_split[i]) * std::abs(reference_solution[i] - test_solution_split[i]);
    // }
    // if(rank == 0){
    //     std::cout << " relative error split sparse1 " << std::sqrt(diff_split / sum) << std::endl; 
    // }
    
    
    // test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse2>(
    //         data_sparse,
    //         col_indices_sparse,
    //         row_ptr_sparse,
    //         dense_subblock_indices,
    //         data_subblock,
    //         col_indices_subblock,
    //         row_ptr_subblock,
    //         subblock_size,
    //         rhs,
    //         test_solution_split,
    //         starting_guess,
    //         matrix_size,
    //         relative_tolerance,
    //         max_iterations,
    //         MPI_COMM_WORLD,
    //         time_split_sparse2,
    //         number_of_measurements
    // );


    // sum = 0.0;
    // diff_split = 0.0;
    // #pragma omp parallel for reduction(+:sum, diff_split)
    // for (int i = 0; i < matrix_size; ++i) {
    //     sum += std::abs(reference_solution[i]) * std::abs(reference_solution[i]);
    //     diff_split += std::abs(reference_solution[i] - test_solution_split[i]) * std::abs(reference_solution[i] - test_solution_split[i]);
    // }
    // if(rank == 0){
    //     std::cout << " relative error split sparse2 " << std::sqrt(diff_split / sum) << std::endl; 
    // }
    

    // test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse3>(
    //         data_sparse,
    //         col_indices_sparse,
    //         row_ptr_sparse,
    //         dense_subblock_indices,
    //         data_subblock,
    //         col_indices_subblock,
    //         row_ptr_subblock,
    //         subblock_size,
    //         rhs,
    //         test_solution_split,
    //         starting_guess,
    //         matrix_size,
    //         relative_tolerance,
    //         max_iterations,
    //         MPI_COMM_WORLD,
    //         time_split_sparse3,
    //         number_of_measurements
    // );


    // sum = 0.0;
    // diff_split = 0.0;
    // #pragma omp parallel for reduction(+:sum, diff_split)
    // for (int i = 0; i < matrix_size; ++i) {
    //     sum += std::abs(reference_solution[i]) * std::abs(reference_solution[i]);
    //     diff_split += std::abs(reference_solution[i] - test_solution_split[i]) * std::abs(reference_solution[i] - test_solution_split[i]);
    // }
    // if(rank == 0){
    //     std::cout << " relative error split sparse3 " << std::sqrt(diff_split / sum) << std::endl; 
    // }
    

    test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse4>(
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
            time_split_sparse4,
            number_of_measurements
    );


    sum = 0.0;
    diff_split = 0.0;
    #pragma omp parallel for reduction(+:sum, diff_split)
    for (int i = 0; i < matrix_size; ++i) {
        sum += std::abs(reference_solution[i]) * std::abs(reference_solution[i]);
        diff_split += std::abs(reference_solution[i] - test_solution_split[i]) * std::abs(reference_solution[i] - test_solution_split[i]);
    }
    if(rank == 0){
        std::cout << " relative error split sparse4 " << std::sqrt(diff_split / sum) << std::endl; 
    }
    

    // test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse5>(
    //         data_sparse,
    //         col_indices_sparse,
    //         row_ptr_sparse,
    //         dense_subblock_indices,
    //         data_subblock,
    //         col_indices_subblock,
    //         row_ptr_subblock,
    //         subblock_size,
    //         rhs,
    //         test_solution_split,
    //         starting_guess,
    //         matrix_size,
    //         relative_tolerance,
    //         max_iterations,
    //         MPI_COMM_WORLD,
    //         time_split_sparse5,
    //         number_of_measurements
    // );


    // sum = 0.0;
    // diff_split = 0.0;
    // #pragma omp parallel for reduction(+:sum, diff_split)
    // for (int i = 0; i < matrix_size; ++i) {
    //     sum += std::abs(reference_solution[i]) * std::abs(reference_solution[i]);
    //     diff_split += std::abs(reference_solution[i] - test_solution_split[i]) * std::abs(reference_solution[i] - test_solution_split[i]);
    // }
    // if(rank == 0){
    //     std::cout << " relative error split sparse5 " << std::sqrt(diff_split / sum) << std::endl; 
    // }
    

    // // rank zero should print average time
    // double average_time_tot = 0;
    // double average_time_split1 = 0;
    // double average_time_split2 = 0;
    // double average_time_split3 = 0;
    // double average_time_split4 = 0;
    // double average_time_split5 = 0;
    // double average_time_split6 = 0;
    // double average_time_split_sparse1 = 0;
    // double average_time_split_sparse2 = 0;
    // double average_time_split_sparse3 = 0;
    // double average_time_split_sparse4 = 0;
    // double average_time_split_sparse5 = 0;
    // for(int i = 1; i < number_of_measurements; i++){
    //     average_time_tot += time_tot[i];
    //     average_time_split1 += time_split1[i];
    //     average_time_split2 += time_split2[i];
    //     average_time_split3 += time_split3[i];
    //     average_time_split4 += time_split4[i];
    //     average_time_split5 += time_split5[i];
    //     average_time_split6 += time_split6[i];
    //     average_time_split_sparse1 += time_split_sparse1[i];
    //     average_time_split_sparse2 += time_split_sparse2[i];
    //     average_time_split_sparse3 += time_split_sparse3[i];
    //     average_time_split_sparse4 += time_split_sparse4[i];
    //     average_time_split_sparse5 += time_split_sparse5[i];
    // }
    // // if(rank == 0){
    // //     std::cout << "average time tot " << average_time_tot / (number_of_measurements-1) << std::endl;
    // //     std::cout << "average time split " << average_time_split1 / (number_of_measurements-1) << std::endl;
    // //     std::cout << "average time split " << average_time_split2 / (number_of_measurements-1) << std::endl;
    // //     std::cout << "average time split " << average_time_split3 / (number_of_measurements-1) << std::endl;
    // //     std::cout << "average time split " << average_time_split4 / (number_of_measurements-1) << std::endl;
    // //     std::cout << "average time split " << average_time_split5 / (number_of_measurements-1) << std::endl;
    // //     std::cout << "average time split " << average_time_split6 / (number_of_measurements-1) << std::endl;
    // //     std::cout << "average time split sparse " << average_time_split_sparse1 / (number_of_measurements-1) << std::endl;
    // //     std::cout << "average time split sparse " << average_time_split_sparse2 / (number_of_measurements-1) << std::endl;
    // //     std::cout << "average time split sparse " << average_time_split_sparse3 / (number_of_measurements-1) << std::endl;
    // //     std::cout << "average time split sparse " << average_time_split_sparse4 / (number_of_measurements-1) << std::endl;
    // //     std::cout << "average time split sparse " << average_time_split_sparse5 / (number_of_measurements-1) << std::endl;
    // // }

    // hipFree(diag_inv_d);
    // delete[] diag_inv_h;
    // delete[] data_sparse;
    // delete[] row_ptr_sparse;
    // delete[] col_indices_sparse;
    // delete[] data_tot;
    // delete[] row_ptr_tot;
    // delete[] col_indices_tot;
    // delete[] reference_solution;
    // delete[] rhs;
    // delete[] starting_guess;
    // delete[] dense_subblock_indices;
    // delete[] dense_subblock_data;
    // delete[] diag;
    // // delete[] dense_tot;
    // // delete[] dense_split;
    std::string save_path = "/scratch/project_465000929/maederal/measurement_100_split/";
    std::string path_solve_tot = get_filename(save_path, "solve_tot", 0, size, rank);
    std::string path_solve_split1 = get_filename(save_path, "solve_split1", 0, size, rank);
    std::string path_solve_split2 = get_filename(save_path, "solve_split2", 0, size, rank);
    std::string path_solve_split3 = get_filename(save_path, "solve_split3", 0, size, rank);
    std::string path_solve_split4 = get_filename(save_path, "solve_split4", 0, size, rank);
    std::string path_solve_split5 = get_filename(save_path, "solve_split5", 0, size, rank);
    std::string path_solve_split6 = get_filename(save_path, "solve_split6", 0, size, rank);
    std::string path_solve_split_sparse1 = get_filename(save_path, "solve_split_sparse1", 0, size, rank);
    std::string path_solve_split_sparse2 = get_filename(save_path, "solve_split_sparse2", 0, size, rank);
    std::string path_solve_split_sparse3 = get_filename(save_path, "solve_split_sparse3", 0, size, rank);
    std::string path_solve_split_sparse4 = get_filename(save_path, "solve_split_sparse4", 0, size, rank);
    std::string path_solve_split_sparse5 = get_filename(save_path, "solve_split_sparse5", 0, size, rank);

    // save_measurements(path_solve_tot,
    //     time_tot + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_solve_split1,
    //     time_split1 + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_solve_split2,
    //     time_split2 + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_solve_split3,
    //     time_split3 + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_solve_split4,
    //     time_split4 + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_solve_split5,
    //     time_split5 + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_solve_split6,
    //     time_split6 + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_solve_split_sparse1,
    //     time_split_sparse1 + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_solve_split_sparse2,
    //     time_split_sparse2 + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_solve_split_sparse3,
    //     time_split_sparse3 + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_solve_split_sparse4,
    //     time_split_sparse4 + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_solve_split_sparse5,
    //     time_split_sparse5 + start_up_measurements,
    //     true_number_of_measurements, true);

    MPI_Finalize();
    return 0;
}
