#include "dist_conjugate_gradient.h"
#include "dist_spmv.h"
namespace iterative_solver{

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
void conjugate_gradient_jacobi_split_sparse(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm)
{

    double a, b, na;
    double alpha, alpham1, r0;
    double    r_norm2_h[1];
    double    dot_h[1];

    alpha = 1.0;
    alpham1 = -1.0;
    r0 = 0.0;
    double norm2_rhs = 0;

    double *p_subblock_d;
    double *Ap_subblock_d;
    double *p_subblock_h;
    rocsparse_dnvec_descr vecp_subblock;
    rocsparse_dnvec_descr vecAp_subblock;


    cudaErrchk(hipMalloc((void **)&p_subblock_d,
        A_subblock.subblock_size * sizeof(double)));
    cudaErrchk(hipMalloc((void **)&Ap_subblock_d,
        A_subblock.count_subblock_h[A_distributed.rank] * sizeof(double)));
    
    cudaErrchk(hipHostMalloc((void**)&p_subblock_h, A_subblock.subblock_size * sizeof(double)));
    rocsparse_create_dnvec_descr(&vecp_subblock, A_subblock.subblock_size, p_subblock_d, rocsparse_datatype_f64_r);
    rocsparse_create_dnvec_descr(&vecAp_subblock, A_subblock.count_subblock_h[A_distributed.rank], Ap_subblock_d, rocsparse_datatype_f64_r);


    //copy data to device
    // starting guess for p
    cudaErrchk(hipMemcpy(p_distributed.vec_d[0], x_local_d,
        p_distributed.counts[A_distributed.rank] * sizeof(double), hipMemcpyDeviceToDevice));
    cudaErrchk(hipMemset(A_distributed.Ap_local_d, 0, A_distributed.rows_this_rank * sizeof(double)));

    //begin CG
    // norm of rhs for convergence check
    
    cublasErrchk(hipblasDdot(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, &norm2_rhs));
    MPI_Allreduce(MPI_IN_PLACE, &norm2_rhs, 1, MPI_DOUBLE, MPI_SUM, comm);

    // A*x0
    distributed_spmv_split_sparse(
        A_subblock,
        A_distributed,
        p_subblock_d,
        p_subblock_h,
        vecp_subblock,
        p_distributed,
        Ap_subblock_d,
        vecAp_subblock,
        A_distributed.vecAp_local,
        A_distributed.Ap_local_d,
        A_distributed.default_stream,
        A_distributed.default_rocsparseHandle
    );


    // cal residual r0 = b - A*x0
    // r_norm2_h = r0*r0
    cublasErrchk(hipblasDaxpy(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, &alpham1, A_distributed.Ap_local_d, 1, r_local_d, 1));
    
    // Mz = r
    elementwise_vector_vector(
        r_local_d,
        diag_inv_local_d,
        A_distributed.z_local_d,
        A_distributed.rows_this_rank,
        A_distributed.default_stream
    ); 
    
    // double r_norm2_true;
    cublasErrchk(hipblasDdot(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, A_distributed.z_local_d, 1, r_norm2_h));
    MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);

    // cublasErrchk(hipblasDdot(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, &r_norm2_true));
    // MPI_Allreduce(MPI_IN_PLACE, &r_norm2_true, 1, MPI_DOUBLE, MPI_SUM, comm);

    int k = 1;
    while (r_norm2_h[0]/norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations) {
        if(k > 1){
            // pk+1 = rk+1 + b*pk
            b = r_norm2_h[0] / r0;
            cublasErrchk(hipblasDscal(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, &b, p_distributed.vec_d[0], 1));
            cublasErrchk(hipblasDaxpy(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, &alpha, A_distributed.z_local_d, 1, p_distributed.vec_d[0], 1)); 
        }
        else {
            // p0 = r0
            cublasErrchk(hipblasDcopy(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, A_distributed.z_local_d, 1, p_distributed.vec_d[0], 1));
        }


        // ak = rk^T * rk / pk^T * A * pk
        // has to be done for k=0 if x0 != 0
        distributed_spmv_split_sparse(
            A_subblock,
            A_distributed,
            p_subblock_d,
            p_subblock_h,
            vecp_subblock,
            p_distributed,
            Ap_subblock_d,
            vecAp_subblock,
            A_distributed.vecAp_local,
            A_distributed.Ap_local_d,
            A_distributed.default_stream,
            A_distributed.default_rocsparseHandle
        );

        cublasErrchk(hipblasDdot(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, p_distributed.vec_d[0], 1, A_distributed.Ap_local_d, 1, dot_h));
        MPI_Allreduce(MPI_IN_PLACE, dot_h, 1, MPI_DOUBLE, MPI_SUM, comm);

        a = r_norm2_h[0] / dot_h[0];

        // xk+1 = xk + ak * pk
        cublasErrchk(hipblasDaxpy(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, &a, p_distributed.vec_d[0], 1, x_local_d, 1));

        // rk+1 = rk - ak * A * pk
        na = -a;
        cublasErrchk(hipblasDaxpy(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, &na, A_distributed.Ap_local_d, 1, r_local_d, 1));
        r0 = r_norm2_h[0];

        // Mz = r
        elementwise_vector_vector(
            r_local_d,
            diag_inv_local_d,
            A_distributed.z_local_d,
            A_distributed.rows_this_rank,
            A_distributed.default_stream
        ); 
        

        // r_norm2_h = r0*r0
        cublasErrchk(hipblasDdot(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, A_distributed.z_local_d, 1, r_norm2_h));
        MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);

        // cublasErrchk(hipblasDdot(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, &r_norm2_true));
        // MPI_Allreduce(MPI_IN_PLACE, &r_norm2_true, 1, MPI_DOUBLE, MPI_SUM, comm);

        k++;
    }

    //end CG
    cudaErrchk(hipDeviceSynchronize());
    if(A_distributed.rank == 0){
        std::cout << "iteration (T) = " << k << ", relative residual = " << sqrt(r_norm2_h[0]/norm2_rhs) << std::endl;
    }

    cudaErrchk(hipFree(p_subblock_d));
    cusparseErrchk(hipsparseDestroyDnVec(vecp_subblock));
    cudaErrchk(hipFree(Ap_subblock_d));
    cusparseErrchk(hipsparseDestroyDnVec(vecAp_subblock));
    cudaErrchk(hipHostFree(p_subblock_h));

}
template 
void conjugate_gradient_jacobi_split_sparse<dspmv_split_sparse::spmm_split_sparse1>(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template
void conjugate_gradient_jacobi_split_sparse<dspmv_split_sparse::spmm_split_sparse2>(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template
void conjugate_gradient_jacobi_split_sparse<dspmv_split_sparse::spmm_split_sparse3>(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);

} // namespace iterative_solver