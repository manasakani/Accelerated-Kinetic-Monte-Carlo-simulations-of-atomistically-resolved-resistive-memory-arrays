#include "dist_spmv.h"

namespace dspmv_split_sparse{

void spmm_split_sparse1(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    cusparseDnVecDescr_t &vecp_subblock,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    cusparseDnVecDescr_t &vecAp_subblock,
    cusparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    cudaStream_t &default_stream,
    cusparseHandle_t &default_cusparseHandle)
{
    // Isend Irecv subblock
    // sparse part
    //gemv

    int rank = A_distributed.rank;
    int size = A_distributed.size;

    double alpha = 1.0;
    double beta = 0.0;

    // pack dense sublblock p
    pack_gpu(p_subblock_d + A_subblock.displ_subblock_h[rank],
        p_distributed.vec_d[0],
        A_subblock.subblock_indices_local_d,
        A_subblock.count_subblock_h[rank],
        default_stream);

    if(size > 1){
        cudaErrchk(cudaMemcpy(p_subblock_h + A_subblock.displ_subblock_h[rank],
            p_subblock_d + A_subblock.displ_subblock_h[rank],
            A_subblock.count_subblock_h[rank] * sizeof(double), cudaMemcpyDeviceToHost));
        for(int i = 0; i < size-1; i++){
            int dest = (rank + 1 + i) % size;
            MPI_Isend(p_subblock_h + A_subblock.displ_subblock_h[rank], A_subblock.count_subblock_h[rank],
                MPI_DOUBLE, dest, dest, A_distributed.comm, &A_subblock.send_subblock_requests[i]);
        }
        for(int i = 0; i < size-1; i++){
            int source = (rank + 1 + i) % size;
            MPI_Irecv(p_subblock_h + A_subblock.displ_subblock_h[source], A_subblock.count_subblock_h[source],
                MPI_DOUBLE, source, rank, A_distributed.comm, &A_subblock.recv_subblock_requests[i]);
        }
    }

    dspmv::gpu_packing(
        A_distributed,
        p_distributed,
        vecAp_local,
        default_stream,
        default_cusparseHandle
    );
    if(size > 1){
        MPI_Waitall(size-1, A_subblock.recv_subblock_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(size-1, A_subblock.send_subblock_requests, MPI_STATUSES_IGNORE);
        // recv whole vector
        cudaErrchk(cudaMemcpyAsync(p_subblock_d,
            p_subblock_h, A_subblock.subblock_size * sizeof(double),
            cudaMemcpyHostToDevice, default_stream));
    }

    cusparseErrchk(cusparseSpMV(
        default_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
        *A_subblock.descriptor, vecp_subblock,
        &beta, vecAp_subblock, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, A_subblock.buffer_d));

    // unpack and add it to Ap
    unpack_add(
        Ap_local_d,
        Ap_subblock_d,
        A_subblock.subblock_indices_local_d,
        A_subblock.count_subblock_h[rank],
        default_stream
    );
}

} // namespace dspmv_split_sparse