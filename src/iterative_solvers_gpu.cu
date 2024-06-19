#include "gpu_solvers.h"
#include <cub/cub.cuh>

// Sparse matrix assembly functions and iterative solvers

#define NUM_THREADS 512

// used to be named 'calc_diagonal_A_gpu'
__global__ void reduce_rows_into_diag( 
    int *col_indices,
    int *row_ptr,
    double *data,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        //reduce the elements in the row
        double tmp = 0.0;
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i != col_indices[j]){
                tmp += data[j];
            }
        }
        //write the diagonal element
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i == col_indices[j]){
                data[j] = -tmp;
            }
        }
    }
}


// used to be called 'set_diag'
__global__ void write_to_diag(double *A, double *diag, int N)
{
    int didx = blockIdx.x * blockDim.x + threadIdx.x;
    if (didx < N)
    {
        double tmp = A[didx * N + didx];
        A[didx * N + didx] = 2 * tmp - diag[didx];
    }
}

// sum the rows of A into the vector diag
// with an explicit instantiation for the linker
template __global__ void row_reduce<NUM_THREADS>(double *A, double *diag, int N); 

template <int NTHREADS>
__global__ void row_reduce(double *A, double *diag, int N)
{

    int num_threads = blockDim.x;
    int blocks_per_row = (N - 1) / num_threads + 1;
    int block_id = blockIdx.x;

    int tid = threadIdx.x;

    __shared__ double buf[NTHREADS];

    for (auto idx = block_id; idx < N * blocks_per_row; idx += gridDim.x)
    {

        int ridx = idx / blocks_per_row;
        int scol = (idx % blocks_per_row) * num_threads;
        int lcol = min(N, scol + num_threads);

        buf[tid] = 0.0;
        if (tid + scol < lcol)
        {
            buf[tid] = A[ridx * N + scol + tid];
        }

        int width = num_threads / 2;
        while (width != 0)
        {
            __syncthreads();
            if (tid < width)
            {
                buf[tid] += buf[tid + width];
            }
            width /= 2;
        }

        if (tid == 0)
        {
            atomicAdd(diag + ridx, buf[0]);
        }
    }
}


__global__ void determine_neighbor_nnz(
    const double *posx_i_d, const double *posy_i_d, const double *posz_i_d,
    const double *posx_j_d, const double *posy_j_d, const double *posz_j_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    long int size_i,
    long int size_j,
    int *dist_nnz_d,
    int *dist_nnz_per_row_d
){
    // this rank has i sites
    // other rank has j sites
    long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(long long int id = idx; id < size_i * size_j; id += blockDim.x * gridDim.x){
        int i = id / size_j;
        int j = id % size_j;
        if(i < size_i && j < size_j){
            double dist = site_dist_gpu(posx_i_d[i], posy_i_d[i], posz_i_d[i],
                                        posx_j_d[j], posy_j_d[j], posz_j_d[j],
                                        lattice_d[0], lattice_d[1], lattice_d[2], pbc);
            if(dist < cutoff_radius){
                atomicAdd(dist_nnz_d, 1);
                atomicAdd(dist_nnz_per_row_d + i, 1);
            }
        }
    }
}

template <typename T>
void writeArrayToBinFile(T* array, int numElements, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<char*>(array), numElements*sizeof(T));
        file.close();
        std::cout << "Array data written to file: " << filename << std::endl;
    } else {
        std::cerr << "Unable to open the file for writing." << std::endl;
    }
}


// check that sparse and dense versions are the same
void check_sparse_dense_match(int m, int nnz, double *dense_matrix, int* d_csrRowPtr, int* d_csrColInd, double* d_csrVal){
    
    double *h_D = (double *)calloc(m*m, sizeof(double));
    double *h_D_csr = (double *)calloc(nnz, sizeof(double));
    int *h_pointers = (int *)calloc((m + 1), sizeof(int));
    int *h_inds = (int *)calloc(nnz, sizeof(int));

    gpuErrchk( cudaMemcpy(h_D, dense_matrix, m*m * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_D_csr, d_csrVal, nnz * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_pointers, d_csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_inds, d_csrColInd, nnz * sizeof(int), cudaMemcpyDeviceToHost) );

    int nnz_count = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < m; col++) {
            int i = row * m + col;  // Linear index in dense matrix
            // Check if the element in the dense matrix is non-zero
            if (h_D[i] != 0) {
                // Compare the row and column indices
                if (h_D[i] != h_D_csr[nnz_count] || col != h_inds[nnz_count]) {
                    std::cout << "Mismatch found at (row, col) = (" << row << ", " << col << ")\n";
                }
                nnz_count++;
            }
        }
    }
}

// dump sparse matrix into a file
void dump_csr_matrix_txt(int m, int nnz, int* d_csrRowPtr, int* d_csrColIndices, double* d_csrValues, int kmc_step_count){

    // Copy matrix back to host memory
    double *h_csrValues = (double *)calloc(nnz, sizeof(double));
    int *h_csrRowPtr = (int *)calloc((m + 1), sizeof(int));
    int *h_csrColIndices = (int *)calloc(nnz, sizeof(int));
    gpuErrchk( cudaMemcpy(h_csrRowPtr, d_csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_csrColIndices, d_csrColIndices, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_csrValues, d_csrValues, nnz * sizeof(double), cudaMemcpyDeviceToHost) );

    // print to file, tagged with the kmc step number
    std::ofstream fout_val("csrValues_step#" + std::to_string(kmc_step_count) + ".txt");
    for(int i = 0; i < nnz; i++){
        fout_val << h_csrValues[i] << " "; 
    }
    std::ofstream fout_row("csrRowPtr_step#" + std::to_string(kmc_step_count) + ".txt");
    for(int i = 0; i < (m + 1); i++){
        fout_row << h_csrRowPtr[i] << " "; 
    }
    std::ofstream fout_col("csrColIndices_step#" + std::to_string(kmc_step_count) + ".txt");
    for(int i = 0; i < nnz; i++){
        fout_col << h_csrColIndices[i] << " "; 
    }

    free(h_csrValues);
    free(h_csrRowPtr);
    free(h_csrColIndices);
}

// Solution of A*x = y using cusolver in host pointer mode
void sparse_system_solve(cusolverSpHandle_t handle, int* d_csrRowPtr, int* d_csrColInd, double* d_csrVal,
                         int nnz, int m, double *d_x, double *d_y){

    // Ref: https://stackoverflow.com/questions/31840341/solving-general-sparse-linear-systems-in-cuda

    // cusolverSpDcsrlsvlu only supports the host path
    int *h_A_RowIndices = (int *)malloc((m + 1) * sizeof(int));
    int *h_A_ColIndices = (int *)malloc(nnz * sizeof(int));
    double *h_A_Val = (double *)malloc(nnz * sizeof(double));
    double *h_x = (double *)malloc(m * sizeof(double));
    double *h_y = (double *)malloc(m * sizeof(double));
    gpuErrchk( cudaMemcpy(h_A_RowIndices, d_csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_A_ColIndices, d_csrColInd, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_A_Val, d_csrVal, nnz * sizeof(double), cudaMemcpyDeviceToHost) );   
    gpuErrchk( cudaMemcpy(h_x, d_x, m * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_y, d_y, m * sizeof(double), cudaMemcpyDeviceToHost) );

    cusparseMatDescr_t matDescrA;
    cusparseCreateMatDescr(&matDescrA);
    cusparseSetMatType(matDescrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(matDescrA, CUSPARSE_INDEX_BASE_ZERO);

    int singularity;
    double tol = 0.00000001;

    // Solve with LU
    // CheckCusolverDnError( cusolverSpDcsrlsvluHost(handle, m, nnz, matDescrA, h_A_Val, h_A_RowIndices, 
    //                       h_A_ColIndices, h_y, tol, 0, h_x, &singularity) );
    
    // Solve with QR
    // CheckCusolverDnError( cusolverSpDcsrlsvqrHost(handle, m, nnz, matDescrA, h_A_Val, h_A_RowIndices, 
    //                       h_A_ColIndices, h_y, tol, 1, h_x, &singularity) );

    // Solve with Cholesky
    CheckCusolverDnError( cusolverSpDcsrlsvcholHost(handle, m, nnz, matDescrA, h_A_Val, h_A_RowIndices,
                          h_A_ColIndices, h_y, tol, 1, h_x, &singularity) );

    gpuErrchk( cudaDeviceSynchronize() );
    if (singularity != -1){
        std::cout << "In sparse_system_solve: Matrix has a singularity at : " << singularity << "\n";
    }

    // copy back the solution vector:
    gpuErrchk( cudaMemcpy(d_x, h_x, m * sizeof(double), cudaMemcpyHostToDevice) );

    cusolverSpDestroy(handle);
    cusparseDestroyMatDescr(matDescrA);
    free(h_A_RowIndices);
    free(h_A_ColIndices);
    free(h_A_Val);
    free(h_x);
    free(h_y);
}

// Extracts the inverse sqrt of the diagonal values into a vector to use for the preconditioning
__global__ void computeDiagonalInvSqrt(const double* A_data, const int* A_row_ptr,
                                       const int* A_col_indices, double* diagonal_values_inv_sqrt_d,
                                       const int matrix_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < matrix_size) {
        // Find the range of non-zero elements for the current row
        int row_start = A_row_ptr[tid];
        int row_end = A_row_ptr[tid + 1];

        // Initialize the sum for the diagonal element
        double diagonal_sum = 0.0;

        // Loop through the non-zero elements in the current row
        for (int i = row_start; i < row_end; ++i) {
            if (A_col_indices[i] == tid) {
                // Found the diagonal element
                diagonal_sum = A_data[i];
                break;
            }
        }

        double diagonal_inv_sqrt = 1.0 / sqrt(diagonal_sum);
        // double diagonal_inv_sqrt = 1.0 / sqrt(abs(diagonal_sum));

        // Store the result in the output array
        diagonal_values_inv_sqrt_d[tid] = diagonal_inv_sqrt;
    }
}

// apply Jacobi preconditioner to an rhs vector
__global__ void jacobi_precondition_array(
    double *array,
    double *diagonal_values_inv_sqrt,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        array[i] = array[i] * diagonal_values_inv_sqrt[i];
    }

}

// apply Jacobi preconditioner to matrix
__global__ void jacobi_precondition_matrix(
    double *data,
    const int *col_indices,
    const int *row_indptr,
    double *diagonal_values_inv_sqrt,
    int matrix_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        // Iterate over the row elements
        for(int j = row_indptr[i]; j < row_indptr[i+1]; j++){
            // Use temporary variables to store the original values
            double original_value = data[j];

            // Update data with the preconditioned value
            data[j] = original_value * diagonal_values_inv_sqrt[i] * diagonal_values_inv_sqrt[col_indices[j]];
        }
    }
}

// apply Jacobi preconditioner to starting guess
__global__ void jacobi_unprecondition_array(
    double *array,
    double *diagonal_values_inv_sqrt,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        array[i] = array[i] * 1/diagonal_values_inv_sqrt[i];
    }

}

// Iterative sparse linear solver using CG steps
void solve_sparse_CG_Jacobi(cublasHandle_t handle_cublas, cusparseHandle_t handle, 
							double* A_data, int* A_row_ptr,
                            int* A_col_indices, const int A_nnz, int m, double *d_x, double *d_y, double tol){

    // A is an m x m sparse matrix represented by CSR format
    // - d_x is right hand side vector in gpu memory,
    // - d_y is solution vector in gpu memory.
    // - d_z is intermediate result on gpu memory.

    // Sets the initial guess for the solution vector to zero
    bool zero_guess = 0;    

    // Error tolerance for the norm of the residual in the CG steps
    // double tol = 1e-14;  // make this an input, used to be 1e-12

    double one = 1.0;
    double n_one = -1.0;
    double zero = 0.0;
    double *one_d, *n_one_d, *zero_d;
    gpuErrchk( cudaMalloc((void**)&one_d, sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&n_one_d, sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&zero_d, sizeof(double)) );
    gpuErrchk( cudaMemcpy(one_d, &one, sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(n_one_d, &n_one, sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(zero_d, &zero, sizeof(double), cudaMemcpyHostToDevice) );
    cusparseStatus_t status;

    // ************************************
    // ** Initial Guess **

    if (zero_guess)
    {
        // Set the initial guess for the solution vector to zero
        gpuErrchk( cudaMemset(d_y, 0, m * sizeof(double)) ); 
        gpuErrchk( cudaDeviceSynchronize() );
    }

    // *******************************
    // ** Preconditioner **

    double* diagonal_values_inv_sqrt_d;
    cudaMalloc((void**)&diagonal_values_inv_sqrt_d, sizeof(double) * m);

    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;

    computeDiagonalInvSqrt<<<grid_size, block_size>>>(A_data, A_row_ptr, A_col_indices,
                                                      diagonal_values_inv_sqrt_d, m);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // print diag
    // double *h_diag = (double *)malloc(m * sizeof(double));
    // gpuErrchk( cudaMemcpy(h_diag, diagonal_values_inv_sqrt_d, m * sizeof(double), cudaMemcpyDeviceToHost) );
    // for(int i = 0; i < m; i++){
    //     std::cout << h_diag[i] << " ";
    // }
    // std::cout << "\n";
    // exit(1);

    // scale rhs
    jacobi_precondition_array<<<grid_size, block_size>>>(d_x, diagonal_values_inv_sqrt_d, m);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    // scale matrix
    jacobi_precondition_matrix<<<grid_size, block_size>>>(A_data, A_col_indices, A_row_ptr, 
                                                          diagonal_values_inv_sqrt_d, m);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // scale starting guess
    jacobi_unprecondition_array<<<grid_size, block_size>>>(d_y, diagonal_values_inv_sqrt_d, m);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cusparseSpMatDescr_t matA;
    status = cusparseCreateCsr(&matA, m, m, A_nnz, A_row_ptr, A_col_indices, A_data, 
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: creation of sparse matrix descriptor in solve_sparse_CG_Jacobi() failed!\n";
    }

    // *******************************
    // ** Iterative refinement loop **

    // initialize variables for the residual calculation
    double h_norm;
    double *d_r, *d_p, *d_temp;
    gpuErrchk( cudaMalloc((void**)&d_r, m * sizeof(double)) ); 
    gpuErrchk( cudaMalloc((void**)&d_p, m * sizeof(double)) ); 
    gpuErrchk( cudaMalloc((void**)&d_temp, m * sizeof(double)) ); 

    // for SpMV:
    // - d_x is right hand side vector
    // - d_y is solution vector
    cusparseDnVecDescr_t vecY, vecR, vecP, vectemp; 
    cusparseCreateDnVec(&vecY, m, d_y, CUDA_R_64F);
    cusparseCreateDnVec(&vecR, m, d_r, CUDA_R_64F);
    cusparseCreateDnVec(&vecP, m, d_p, CUDA_R_64F);
    cusparseCreateDnVec(&vectemp, m, d_temp, CUDA_R_64F);

    size_t MVBufferSize;
    void *MVBuffer = 0;
    status = cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                          vecY, zero_d, vectemp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &MVBufferSize);  
    gpuErrchk( cudaMalloc((void**)&MVBuffer, sizeof(double) * MVBufferSize) );

    // Initialize the residual and conjugate vectors
    // r = A*y - x & p = -r
    status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA,                         
                          vecY, zero_d, vecR, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, MVBuffer);           // r = A*y
    CheckCublasError( cublasDaxpy(handle_cublas, m, &n_one, d_x, 1, d_r, 1) );                            // r = -x + r
    CheckCublasError( cublasDcopy(handle_cublas, m, d_r, 1, d_p, 1) );                                    // p = r
    CheckCublasError( cublasDscal(handle_cublas, m, &n_one, d_p, 1) );                                    // p = -p

    // calculate the error (norm of the residual)
    CheckCublasError( cublasDnrm2(handle_cublas, m, d_r, 1, &h_norm) );
    gpuErrchk( cudaDeviceSynchronize() );
    
    // Conjugate Gradient steps
    int counter = 0;
    double t, tnew, alpha, beta, alpha_temp;
    while (h_norm > tol*tol){

        // alpha = rT * r / (pT * A * p)
        CheckCublasError( cublasDdot (handle_cublas, m, d_r, 1, d_r, 1, &t) );                           // t = rT * r
        status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                              vecP, zero_d, vectemp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, MVBuffer);   // temp = A*p
        CheckCublasError( cublasDdot (handle_cublas, m, d_p, 1, d_temp, 1, &alpha_temp) );               // alpha = pT*temp = pT*A*p
        alpha = t / alpha_temp; 

        // y = y + alpha * p
        CheckCublasError( cublasDaxpy(handle_cublas, m, &alpha, d_p, 1, d_y, 1) );                       // y = y + alpha * p

        // r = r + alpha * A * p 
        CheckCublasError( cublasDaxpy(handle_cublas, m, &alpha, d_temp, 1, d_r, 1) );                    // r = r + alpha * temp

        // beta = (rT * r) / t
        CheckCublasError( cublasDdot (handle_cublas, m, d_r, 1, d_r, 1, &tnew) );                        // tnew = rT * r
        beta = tnew / t;

        // p = -r + beta * p
        CheckCublasError( cublasDscal(handle_cublas, m, &beta, d_p, 1) );                                 // p = p * beta
        CheckCublasError( cublasDaxpy(handle_cublas, m, &n_one, d_r, 1, d_p, 1) );                        // p = p - r

        // calculate the error (norm of the residual)
        CheckCublasError( cublasDdot(handle_cublas, m, d_r, 1, d_r, 1, &h_norm) );
        // std::cout << h_norm << "\n";

        counter++;
        if (counter > 50000){
            std::cout << "WARNING: might be stuck in diverging CG iterations, check the residual!\n";
        }
    }
    std::cout << "# CG steps T: " << counter << "\n";

    // unprecondition the solution vector
    jacobi_precondition_array<<<grid_size, block_size>>>(d_y, diagonal_values_inv_sqrt_d, m);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // // check solution vector
    // double *copy_back = (double *)calloc(m, sizeof(double));
    // gpuErrchk( cudaMemcpy(copy_back, d_y, m * sizeof(double), cudaMemcpyDeviceToHost) );
    // for (int i = 0; i < m; i++){
    //     std::cout << copy_back[i] << " ";
    // }
    // std::cout << "\nPrinted solution vector, now exiting\n";
    // exit(1);

    cudaFree(diagonal_values_inv_sqrt_d);
    cudaFree(MVBuffer); 
    cudaFree(one_d);
    cudaFree(n_one_d);
    cudaFree(zero_d);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_temp);
}

// Iterative sparse linear solver using CG steps
void solve_sparse_CG(cublasHandle_t handle_cublas, cusparseHandle_t handle, 
					 double* A_data, int* A_row_ptr, int* A_col_indices, const int A_nnz, 
                     int m, double *d_x, double *d_y){

    // A is an m x m sparse matrix represented by CSR format
    // - d_x is right hand side vector in gpu memory,
    // - d_y is solution vector in gpu memory.
    // - d_z is intermediate result on gpu memory.

    // Sets the initial guess for the solution vector to zero
    bool zero_guess = 0;

    // Error tolerance for the norm of the residual in the CG steps
    double tol = 1e-10*m;

    double one = 1.0;
    double n_one = -1.0;
    double zero = 0.0;
    double *one_d, *n_one_d, *zero_d;
    gpuErrchk( cudaMalloc((void**)&one_d, sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&n_one_d, sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&zero_d, sizeof(double)) );
    gpuErrchk( cudaMemcpy(one_d, &one, sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(n_one_d, &n_one, sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(zero_d, &zero, sizeof(double), cudaMemcpyHostToDevice) );
    cusparseStatus_t status;

    cusparseSpMatDescr_t matA;
    status = cusparseCreateCsr(&matA, m, m, A_nnz, A_row_ptr, A_col_indices, A_data, 
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: creation of sparse matrix descriptor in solve_sparse_CG_Jacobi() failed!\n";
    }

    // ************************************
    // ** Set Initial Guess **

    if (zero_guess)
    {
        // Set the initial guess for the solution vector to zero
        gpuErrchk( cudaMemset(d_y, 0, m * sizeof(double)) ); 
        gpuErrchk( cudaDeviceSynchronize() );
    }

    // *******************************
    // ** Iterative refinement loop **

    // initialize variables for the residual calculation
    double h_norm;
    double *d_r, *d_p, *d_temp;
    gpuErrchk( cudaMalloc((void**)&d_r, m * sizeof(double)) ); 
    gpuErrchk( cudaMalloc((void**)&d_p, m * sizeof(double)) ); 
    gpuErrchk( cudaMalloc((void**)&d_temp, m * sizeof(double)) ); 

    // for SpMV:
    // - d_x is right hand side vector
    // - d_y is solution vector
    cusparseDnVecDescr_t vecY, vecR, vecP, vectemp; 
    cusparseCreateDnVec(&vecY, m, d_y, CUDA_R_64F);
    cusparseCreateDnVec(&vecR, m, d_r, CUDA_R_64F);
    cusparseCreateDnVec(&vecP, m, d_p, CUDA_R_64F);
    cusparseCreateDnVec(&vectemp, m, d_temp, CUDA_R_64F);

    size_t MVBufferSize;
    void *MVBuffer = 0;
    status = cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                          vecY, zero_d, vectemp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &MVBufferSize);
    gpuErrchk( cudaMalloc((void**)&MVBuffer, sizeof(double) * MVBufferSize) );
    
    // Initialize the residual and conjugate vectors
    // r = A*y - x & p = -r
    status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                          vecY, zero_d, vecR, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, MVBuffer);         // r = A*y
    //gpuErrchk( cudaDeviceSynchronize() );
    CheckCublasError( cublasDaxpy(handle_cublas, m, &n_one, d_x, 1, d_r, 1) );                          // r = -x + r
    //gpuErrchk( cudaDeviceSynchronize() );
    CheckCublasError(cublasDcopy(handle_cublas, m, d_r, 1, d_p, 1));                                    // p = r
    //gpuErrchk( cudaDeviceSynchronize() );
    CheckCublasError(cublasDscal(handle_cublas, m, &n_one, d_p, 1));                                    // p = -p
    //gpuErrchk( cudaDeviceSynchronize() );

    // calculate the error (norm of the residual)
    CheckCublasError( cublasDnrm2(handle_cublas, m, d_r, 1, &h_norm) );
    gpuErrchk( cudaDeviceSynchronize() );
    
    // Conjugate Gradient steps
    int counter = 0;
    double t, tnew, alpha, beta, alpha_temp;
    while (h_norm > tol){

        // alpha = rT * r / (pT * A * p)
        CheckCublasError( cublasDdot (handle_cublas, m, d_r, 1, d_r, 1, &t) );                         // t = rT * r
        //gpuErrchk( cudaDeviceSynchronize() );
        status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                              vecP, zero_d, vectemp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, MVBuffer); // temp = A*p
        //gpuErrchk( cudaDeviceSynchronize() );
        CheckCublasError( cublasDdot (handle_cublas, m, d_p, 1, d_temp, 1, &alpha_temp) );             // alpha = pT*temp = pT*A*p
        //gpuErrchk( cudaDeviceSynchronize() );
        alpha = t / alpha_temp; 

        // y = y + alpha * p
        CheckCublasError(cublasDaxpy(handle_cublas, m, &alpha, d_p, 1, d_y, 1));                       // y = y + alpha * p
        //gpuErrchk( cudaDeviceSynchronize() );

        // r = r + alpha * A * p 
        CheckCublasError(cublasDaxpy(handle_cublas, m, &alpha, d_temp, 1, d_r, 1));                    // r = r + alpha * temp
        //gpuErrchk( cudaDeviceSynchronize() );

        // beta = (rT * r) / t
        CheckCublasError( cublasDdot (handle_cublas, m, d_r, 1, d_r, 1, &tnew) );                       // tnew = rT * r
        //gpuErrchk( cudaDeviceSynchronize() );
        beta = tnew / t;

        // p = -r + beta * p
        CheckCublasError(cublasDscal(handle_cublas, m, &beta, d_p, 1));                                  // p = p * beta
        //gpuErrchk( cudaDeviceSynchronize() );
        CheckCublasError(cublasDaxpy(handle_cublas, m, &n_one, d_r, 1, d_p, 1));                         // p = p - r
        //gpuErrchk( cudaDeviceSynchronize() );

        // calculate the error (norm of the residual)
        CheckCublasError( cublasDnrm2(handle_cublas, m, d_r, 1, &h_norm) );
        //gpuErrchk( cudaDeviceSynchronize() );
        // std::cout << h_norm << "\n";

        counter++;
        if (counter > 50000){
            std::cout << "WARNING: might be stuck in diverging CG iterations, check the residual!\n";
        }
    }
    std::cout << "# CG steps: " << counter << "\n";

    cudaFree(MVBuffer); 
    cudaFree(one_d);
    cudaFree(n_one_d);
    cudaFree(zero_d);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_temp);

    // // check solution vector
    // double *copy_back = (double *)calloc(m, sizeof(double));
    // gpuErrchk( cudaMemcpy(copy_back, d_y, m * sizeof(double), cudaMemcpyDeviceToHost) );
    // for (int i = 0; i < m; i++){
    //     std::cout << copy_back[i] << " ";
    // }
    // exit(1);
    
}

// helper function for the splitmatrix CG implementation, multiplies the compressed submatrix with the solution vector
__global__ void add_submatrix_product(double *M, double *y, double *r, int msub, int *insertion_indices) {

    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    // Each thread gets a row of the matrix
    for (int idx = tid_total; idx < msub; idx += num_threads_total)
    {   
        // it needs to multiply it's row by staggered elements of y
        double row_sum = 0.0;
        for ( int j = 0; j < msub; j++ )
        {
            row_sum += M[idx * msub + j] * y[insertion_indices[j] + 2];
        }

        // add this row_sum to index insertion_indices[idx] + 2 of the output vector
        r[insertion_indices[idx] + 2] += row_sum;
    }
}

//remove this as it exists already
__global__ void elementwise_vector_vector_tmp(
    double * __restrict__ array1,
    double * __restrict__ array2,
    double * __restrict__ result,
    int size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < size; i += blockDim.x * gridDim.x){
        result[i] = array1[i] * array2[i];
    }

}

// Iterative sparse linear solver using CG steps on matrix represented in mixed sparse/dense format 
// the insertion indices specify the correspondence between the (dense) submatrix rows/cols and the (sparse) full matrix rows/cols
void solve_sparse_CG_splitmatrix(cublasHandle_t handle_cublas, cusparseHandle_t handle, 
                                 double* M, int msub, double* A_data, int* A_row_ptr, int* A_col_indices, const int A_nnz, 
                                 int m, int *insertion_indices, double *d_x, double *d_y,
                                 double *diagonal_inv_d){
    // A is an m x m sparse matrix in CSR format
    // M is an msub x msub dense matrix
    // the full system matrix is 

    // - d_x is right hand side vector in gpu memory,
    // - d_y is solution vector in gpu memory.
    // - d_z is intermediate result on gpu memory.

    // Sets the initial guess for the solution vector to zero
    bool zero_guess = 0;

    // Error tolerance for the norm of the residual in the CG steps
    double tol = 1e-5;//1e-12;

    double one = 1.0;
    double n_one = -1.0;
    double zero = 0.0;
    double *one_d, *n_one_d, *zero_d;
    gpuErrchk( cudaMalloc((void**)&one_d, sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&n_one_d, sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&zero_d, sizeof(double)) );
    gpuErrchk( cudaMemcpy(one_d, &one, sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(n_one_d, &n_one, sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(zero_d, &zero, sizeof(double), cudaMemcpyHostToDevice) );
    cusparseStatus_t status;

    cusparseSpMatDescr_t matA;
    status = cusparseCreateCsr(&matA, m, m, A_nnz, A_row_ptr, A_col_indices, A_data, 
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: creation of sparse matrix descriptor in solve_sparse_CG_splitmatrix() failed!\n";
    }

    // ************************************
    // ** Set Initial Guess **

    if (zero_guess)
    {
        // Set the initial guess for the solution vector to zero
        gpuErrchk( cudaMemset(d_y, 0, m * sizeof(double)) ); 
        gpuErrchk( cudaDeviceSynchronize() );
    }

    // *******************************
    // ** Iterative refinement loop **

    // initialize variables for the residual calculation
    double h_norm;
    double *d_r, *d_p, *d_temp, *d_z;
    gpuErrchk( cudaMalloc((void**)&d_r, m * sizeof(double)) ); 
    gpuErrchk( cudaMalloc((void**)&d_p, m * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_temp, m * sizeof(double)) ); 
    gpuErrchk( cudaMalloc((void**)&d_z, m * sizeof(double)) );
    gpuErrchk(cudaMemcpy(d_r, d_x, m * sizeof(double), cudaMemcpyDeviceToDevice));

    // gpuErrchk(cuda)

    // for SpMV:
    // - d_x is right hand side vector
    // - d_y is solution vector
    cusparseDnVecDescr_t vecY, vecR, vecP, vectemp; 
    cusparseCreateDnVec(&vecY, m, d_y, CUDA_R_64F);
    cusparseCreateDnVec(&vecR, m, d_r, CUDA_R_64F);
    cusparseCreateDnVec(&vecP, m, d_p, CUDA_R_64F);
    cusparseCreateDnVec(&vectemp, m, d_temp, CUDA_R_64F);

    size_t MVBufferSize;
    void *MVBuffer = 0;
    status = cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                          vecY, zero_d, vectemp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &MVBufferSize);
    gpuErrchk( cudaMalloc((void**)&MVBuffer, sizeof(double) * MVBufferSize) );
    
    // Initialize the residual and conjugate vectors
    // r = A*y - x & p = -r

    // r = (A + M)*y = A*y + M*y
    status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                          vecY, zero_d, vectemp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, MVBuffer);         // r = A*y

    // do r += M*y in a single CUDA kernel by multiplying M by the sub-vector of y 
    // specified by in the indices in 'insertion indices', and adding the result to r. 
    int threads = 1024;
    int blocks = (m + threads - 1) / threads;
    add_submatrix_product<<<blocks, threads>>>(M, d_y, d_temp, msub, insertion_indices);                   // r += M*y

    //gpuErrchk( cudaDeviceSynchronize() );
    //CheckCublasError( cublasDaxpy(handle_cublas, m, &n_one, d_x, 1, d_r, 1) );                          // r = -x + r
    
    // r = b - Ax0
    CheckCublasError(cublasDaxpy(handle_cublas, m, &n_one, d_temp, 1, d_r, 1));

    // Mz0 = r0
    elementwise_vector_vector_tmp<<<blocks, threads>>>(
        d_r,
        diagonal_inv_d,
        d_z,
        m
    ); 


    double r1, r0, b, a, na, dot;

    // calculate the error (norm of the residual)
    CheckCublasError(cublasDdot(handle_cublas, m, d_r, 1, d_z, 1, &r1));

    int k = 1;
    while (r1 > tol * tol && k <= 1000) {
        
        if(k > 1){
            b = r1 / r0;
            CheckCublasError(cublasDscal(handle_cublas, m, &b, d_p, 1));

            CheckCublasError(cublasDaxpy(handle_cublas, m, &one, d_z, 1, d_p, 1));   
        }
        else {
            CheckCublasError(cublasDcopy(handle_cublas, m, d_z, 1, d_p, 1));
        }
        // temp = (A + M)*p = A*p + M*p
        status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                              vecP, zero_d, vectemp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, MVBuffer); // temp = A*p
        add_submatrix_product<<<blocks, threads>>>(M, d_p, d_temp, msub, insertion_indices);           // temp += M*p
        CheckCublasError(cublasDdot(handle_cublas, m, d_p, 1, d_temp, 1, &dot));
        a = r1 / dot;
        CheckCublasError(cublasDaxpy(handle_cublas, m, &a, d_p, 1, d_y, 1));
        na = -a;
        CheckCublasError(cublasDaxpy(handle_cublas, m, &na, d_temp, 1, d_r, 1));
        // Mz = r
        elementwise_vector_vector_tmp<<<blocks, threads>>>(
            d_r,
            diagonal_inv_d,
            d_z,
            m
        ); 
        r0 = r1;
        CheckCublasError(cublasDdot(handle_cublas, m, d_r, 1, d_z, 1, &r1));
        // gpuErrchk(cudaStreamSynchronize(stream));
        k++;
    }

    double *h_y = (double *)calloc(m, sizeof(double));
    gpuErrchk( cudaMemcpy(h_y, d_y, m * sizeof(double), cudaMemcpyDeviceToHost) );
    double sum = 0.0;
    for (int i = 0; i < m; i++){
        sum += 1e30*h_y[i]*h_y[i];
    }
    std::cout << "sum of solution vector: " << sum << "\n";

    std::cout << "# CG split steps: " << k << "\n";
    std::cout << "solve_sparse_CG_splitmatrix residual: " << r1 << "\n";

    cudaFree(MVBuffer); 
    cudaFree(one_d);
    cudaFree(n_one_d);
    cudaFree(zero_d);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_temp);
    cudaFree(d_z);

    // // check solution vector
    // double *copy_back = (double *)calloc(m, sizeof(double));
    // gpuErrchk( cudaMemcpy(copy_back, d_y, m * sizeof(double), cudaMemcpyDeviceToHost) );
    // for (int i = 0; i < m; i++){
    //     std::cout << copy_back[i] << " ";
    // }
    // std::cout << "exiting after printing the solution vector\n";
    // exit(1);
    
}

// Function to convert dense matrix to CSR format using cuSPARSE
void denseToCSR(cusparseHandle_t handle, double* d_dense, int num_rows, int num_cols,
                double** d_csr_values, int** d_csr_offsets, int** d_csr_columns, int* total_nnz)
{
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    int                          ld = num_cols;

    // Create dense matrix A
    cusparseCreateDnMat(&matA, num_rows, num_cols, ld, d_dense,
                        CUDA_R_64F, CUSPARSE_ORDER_ROW);

    // Create sparse matrix B in CSR format
    cusparseCreateCsr(&matB, num_rows, num_cols, 0,
                      *d_csr_offsets, NULL, NULL,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    // allocate an external buffer if needed
    cusparseDenseToSparse_bufferSize(handle, matA, matB,
                                     CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                     &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // execute Sparse to Dense conversion
    cusparseDenseToSparse_analysis(handle, matA, matB,
                                   CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                   dBuffer);
                            
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp, &nnz); 
    *total_nnz = static_cast<int>(nnz);

    // allocate CSR column indices and values
    cudaMalloc((void**) d_csr_columns, nnz * sizeof(int));
    cudaMalloc((void**) d_csr_values,  nnz * sizeof(double));

    // reset offsets, column indices, and values pointers
    cusparseStatus_t status = cusparseCsrSetPointers(matB, *d_csr_offsets, *d_csr_columns, *d_csr_values);
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cerr << "cusparseCsrSetPointers failed." << std::endl;
        return;
    }

    // execute Sparse to Dense conversion
    cusparseDenseToSparse_convert(handle, matA, matB,
                                  CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                  dBuffer);

    // destroy matrix/vector descriptors
    cusparseDestroyDnMat(matA);
    cusparseDestroySpMat(matB);
}