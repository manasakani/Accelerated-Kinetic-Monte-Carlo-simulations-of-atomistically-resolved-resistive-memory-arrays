#include "gpu_solvers.h"

//#define NUM_THREADS 512
#define NUM_THREADS 512

//******************************************
// Updating the charge of every charged site
//******************************************

__global__ void update_charge(const ELEMENT *element, 
                              int *charge, 
                              const int *neigh_idx, 
                              const int N, 
                              const int nn,
                              const ELEMENT* metals, const int num_metals){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int Vnn = 0;

    // each thread gets a different site to evaluate
    for (int idx = tid; idx < N; idx += blockDim.x * gridDim.x) {

        charge[tid] = 0;
        
        if (tid < N && element[tid] == VACANCY){
            charge[tid] = 2;

            // iterate over the neighbors
            for (int j = tid * nn; j < (tid + 1) * nn; ++j){
                if (neigh_idx[j] >=0)
                {
                    if (element[neigh_idx[j]] == VACANCY){
                        Vnn++;
                    }
                    if (is_in_array_gpu(metals, element[neigh_idx[j]], num_metals)){
                        charge[tid] = 0;
                    }
                    if (Vnn >= 2){
                        charge[tid] = 0;
                    }
                }
            }
        }

        if (tid < N && element[tid] == OXYGEN_DEFECT){
            charge[tid] = -2;

            // iterate over the neighbors
            for (int j = tid * nn; j < (tid + 1) * nn; ++j){
                if (neigh_idx[j] >=0)
                {
                    if (is_in_array_gpu(metals, element[neigh_idx[j]], num_metals)){
                        charge[tid] = 0;
                    }
                }
            }
        }
    }
}

void update_charge_gpu(ELEMENT *d_site_element, 
                       int *d_site_charge,
                       int *d_neigh_idx, int N, int nn, 
                       const ELEMENT *d_metals, const int num_metals){

    int num_threads = 1024;
    int num_blocks = (N * nn + num_threads - 1) / num_threads;

    update_charge<<<num_blocks, num_threads>>>(d_site_element, d_site_charge, d_neigh_idx, N, nn, d_metals, num_metals);
}


//**************************************************************************
// Solution for homogenous poisson equation with varying boundary conditions
//   _CB are the functions which use only the contacts as the boundaries
//**************************************************************************


__global__ void set_potential(double *A, double *B, int N)
{
    int didx = blockIdx.x * blockDim.x + threadIdx.x;
    for (auto i = didx; i < N; i += gridDim.x * blockDim.x)
    {
        A[i] = -B[i];
    }
}

__global__ void set_diag_K(double *A, double *diag, int N)
{
    int didx = blockIdx.x * blockDim.x + threadIdx.x;
    for (auto i = didx; i < N; i += gridDim.x * blockDim.x)
    {
        double tmp = A[i];
        A[i] = tmp + diag[i];
    }
}

template <int NTHREADS>
__global__ void diagonal_sum_K(
    double *A,
    double *diag,
    double *V,
    int N, int NI, int NJ)
{

    int num_threads = blockDim.x;
    int blocks_per_row = (NJ - 1) / num_threads + 1;
    int block_id = blockIdx.x;

    int row = block_id / blocks_per_row;
    int scol = (block_id % blocks_per_row) * num_threads;
    int lcol = min(NJ, scol + num_threads);

    int tid = threadIdx.x;

    __shared__ double buf[NTHREADS];

    for (auto ridx = row; ridx < NI; ridx += gridDim.x)
    {

        buf[tid] = 0.0;
        if (tid + scol < lcol)
        {
            buf[tid] = A[ridx * N + scol + tid] * V[tid + scol];
            // if (ridx == 7039) {
            //     printf("Thread %d (%d, %d) A=%E, V=%E, buf=%E\n", tid, ridx, tid + scol, A[ridx * N + tid + scol], V[tid + scol], buf[tid]);
            // }
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


__global__ void create_K(
    double *X,
    const double *posx, const double *posy, const double *posz,
    const ELEMENT *metals, const ELEMENT *element, const int *site_charge,
    const double *lattice, const bool pbc, const double high_G, const double low_G,
    const double nn_dist, const int N, const int num_metals)
{

    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    for (auto idx = tid_total; idx < N * N; idx += num_threads_total)
    {
        int i = idx / N;
        int j = idx % N;

        bool metal1 = is_in_array_gpu(metals, element[i], num_metals);
        bool metal2 = is_in_array_gpu(metals, element[j], num_metals);
        bool ischarged1 = site_charge[i] != 0;
        bool ischarged2 = site_charge[j] != 0;
        bool isVacancy1 = element[i] == VACANCY;
        bool isVacancy2 = element[j] == VACANCY;
        bool cvacancy1 = isVacancy1 && !ischarged1;
        bool cvacancy2 = isVacancy2 && !ischarged2;
        double dist = site_dist_gpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j], lattice[0], lattice[1], lattice[2], pbc);

        bool neighbor = false;
        if (dist < nn_dist && i != j)
            neighbor = true;

        // direct terms:
        if (i != j && neighbor)
        {
            if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
            {
                X[N * (i) + (j)] = -high_G;
            }
            else
            {
                X[N * (i) + (j)] = -low_G;
            }
        }
    }
}

__global__ void calc_off_diagonal_A_gpu(
    const ELEMENT *metals, const ELEMENT *element, const int *site_charge,
    int num_metals,
    double d_high_G, double d_low_G,
    int matrix_size,
    int *col_indices,
    int *row_ptr,
    double *data
)
{
    // parallelize over rows
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i != col_indices[j]){
                bool metal1 = is_in_array_gpu(metals, element[i], num_metals);
                bool metal2 = is_in_array_gpu(metals, element[col_indices[j]], num_metals);
                bool ischarged1 = site_charge[i] != 0;
                bool ischarged2 = site_charge[col_indices[j]] != 0;
                bool isVacancy1 = element[i] == VACANCY;
                bool isVacancy2 = element[col_indices[j]] == VACANCY;
                bool cvacancy1 = isVacancy1 && !ischarged1;
                bool cvacancy2 = isVacancy2 && !ischarged2;
                if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
                {
                    data[j] = -d_high_G;
                }
                else
                {
                    data[j] = -d_low_G;
                }
            }
        }
    }
}

__global__ void calc_off_diagonal_dist(
    const ELEMENT *metals, const ELEMENT *element, const int *site_charge,
    int size_i,
    int size_j,
    int start_i,
    int start_j,
    int num_metals,
    double d_high_G, double d_low_G,
    int *col_indices,
    int *row_ptr,
    double *data
)
{
    // parallelize over rows
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int id = idx; id < size_i; id += blockDim.x * gridDim.x){
        for(int jd = row_ptr[id]; jd < row_ptr[id+1]; jd++){
            int i = start_i + id;
            int j = start_j + col_indices[jd];
            if(i != j){
                bool metal1 = is_in_array_gpu(metals, element[i], num_metals);
                bool metal2 = is_in_array_gpu(metals, element[j], num_metals);
                bool ischarged1 = site_charge[i] != 0;
                bool ischarged2 = site_charge[j] != 0;
                bool isVacancy1 = element[i] == VACANCY;
                bool isVacancy2 = element[j] == VACANCY;
                bool cvacancy1 = isVacancy1 && !ischarged1;
                bool cvacancy2 = isVacancy2 && !ischarged2;
                if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
                {
                    data[jd] = -d_high_G;
                }
                else
                {
                    data[jd] = -d_low_G;
                }
            }
        }
    }
}



// used for CB edge calculation
__global__ void calc_off_diagonal_A_CB_gpu(
    const ELEMENT *metals, const ELEMENT *element, const int *site_charge,
    int num_metals,
    double d_high_G, double d_low_G,
    int matrix_size,
    int *col_indices,
    int *row_ptr,
    double *data
)
{
    // parallelize over rows
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i != col_indices[j]){
                bool metal1 = is_in_array_gpu(metals, element[i], num_metals);
                bool metal2 = is_in_array_gpu(metals, element[col_indices[j]], num_metals);

                if (metal1 || metal2)
                {
                    data[j] = -d_high_G;
                }
                else
                {
                    data[j] = -d_low_G;
                }
            }
        }
    }
}



__global__ void reduce_contact_into_diag(
    const ELEMENT *metals_d, const ELEMENT *element_d, const int *site_charge_d,
    int block_size_i,
    int block_size_j,
    int block_start_i,
    int block_start_j,
    const int num_metals,
    const double d_high_G, const double d_low_G,    
    int *col_indices_d,
    int *row_ptr_d,
    double *rows_reduced_d
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int row = idx; row < block_size_i; row += blockDim.x * gridDim.x){
        double tmp = 0.0;
        for(int col = row_ptr_d[row]; col < row_ptr_d[row+1]; col++){
            int i = block_start_i + row;
            int j = block_start_j + col_indices_d[col];

            bool metal1 = is_in_array_gpu(metals_d, element_d[i], num_metals);
            bool metal2 = is_in_array_gpu(metals_d, element_d[j], num_metals);
            bool ischarged1 = site_charge_d[i] != 0;
            bool ischarged2 = site_charge_d[j] != 0;
            bool isVacancy1 = element_d[i] == VACANCY;
            bool isVacancy2 = element_d[j] == VACANCY;
            bool cvacancy1 = isVacancy1 && !ischarged1;
            bool cvacancy2 = isVacancy2 && !ischarged2;

            // sign is switched since the diagonal is positive
            if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
            {
                tmp += d_high_G;
            }
            else
            {
                tmp += d_low_G;
            }
        }
        rows_reduced_d[row] = tmp;

    }

}


__global__ void row_reduce_K_CB_off_diagonal_block_with_precomputing(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    const ELEMENT *metals_d, const ELEMENT *element_d, const int *site_charge_d,
    const int num_metals,
    const double d_high_G, const double d_low_G,
    int block_size_i,
    int block_size_j,
    int block_start_i,
    int block_start_j,
    int *col_indices_d,
    int *row_ptr_d,
    double *rows_reduced_d
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int row = idx; row < block_size_i; row += blockDim.x * gridDim.x){
        double tmp = 0.0;
        for(int col = row_ptr_d[row]; col < row_ptr_d[row+1]; col++){
            int i = block_start_i + row;
            int j = block_start_j + col_indices_d[col];

            bool metal1 = is_in_array_gpu(metals_d, element_d[i], num_metals);
            bool metal2 = is_in_array_gpu(metals_d, element_d[j], num_metals);
            double dist = site_dist_gpu(posx_d[i], posy_d[i], posz_d[i], posx_d[j], posy_d[j], posz_d[j], lattice_d[0], lattice_d[1], lattice_d[2], pbc);

            if (dist < cutoff_radius)
            {
                // sign is switched since the diagonal is positive
                if (metal1 || metal2)
                {
                    tmp += d_high_G;
                }
                else
                {
                    tmp += d_low_G;
                }
            }            
        }
        rows_reduced_d[row] = tmp;

    }

}


__global__ void add_vector_to_diagonal(
    double *data,
    int *row_ptr,
    int *col_indices,
    int matrix_size,
    double *vector
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i == col_indices[j]){
                data[j] += vector[i];
            }
        }
    }
}


// Creates the rhs vector
__global__ void calc_rhs_for_A(const double* K_left_reduced_d, 
                               const double* K_right_reduced_d, 
                               const double* VL, const double* VR, 
                               double* rhs, 
                               int N_interface, int N_left_tot, int N_right_tot) {

    // K_left_reduced_d and K_right_reduced_d are of size N_interface
    // VL is of size N_left_tot and VR is of size N_right_tot

    // N_interface is the number of rows being multiplied by this rank

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < N_interface; i += blockDim.x * gridDim.x) {
        rhs[i] = K_left_reduced_d[i] * (*VL) + K_right_reduced_d[i] * (*VR);
    }
}


// Assemble the conductance matrix for the device and the reduced contact terms
void Assemble_A(
    const double *posx, const double *posy, const double *posz,
    const double *lattice, const bool pbc,
    const double cutoff_radius,
    const ELEMENT *metals_d, const ELEMENT *element_d, const int *site_charge_d,
    const int num_metals, const double d_high_G, const double d_low_G,
    int K_size, int contact_left_size, int contact_right_size,
    double **A_data, int **A_row_ptr, int **A_col_indices, int *A_nnz,
    int **contact_left_col_indices, int **contact_left_row_ptr, int *contact_left_nnz,
    int **contact_right_col_indices, int **contact_right_row_ptr, int *contact_right_nnz,
    double **K_left_reduced, double **K_right_reduced
)
{

    int system_size = K_size - contact_left_size - contact_right_size;

    gpuErrchk(cudaMalloc((void **)K_left_reduced, system_size * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)K_right_reduced, system_size * sizeof(double)));

    // parallelize over rows
    int threads = 512;
    int blocks = (system_size + threads - 1) / threads;

    // allocate the data array
    gpuErrchk(cudaMalloc((void **)A_data, A_nnz[0] * sizeof(double)));
    gpuErrchk(cudaMemset((*A_data), 0, A_nnz[0] * sizeof(double)));

    // assemble only smaller part of K
    calc_off_diagonal_A_gpu<<<blocks, threads>>>(
        metals_d, element_d + contact_left_size, 
        site_charge_d + contact_left_size,
        num_metals,
        d_high_G, d_low_G,
        system_size,
        *A_col_indices,
        *A_row_ptr,
        *A_data);
    gpuErrchk( cudaDeviceSynchronize() );

    reduce_rows_into_diag<<<blocks, threads>>>(*A_col_indices, *A_row_ptr, *A_data, system_size);
    gpuErrchk( cudaDeviceSynchronize() );

    // reduce the left part of K
    // block starts at i = contact_left_size (first downshifted row)
    // block starts at j = 0 (first column)
    reduce_contact_into_diag<<<blocks, threads>>>(
        metals_d, element_d, site_charge_d,
        system_size,
        contact_left_size,
        contact_left_size,
        0,
        num_metals,
        d_high_G, d_low_G,        
        *contact_left_col_indices,
        *contact_left_row_ptr,
        *K_left_reduced
    );

    // reduce the right part of K
    // block starts at i = contact_left_size (first downshifted row)
    // block starts at j = contact_left_size + system_size (first column)
    reduce_contact_into_diag<<<blocks, threads>>>(
        metals_d, element_d, site_charge_d,
        system_size,
        contact_right_size,
        contact_left_size,
        contact_left_size + system_size,
        num_metals,
        d_high_G, d_low_G,        
        *contact_right_col_indices,
        *contact_right_row_ptr,
        *K_right_reduced
    );

    // add left and right part of K to the diagonal of the data array
    add_vector_to_diagonal<<<blocks, threads>>>(
        *A_data,
        *A_row_ptr,
        *A_col_indices,
        system_size,
        *K_left_reduced
    );
    add_vector_to_diagonal<<<blocks, threads>>>(
        *A_data,
        *A_row_ptr,
        *A_col_indices,
        system_size,
        *K_right_reduced
    );

}


// Assemble the conductance matrix for the device and the reduced contact terms
void Assemble_A_CB(
    const double *posx, const double *posy, const double *posz,
    const double *lattice, const bool pbc,
    const double cutoff_radius,
    const ELEMENT *metals_d, const ELEMENT *element_d, const int *site_charge_d,
    const int num_metals, const double d_high_G, const double d_low_G,
    int K_size, int contact_left_size, int contact_right_size,
    double **A_data, int **A_row_ptr, int **A_col_indices, int *A_nnz,
    int **contact_left_col_indices, int **contact_left_row_ptr, int *contact_left_nnz,
    int **contact_right_col_indices, int **contact_right_row_ptr, int *contact_right_nnz,
    double **K_left_reduced, double **K_right_reduced
)
{

    int system_size = K_size - contact_left_size - contact_right_size;

    gpuErrchk(cudaMalloc((void **)K_left_reduced, system_size * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)K_right_reduced, system_size * sizeof(double)));

    // parallelize over rows
    int threads = 512;
    int blocks = (system_size + threads - 1) / threads;

    // allocate the data array
    gpuErrchk(cudaMalloc((void **)A_data, A_nnz[0] * sizeof(double)));
    gpuErrchk(cudaMemset((*A_data), 0, A_nnz[0] * sizeof(double)));

    // assemble only smaller part of K
    calc_off_diagonal_A_CB_gpu<<<blocks, threads>>>(
        metals_d, element_d + contact_left_size, 
        site_charge_d + contact_left_size,
        num_metals,
        d_high_G, d_low_G,
        system_size,
        *A_col_indices,
        *A_row_ptr,
        *A_data);
    gpuErrchk( cudaDeviceSynchronize() );

    reduce_rows_into_diag<<<blocks, threads>>>(*A_col_indices, *A_row_ptr, *A_data, system_size);
    gpuErrchk( cudaDeviceSynchronize() );

    // reduce the left part of K
    // block starts at i = contact_left_size (first downshifted row)
    // block starts at j = 0 (first column)
    row_reduce_K_CB_off_diagonal_block_with_precomputing<<<blocks, threads>>>(
        posx, posy, posz,
        lattice, pbc,
        cutoff_radius,
        metals_d, element_d, site_charge_d,
        num_metals,
        d_high_G, d_low_G,
        system_size,
        contact_left_size,
        contact_left_size,
        0,
        *contact_left_col_indices,
        *contact_left_row_ptr,
        *K_left_reduced
    );

    // reduce the right part of K
    // block starts at i = contact_left_size (first downshifted row)
    // block starts at j = contact_left_size + system_size (first column)
    row_reduce_K_CB_off_diagonal_block_with_precomputing<<<blocks, threads>>>(
        posx, posy, posz,
        lattice, pbc,
        cutoff_radius,
        metals_d, element_d, site_charge_d,
        num_metals,
        d_high_G, d_low_G,
        system_size,
        contact_right_size,
        contact_left_size,
        contact_left_size + system_size,
        *contact_right_col_indices,
        *contact_right_row_ptr,
        *K_right_reduced
    );

    // add left and right part of K to the diagonal of the data array
    add_vector_to_diagonal<<<blocks, threads>>>(
        *A_data,
        *A_row_ptr,
        *A_col_indices,
        system_size,
        *K_left_reduced
    );
    add_vector_to_diagonal<<<blocks, threads>>>(
        *A_data,
        *A_row_ptr,
        *A_col_indices,
        system_size,
        *K_right_reduced
    );

}

void update_CB_edge_gpu_sparse(cublasHandle_t handle_cublas, cusolverDnHandle_t handle, GPUBuffers &gpubuf,
                               const int N, const int N_left_tot, const int N_right_tot,
                               const double Vd, const int pbc, const double high_G, const double low_G, const double nn_dist, 
                               const int num_metals)
{
    // *********************************************************************
    // 1. Assemble the device conductance matrix (A) and the boundaries (rhs)
    // reuse the precalculated sparsity of the potential matrix
    // the solution is written into the buffer for site_CB_edge, and then filtered into atom_CB_edge

    // device submatrix size
    int N_interface = N - (N_left_tot + N_right_tot);

    // Prepare the matrix (fill in the sparsity pattern)
    double *A_data_d = NULL;
    double *K_left_reduced_d = NULL;
    double *K_right_reduced_d = NULL;
    
    // assemble the matrix that will solve for the CB edge vector
    Assemble_A_CB( gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
                   gpubuf.lattice, pbc, nn_dist,
                   gpubuf.metal_types, gpubuf.site_element, gpubuf.site_charge,
                   num_metals, high_G, low_G,
                   N, N_left_tot, N_right_tot,
                   &A_data_d, &gpubuf.Device_row_ptr_d, &gpubuf.Device_col_indices_d, &gpubuf.Device_nnz,
                   &gpubuf.contact_left_col_indices, &gpubuf.contact_left_row_ptr, &gpubuf.contact_left_nnz,
                   &gpubuf.contact_right_col_indices, &gpubuf.contact_right_row_ptr, &gpubuf.contact_left_nnz,
                   &K_left_reduced_d, &K_right_reduced_d );

    // //DEBUG
    // // dump A into a text file:
    // dump_csr_matrix_txt(N_interface, gpubuf.Device_nnz, gpubuf.Device_row_ptr_d, gpubuf.Device_col_indices_d, A_data_d, 0);
    // std::cout << "dumped csr matrix\n";
    // exit(1);
    // //DEBUG

    // Prepare the RHS vector: rhs = -K_left_interface * VL - K_right_interface * VR
    // we take the negative and do rhs = K_left_interface * VL + K_right_interface * VR to account for a sign change in v_soln
    double *VL, *VR, *rhs;
    double Vl_h = Vd/2;
    double Vr_h = -Vd/2;
    gpuErrchk( cudaMalloc((void **)&VL, 1 * sizeof(double)) );
    gpuErrchk( cudaMalloc((void **)&VR, 1 * sizeof(double)) );
    gpuErrchk( cudaMemcpy(VL, &Vl_h, 1 * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(VR, &Vr_h, 1 * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMalloc((void **)&rhs, N_interface * sizeof(double)) ); 
    gpuErrchk( cudaMemset(rhs, 0, N_interface * sizeof(double)) );
    gpuErrchk( cudaDeviceSynchronize() );

    int num_threads = 256;
    int num_blocks = (N_interface + num_threads - 1) / num_threads;
    calc_rhs_for_A<<<num_blocks, num_threads>>>(K_left_reduced_d, K_right_reduced_d, VL, VR, rhs, N_interface, N_left_tot, N_right_tot);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    // ***********************************
    // 2. Solve system of linear equations 

    // the initial guess for the solution is the current site-resolved potential inside the device
    double *v_soln = gpubuf.site_CB_edge + N_left_tot;

    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);
    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_DEVICE);

    // sparse solver with Jacobi preconditioning:
    // double tol = 1e-14*N_interface;
    double tol = 1e-10*N_interface;
    solve_sparse_CG_Jacobi(handle_cublas, cusparseHandle, A_data_d, gpubuf.Device_row_ptr_d, gpubuf.Device_col_indices_d, gpubuf.Device_nnz, N_interface, rhs, v_soln, tol);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // ***************************************************************************
    // 3. Re-fix the boundary (for changes in applied potential across an IV sweep)

    thrust::device_ptr<double> left_boundary = thrust::device_pointer_cast(gpubuf.site_CB_edge);
    thrust::fill(left_boundary, left_boundary + N_left_tot, Vd/2);
    thrust::device_ptr<double> right_boundary = thrust::device_pointer_cast(gpubuf.site_CB_edge + N_left_tot + N_interface);
    thrust::fill(right_boundary, right_boundary + N_right_tot, -Vd/2);

    // Multiply by ev_to_J for the correct units of energy
    CheckCublasError( cublasDscal(handle_cublas, N, &eV_to_J, gpubuf.site_CB_edge, 1) ); 

    // check solution vector
    // double *copy_back = (double *)calloc(N, sizeof(double));
    // gpuErrchk( cudaMemcpy(copy_back, gpubuf.site_CB_edge, N * sizeof(double), cudaMemcpyDeviceToHost) );
    // for (int i = 0; i < N; i++){
    //     std::cout << copy_back[i] << " ";
    // }
    // std::cout << "\nPrinted solution vector, now exiting\n";
    // exit(1);

    cusparseDestroy(cusparseHandle);
    cudaFree(A_data_d);
    cudaFree(VL);
    cudaFree(VR);
    cudaFree(rhs);
    cudaFree(K_left_reduced_d);
    cudaFree(K_right_reduced_d);
    gpuErrchk( cudaPeekAtLastError() );

}

__global__ void reduce_rows_into_diag( 
    int *col_indices,
    int *row_ptr,
    double *data,
    double *diag,
    int matrix_size
)
{
    // reduce the elements in the row

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        //reduce the elements in the row
        double tmp = 0.0;
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            tmp += data[j];
        }
        // write the diagonal element   
        diag[i] -= tmp;
    }
}
__global__ void insert_into_diag(
    double *device_diag,
    double *left_boundary_diag,
    double *right_boundary_diag,
    int *col_indices,
    int *row_ptr,
    double *data,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i == col_indices[j]){
                double tmp = device_diag[i] + left_boundary_diag[i] + right_boundary_diag[i];
                data[j] = tmp;
            }
        }
    }
}


__global__ void inverse_diag(
    double *inv_diag,
    double *diag1,
    double *diag2,
    double *diag3,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        inv_diag[i] = 1.0 / (diag1[i] + diag2[i] + diag3[i]);
    }
}

__global__ void sum_AB_into_A(
    double * __restrict__ A,
    double * __restrict__ B,
    int N
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < N; i += blockDim.x * gridDim.x){
        A[i] += B[i];
    }
}

void background_potential_gpu_sparse_local(cublasHandle_t handle_cublas, cusolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf, const int N, const int N_left_tot, const int N_right_tot,
                                           const double Vd, const int pbc, const double high_G, const double low_G, const double nn_dist,
                                           const int num_metals, int kmc_step_count)
{

    // *********************************************************************
    // 1. Assemble the device conductance matrix (A) and the boundaries (rhs)
    // based on the precalculated sparsity of the neighbor connections (CSR rows/cols)

    // device submatrix size
    int N_interface = N - (N_left_tot + N_right_tot);

    // Prepare the matrix (fill in the sparsity pattern)
    double *A_data_d = NULL;
    double *K_left_reduced_d = NULL;
    double *K_right_reduced_d = NULL;

    // std::cout << "mpi rank" << gpubuf.rank << "  before Assemble_A " << std::endl;

    // the sparsity of the graph connectivity (which goes into A) is precomputed and stored in the buffers:
    Assemble_A( gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
                gpubuf.lattice, pbc, nn_dist,
                gpubuf.metal_types, gpubuf.site_element, gpubuf.site_charge,
                num_metals, high_G, low_G,
                N, N_left_tot, N_right_tot,
                &A_data_d, &gpubuf.Device_row_ptr_d, &gpubuf.Device_col_indices_d, &gpubuf.Device_nnz,
                &gpubuf.contact_left_col_indices, &gpubuf.contact_left_row_ptr, &gpubuf.contact_left_nnz,
                &gpubuf.contact_right_col_indices, &gpubuf.contact_right_row_ptr, &gpubuf.contact_left_nnz,
                &K_left_reduced_d, &K_right_reduced_d );

    // std::cout << "mpi rank" << gpubuf.rank << "  after Assemble_A " << std::endl;

    // // DEBUG
    // dump A into a text file:
    // dump_csr_matrix_txt(N_interface, gpubuf.Device_nnz, gpubuf.Device_row_ptr_d, gpubuf.Device_col_indices_d, A_data_d, kmc_step_count);
    // std::cout << "dumped csr matrix from non-mpi version\n";
    // exit(1);
    // // DEBUG

    // Prepare the RHS vector: rhs = -K_left_interface * VL - K_right_interface * VR
    // we take the negative and do rhs = K_left_interface * VL + K_right_interface * VR to account for a sign change in v_soln
    double *VL, *VR, *rhs;
    double Vl_h = -Vd/2;
    double Vr_h = Vd/2;
    gpuErrchk( cudaMalloc((void **)&VL, 1 * sizeof(double)) );
    gpuErrchk( cudaMalloc((void **)&VR, 1 * sizeof(double)) );
    gpuErrchk( cudaMemcpy(VL, &Vl_h, 1 * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(VR, &Vr_h, 1 * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMalloc((void **)&rhs, N_interface * sizeof(double)) ); 
    gpuErrchk( cudaMemset(rhs, 0, N_interface * sizeof(double)) );
    gpuErrchk( cudaDeviceSynchronize() );

    int num_threads = 256;
    int num_blocks = (N_interface + num_threads - 1) / num_threads;
    calc_rhs_for_A<<<num_blocks, num_threads>>>(K_left_reduced_d, K_right_reduced_d, VL, VR, rhs, N_interface, N_left_tot, N_right_tot);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // std::cout << "mpi rank" << gpubuf.rank << "  after calc_rhs_for_A " << std::endl;
    
    // ***********************************
    // 2. Solve system of linear equations 

    // the initial guess for the solution is the current site-resolved potential inside the device
    double *v_soln = gpubuf.site_potential_boundary + N_left_tot;

    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);
    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_DEVICE);

    // std::cout << "mpi rank" << gpubuf.rank << "  going to solve " << std::endl;

    // sparse solver with Jacobi preconditioning:
    double tol = 1e-14*N_interface;
    // double tol = 1e-28*N_interface; // low tol for debugging
    solve_sparse_CG_Jacobi(handle_cublas, cusparseHandle, A_data_d, gpubuf.Device_row_ptr_d, gpubuf.Device_col_indices_d, gpubuf.Device_nnz, N_interface, rhs, v_soln, tol);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // std::cout << "mpi rank" << gpubuf.rank << "  after solve " << std::endl;

    // ***************************************************************************
    // 3. Re-fix the boundary (for changes in applied potential across an IV sweep)

    thrust::device_ptr<double> left_boundary = thrust::device_pointer_cast(gpubuf.site_potential_boundary);
    thrust::fill(left_boundary, left_boundary + N_left_tot, -Vd/2);
    thrust::device_ptr<double> right_boundary = thrust::device_pointer_cast(gpubuf.site_potential_boundary + N_left_tot + N_interface);
    thrust::fill(right_boundary, right_boundary + N_right_tot, Vd/2);

    // std::cout << "mpi rank" << gpubuf.rank << " refixed the boundary " << std::endl;

    cusparseDestroy(cusparseHandle);
    cudaFree(A_data_d);
    cudaFree(VL);
    cudaFree(VR);
    cudaFree(rhs);
    cudaFree(K_left_reduced_d);
    cudaFree(K_right_reduced_d);
    gpuErrchk( cudaPeekAtLastError() );

    // std::cout << "mpi rank" << gpubuf.rank << " freed stuff " << std::endl;
    
}


void background_potential_gpu_sparse(cublasHandle_t handle_cublas, cusolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf, const int N, const int N_left_tot, const int N_right_tot,
                                     const double Vd, const int pbc, const double high_G, const double low_G, const double nn_dist,
                                     const int num_metals, int kmc_step_count)
{

    // *********************************************************************
    // 1. Assemble the device conductance matrix (A) and the boundaries (rhs)
    // based on the precalculated sparsity of the neighbor connections (CSR rows/cols)

    // device submatrix size
    int N_interface = N - (N_left_tot + N_right_tot);

    // ** prepare the matrix (fill in the sparsity pattern)
    Distributed_matrix *A_distributed = gpubuf.K_distributed;
    int rows_this_rank = A_distributed->rows_this_rank;
    int disp_this_rank = A_distributed->displacements[A_distributed->rank];

    // std::cout << "mpi rank" << gpubuf.rank << "  rows_this_rank " << rows_this_rank << " disp_this_rank " << disp_this_rank << std::endl;

    // TODO switch to COO format to assemble faster
    int threads = 1024;
    int blocks = (A_distributed->rows_this_rank + threads - 1) / threads;    
    for(int i = 0; i < A_distributed->number_of_neighbours; i++){

        int rows_neighbour = A_distributed->counts[A_distributed->neighbours[i]];
        int disp_neighbour = A_distributed->displacements[A_distributed->neighbours[i]];

        gpuErrchk(cudaMemset(A_distributed->data_d[i], 0,
            A_distributed->nnz_per_neighbour[i] * sizeof(double)) );

        calc_off_diagonal_dist<<<blocks, threads>>>(
            gpubuf.metal_types, gpubuf.site_element, gpubuf.site_charge,
            rows_this_rank,
            rows_neighbour,
            N_left_tot + disp_this_rank,
            N_left_tot + disp_neighbour,
            num_metals,
            high_G, low_G,
            A_distributed->col_indices_d[i],
            A_distributed->row_ptr_d[i],
            A_distributed->data_d[i]);
    }

    // ** update the diagonal

    // compute the local pieces of the diagonal
    double *diagonal_local_d;
    gpuErrchk( cudaMalloc((void **)&diagonal_local_d, A_distributed->rows_this_rank * sizeof(double)) );
    gpuErrchk( cudaMemset(diagonal_local_d, 0, A_distributed->rows_this_rank * sizeof(double)) );

    // contribution to diagonal from left boundary
    double *left_boundary_d;
    gpuErrchk( cudaMalloc((void **)&left_boundary_d, A_distributed->rows_this_rank * sizeof(double)) );
    gpuErrchk( cudaMemset(left_boundary_d, 0, A_distributed->rows_this_rank * sizeof(double)) );
    
    // contribution to diagonal from right boundary
    double *right_boundary_d;
    gpuErrchk( cudaMalloc((void **)&right_boundary_d, A_distributed->rows_this_rank * sizeof(double)) );
    gpuErrchk( cudaMemset(right_boundary_d, 0, A_distributed->rows_this_rank * sizeof(double)) );

    // sum the sub-blocks of the matrix owned by this rank into its diagonal
    for(int i = 0; i < A_distributed->number_of_neighbours; i++){
        // needs that the diagonal element is zero
        reduce_rows_into_diag<<<blocks, threads>>>(
            A_distributed->col_indices_d[i],
            A_distributed->row_ptr_d[i],
            A_distributed->data_d[i],
            diagonal_local_d,
            A_distributed->rows_this_rank
            );
    }

    // // DEBUG
    // // dump A into a text file:
    // dump_csr_matrix_txt(N_interface, A_distributed->nnz, A_distributed->row_ptr_d[0],  A_distributed->col_indices_d[0], A_distributed->data_d[0], 0);
    // std::cout << "dumped csr matrix\n";
    // exit(1);
    // // DEBUG

    // copy back and print gpubuf.left_row_ptr_d
    // int *left_row_ptr_h = (int *)calloc(rows_this_rank + 1, sizeof(int));
    // gpuErrchk( cudaMemcpy(left_row_ptr_h, gpubuf.left_row_ptr_d, (rows_this_rank + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    // for (int i = 0; i < rows_this_rank + 1; i++){
    //     std::cout << left_row_ptr_h[i] << " ";
    // }
    // exit(1);

    // update the diagonal with the terms corresponding to the left boundary
    reduce_contact_into_diag<<<blocks, threads>>>(
        gpubuf.metal_types, gpubuf.site_element, gpubuf.site_charge,
        A_distributed->rows_this_rank,
        N_left_tot,
        N_left_tot + disp_this_rank,
        0,
        num_metals,
        high_G, low_G,        
        gpubuf.left_col_indices_d,
        gpubuf.left_row_ptr_d,
        left_boundary_d
    );

    // update the diagonal with the terms corresponding to the right boundary
    reduce_contact_into_diag<<<blocks, threads>>>(
        gpubuf.metal_types, gpubuf.site_element, gpubuf.site_charge,
        A_distributed->rows_this_rank,
        N_right_tot,
        N_left_tot + disp_this_rank,
        N_left_tot + N_interface,
        num_metals,
        high_G, low_G,        
        gpubuf.right_col_indices_d,
        gpubuf.right_row_ptr_d,
        right_boundary_d
    );

    // insert the diagonal elements into the matrix
    insert_into_diag<<<blocks, threads>>>(
        diagonal_local_d,
        left_boundary_d,
        right_boundary_d,
        A_distributed->col_indices_d[0],
        A_distributed->row_ptr_d[0],
        A_distributed->data_d[0],
        A_distributed->rows_this_rank
    );

    // // DEBUG
    // dump A into a text file:
    // dump_csr_matrix_txt(N_interface, A_distributed->nnz, A_distributed->row_ptr_d[0],  A_distributed->col_indices_d[0], A_distributed->data_d[0], 0);
    // std::cout << "dumped csr matrix\n";
    // exit(1);
    // // DEBUG

    // Prepare the RHS vector: rhs = -K_left_interface * VL - K_right_interface * VR
    // we take the negative and do rhs = K_left_interface * VL + K_right_interface * VR to account for a sign change in v_soln
    double *VL, *VR;
    double Vl_h = -Vd/2;
    double Vr_h = Vd/2;
    gpuErrchk( cudaMalloc((void **)&VL, 1 * sizeof(double)) );
    gpuErrchk( cudaMalloc((void **)&VR, 1 * sizeof(double)) );
    gpuErrchk( cudaMemcpy(VL, &Vl_h, 1 * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(VR, &Vr_h, 1 * sizeof(double), cudaMemcpyHostToDevice) );

    // ***********************************
    // 2. Solve system of linear equations 

    // the initial guess for the solution is the current site-resolved potential inside the device
    double *v_soln = gpubuf.site_potential_boundary + N_left_tot + disp_this_rank;

    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);
    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_DEVICE);

    double *inv_diagonal_d;
    gpuErrchk( cudaMalloc((void **)&inv_diagonal_d, A_distributed->rows_this_rank * sizeof(double)) );

    inverse_diag<<<blocks, threads>>>(
        inv_diagonal_d,
        diagonal_local_d,
        left_boundary_d,
        right_boundary_d,
        A_distributed->rows_this_rank);

    double *rhs_local_d;
    gpuErrchk( cudaMalloc((void **)&rhs_local_d, A_distributed->rows_this_rank * sizeof(double)) );
    gpuErrchk( cudaMemset(rhs_local_d, 0, A_distributed->rows_this_rank * sizeof(double)) );

    calc_rhs_for_A<<<blocks, threads>>>(left_boundary_d, right_boundary_d, VL, VR,
        rhs_local_d, A_distributed->rows_this_rank, N_left_tot, N_right_tot);

    // TODO: remove the MPI barrier inside
    // double relative_tolerance = 1e-10 * N_interface;
    double relative_tolerance = 1e-28 * N_interface; // low tol for debugging
    int max_iterations = 50000;
    iterative_solver::conjugate_gradient_jacobi<dspmv::gpu_packing>(
        *gpubuf.K_distributed,
        *gpubuf.K_p_distributed,
        rhs_local_d,
        v_soln,
        inv_diagonal_d,
        relative_tolerance,
        max_iterations,
        A_distributed->comm);

    // double *v_soln_local_h = (double *)calloc(A_distributed->rows_this_rank, sizeof(double));
    // gpuErrchk( cudaMemcpy(v_soln_local_h, v_soln, A_distributed->rows_this_rank * sizeof(double), cudaMemcpyDeviceToHost) );
    // double *v_soln_h = (double *)calloc(N_interface, sizeof(double));

    // MPI_Allgatherv(v_soln_local_h, A_distributed->rows_this_rank,
    //     MPI_DOUBLE, v_soln_h,
    //     A_distributed->counts, A_distributed->displacements, MPI_DOUBLE, MPI_COMM_WORLD);
    // gpuErrchk( cudaMemcpy(gpubuf.site_potential_boundary + N_left_tot, v_soln_h,
    //     N_interface * sizeof(double), cudaMemcpyHostToDevice) );

    // TODO: remove the reset inside each bias point
    // 3. Re-fix the boundary (for changes in applied potential across an IV sweep)
    thrust::device_ptr<double> left_boundary = thrust::device_pointer_cast(gpubuf.site_potential_boundary);
    thrust::fill(left_boundary, left_boundary + N_left_tot, -Vd/2);
    thrust::device_ptr<double> right_boundary = thrust::device_pointer_cast(gpubuf.site_potential_boundary + N_left_tot + N_interface);
    thrust::fill(right_boundary, right_boundary + N_right_tot, Vd/2);

    // cusparseDestroy(cusparseHandle);
    cudaFree(VL);
    cudaFree(VR);
    gpuErrchk( cudaFree(rhs_local_d) );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaFree(diagonal_local_d) );
    gpuErrchk( cudaFree(left_boundary_d) );
    gpuErrchk( cudaFree(right_boundary_d) );
    gpuErrchk( cudaFree(inv_diagonal_d) );

}

void sum_and_gather_potential(GPUBuffers &gpubuf)
{
    // sum the potential vectors
    int threads = 512;
    int blocks = (gpubuf.count_sites[gpubuf.rank] + threads- 1) / threads;

    // compute the off-diagonal elements of K
    sum_AB_into_A<<<blocks, threads>>>(gpubuf.site_potential_charge + gpubuf.displ_sites[gpubuf.rank], 
                                       gpubuf.site_potential_boundary + gpubuf.displ_sites[gpubuf.rank],
                                       gpubuf.count_sites[gpubuf.rank]);
    
    // copy potential vectors to host
    //double *potential_local_h = (double *)calloc(gpubuf.count_sites[gpubuf.rank], sizeof(double));
    //double *potential_h = (double *)calloc(gpubuf.N_, sizeof(double));

    gpuErrchk( cudaMemcpy(gpubuf.potential_local_h, gpubuf.site_potential_charge + gpubuf.displ_sites[gpubuf.rank], 
                gpubuf.count_sites[gpubuf.rank] * sizeof(double), cudaMemcpyDeviceToHost) );

    // allgather potential vector - site_potential_charge contains the sum
    MPI_Allgatherv(gpubuf.potential_local_h, gpubuf.count_sites[gpubuf.rank],
        MPI_DOUBLE, gpubuf.potential_h,
        gpubuf.count_sites, gpubuf.displ_sites, MPI_DOUBLE, MPI_COMM_WORLD);

    // copy potential vector to device    
    gpuErrchk( cudaMemcpy(gpubuf.site_potential_charge, gpubuf.potential_h,
        gpubuf.N_ * sizeof(double), cudaMemcpyHostToDevice) );

}

// solves site-resolved background potential using dense matrix assembly and direct LU-solver schemes
void background_potential_gpu(cusolverDnHandle_t handle, GPUBuffers &gpubuf, const int N, const int N_left_tot, const int N_right_tot,
                              const double Vd, const int pbc, const double d_high_G, const double d_low_G, const double nn_dist,
                              const int num_metals, int kmc_step_count)
{

    int N_interface = N - (N_left_tot + N_right_tot);

    double *VL, *VR;
    gpuErrchk( cudaMalloc((void **)&VL, N_left_tot * sizeof(double)) );
    gpuErrchk( cudaMalloc((void **)&VR, N_right_tot * sizeof(double)) );

    double *gpu_k;
    double *gpu_diag;
    gpuErrchk( cudaMalloc((void **)&gpu_k, N * N * sizeof(double)) );
    gpuErrchk( cudaMalloc((void **)&gpu_diag, N * sizeof(double)) );
    gpuErrchk( cudaMemset(gpu_k, 0, N * N * sizeof(double)) );
    gpuErrchk( cudaDeviceSynchronize() );

    // prepare contact potentials
    thrust::device_ptr<double> VL_ptr = thrust::device_pointer_cast(VL);
    thrust::fill(VL_ptr, VL_ptr + N_left_tot, -Vd/2);
    thrust::device_ptr<double> VR_ptr = thrust::device_pointer_cast(VR);
    thrust::fill(VR_ptr, VR_ptr + N_right_tot, Vd/2);

    //  BUILDING THE CONDUCTIVITY MATRIX
    int num_threads = 512;
    int blocks_per_row = (N - 1) / num_threads + 1;
    int num_blocks = blocks_per_row * N;

    // compute the off-diagonal elements of K
    create_K<<<num_blocks, num_threads>>>(
        gpu_k, gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
        gpubuf.metal_types, gpubuf.site_element, gpubuf.site_charge,
        gpubuf.lattice, pbc, d_high_G, d_low_G,
        nn_dist, N, num_metals);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Update the diagonal of K
    gpuErrchk( cudaMemset(gpu_diag, 0, N * sizeof(double)) );
    gpuErrchk( cudaDeviceSynchronize() );
    row_reduce<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(gpu_k, gpu_diag, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    num_blocks = (N - 1) / num_threads + 1;
    write_to_diag<<<num_blocks, num_threads>>>(gpu_k, gpu_diag, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemset(gpu_diag, 0, N * sizeof(double)) );
    gpuErrchk( cudaDeviceSynchronize() );

    blocks_per_row = (N_left_tot - 1) / num_threads + 1;
    num_blocks = blocks_per_row * N_interface;
    diagonal_sum_K<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(&gpu_k[N_left_tot * N], gpu_diag, VL, N, N_interface, N_left_tot);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    blocks_per_row = (N_right_tot - 1) / num_threads + 1;
    num_blocks = blocks_per_row * N_interface;
    diagonal_sum_K<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(&gpu_k[N_left_tot * N + N - N_right_tot], gpu_diag, VR, N, N_interface, N_right_tot);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //  SOLVING FOR THE NEGATIVE INTERNAL POTENTIALS (KSUB)
    double *gpu_k_sub;
    gpuErrchk( cudaMalloc((void **)&gpu_k_sub, N_interface * sizeof(double)) ); 
    gpuErrchk( cudaMemset(gpu_k_sub, 0, N_interface * sizeof(double)) );
    gpuErrchk( cudaDeviceSynchronize() );
    num_blocks = (N_interface - 1) / num_threads + 1;
    set_diag_K<<<blocks_per_row, num_threads>>>(gpu_k_sub, gpu_diag, N_interface);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaFree(gpu_diag);

    // ** Solve Ax=B through LU factorization **

    int lwork = 0;              /* size of workspace */
    double *gpu_work = nullptr; /* device workspace for getrf */
    int *gpu_info = nullptr;    /* error info */
    int *gpu_ipiv; // int info;
    gpuErrchk( cudaMalloc((void **)&gpu_ipiv, N_interface * sizeof(int)) ); 
    gpuErrchk( cudaMalloc((void **)(&gpu_info), sizeof(int)) );

    // points to the start of Koxide inside K:
    double* gpu_D = gpu_k + (N_left_tot * N) + N_left_tot;

    CheckCusolverDnError(cusolverDnDgetrf_bufferSize(handle, N_interface, N_interface, gpu_D, N, &lwork));
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMalloc((void **)(&gpu_work), sizeof(double) * lwork) );

    CheckCusolverDnError(cusolverDnDgetrf(handle, N_interface, N_interface, gpu_D, N, gpu_work, gpu_ipiv, gpu_info));
    // cudaMemcpy(&info, gpu_info, sizeof(int), cudaMemcpyDeviceToHost); // printf("info for cusolverDnDgetrf: %i \n", info);
    gpuErrchk( cudaDeviceSynchronize() );

    CheckCusolverDnError(cusolverDnDgetrs(handle, CUBLAS_OP_N, N_interface, 1, gpu_D, N, gpu_ipiv, gpu_k_sub, N_interface, gpu_info));
    // cudaMemcpy(&info, gpu_info, sizeof(int), cudaMemcpyDeviceToHost); // printf("info for cusolverDnDgetrs: %i \n", info);
    gpuErrchk( cudaDeviceSynchronize() );

    cudaFree(gpu_k);

    num_blocks = (N_interface - 1) / num_threads + 1;
    set_potential<<<num_blocks, num_threads>>>(gpubuf.site_potential_boundary + N_left_tot, gpu_k_sub, N_interface);
    gpuErrchk( cudaPeekAtLastError() ); 
    gpuErrchk( cudaDeviceSynchronize() ); 
    cudaFree(gpu_k_sub);

    gpuErrchk( cudaMemcpy(gpubuf.site_potential_boundary, VL, N_left_tot * sizeof(double), cudaMemcpyDeviceToDevice) );
    gpuErrchk( cudaMemcpy(gpubuf.site_potential_boundary + N_left_tot + N_interface, VR, N_right_tot * sizeof(double), cudaMemcpyDeviceToDevice) );

    cudaFree(gpu_ipiv);
    cudaFree(gpu_work);
    cudaFree(gpu_info);
    cudaFree(VL);
    cudaFree(VR);

}


//**********************************************************************
// Solution for the poisson equation with superposition of point sources
//**********************************************************************

template <int NTHREADS>
__global__ void calculate_pairwise_interaction(const double* posx, const double* posy, const double*posz, 
                                               const double *lattice, const int pbc, 
                                               const int N, const double *sigma, const double *k, 
                                               const int *charge, double* potential,
                                               const int row_start, const int row_end){

    // Version with reduction, where every thread evaluates site-site interaction term
    int num_threads = blockDim.x;
    int blocks_per_row = (N + num_threads - 1) / num_threads;
    int block_id = blockIdx.x;

    int row = block_id / blocks_per_row + row_start;
    int scol = (block_id % blocks_per_row) * num_threads;
    int lcol = min(N, scol + num_threads);

    int tid = threadIdx.x;

    __shared__ double buf[NTHREADS];
    double dist;
    int i, j;

    for (int ridx = row; ridx < row_end; ridx += gridDim.x/blocks_per_row) {

        buf[tid] = 0.0;
        if (tid + scol < lcol) {

            i = ridx;
            j = scol+tid;
            if (i != j && charge[j] != 0){
                dist = 1e-10 * site_dist_gpu(posx[i], posy[i], posz[i], 
                                             posx[j], posy[j], posz[j], 
                                             lattice[0], lattice[1], lattice[2], pbc);
                // implement cutoff radius
                if (dist < 2e-9) { // HARDCODED
                    buf[tid] = v_solve_gpu(dist, charge[j], sigma, k);
                }
            }
        }

        int width = num_threads / 2;
        while (width != 0) {
            __syncthreads();
            if (tid < width) {
                buf[tid] += buf[tid + width];
            }
            width /= 2;
        }

        if (tid == 0) {
            atomicAdd(potential + ridx, buf[0]);
        }
    
    }
}

__global__ void calculate_pairwise_interaction_windowed(const double* posx, const double* posy, const double*posz, 
                                                        const double *lattice, const int pbc, 
                                                        const int N, const double *sigma, const double *k, 
                                                        const int *charge, double* potential,
                                                        const int row_start, const int row_end, const int *cutoff_window){

    // Version without reduction, where every thread evaluates a row
    int num_threads = blockDim.x;
    int blocks_per_row = (N + num_threads - 1) / num_threads;
    int block_id = blockIdx.x;

    int row = block_id / blocks_per_row + row_start;

    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    // TODO: switch to reduction
    for (int i = tid_total + row_start; i < row_end; i += num_threads_total)
    {
        int col_start = cutoff_window[2*i];
        int col_end = cutoff_window[2*i + 1];

        for (int j = col_start; j < col_end; j++)
        {
            if (i != j && charge[j] != 0){
                double dist = 1e-10 * site_dist_gpu(posx[i], posy[i], posz[i], 
                                             posx[j], posy[j], posz[j], 
                                             lattice[0], lattice[1], lattice[2], pbc);
                potential[i] += v_solve_gpu(dist, charge[j], sigma, k);
            }
        }
        
    }
}

// __global__ void calculate_pairwise_interaction_indexed(const double* posx, const double* posy, const double*posz, 
//                                                        const double *lattice, const int pbc, 
//                                                        const int N, const double *sigma, const double *k, 
//                                                        const int *charge, double* potential,
//                                                        const int row_start, const int row_end, const int *cutoff_idx, const int N_cutoff){

//     // Version without reduction, where every thread evaluates a row
//     int num_threads = blockDim.x;
//     int blocks_per_row = (N + num_threads - 1) / num_threads;
//     int block_id = blockIdx.x;

//     int row = block_id / blocks_per_row + row_start;

//     int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
//     int num_threads_total = blockDim.x * gridDim.x;

//     // TODO: switch to reduction
//     for (int i = tid_total + row_start; i < row_end; i += num_threads_total)
//     {
//         int j;

//         for (int j_idx = 0; j_idx < N_cutoff; j_idx++)
//         {
//             long int idx_counter = (size_t)i*(size_t)N_cutoff + j_idx;
//             // j = cutoff_idx[i*N_cutoff + j_idx];
//             j = cutoff_idx[idx_counter];

//             // if ((i != j) && (charge[j] != 0) && (j != 0)){
//             if (j >= 0 && i != j && charge[j] != 0) {
//                 double dist = 1e-10 * site_dist_gpu(posx[i], posy[i], posz[i], 
//                                             posx[j], posy[j], posz[j], 
//                                             lattice[0], lattice[1], lattice[2], pbc);

//                 // double dist = (1e-10) * cutoff_dists[i*N_cutoff + j_idx];
//                 potential[i] += v_solve_gpu(dist, charge[j], sigma, k);

//             }
//         }
        
//     }
// }

__global__ void calculate_pairwise_interaction_indexed(const double* posx, const double* posy, const double*posz, 
                                                       const double *lattice, const int pbc, 
                                                       const int N, const double *sigma, const double *k, 
                                                       const int *charge, double* potential,
                                                       const int row_start, const int row_end, const int *cutoff_idx, const int N_cutoff){

    // Version without reduction, where every thread evaluates a row
    int num_threads = blockDim.x;
    int blocks_per_row = (N + num_threads - 1) / num_threads;
    int block_id = blockIdx.x;

    int row = block_id / blocks_per_row + row_start;

    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;
    

    for (int i = tid_total + row_start; i < row_end; i += num_threads_total)
    {
        int j;
        double local_potential = 0.0;

        for (int j_idx = 0; j_idx < N_cutoff; j_idx++)
        {
            long int idx_next = (size_t)i*(size_t)N_cutoff + (size_t)j_idx;
            j = cutoff_idx[idx_next];
            // j = cutoff_idx[i*N_cutoff + j_idx];

            if (j >= 0 && i != j && charge[j] != 0) {
                double dist = 1e-10 * site_dist_gpu(posx[i], posy[i], posz[i], 
                                                    posx[j], posy[j], posz[j], 
                                                    lattice[0], lattice[1], lattice[2], pbc);

                local_potential += v_solve_gpu(dist, charge[j], sigma, k);
            }
        }
        atomicAdd(&potential[i], local_potential);
    }
    
}

// __global__ void calculate_pairwise_interaction_indexed(const double* posx, const double* posy, const double*posz, 
//                                                        const double *lattice, const int pbc, 
//                                                        const int N, const double *sigma, const double *k, 
//                                                        const int *charge, double* potential,
//                                                        const int counts_this_rank, const int displ_this_rank, const int *cutoff_idx, const int N_cutoff){

//     // Version without reduction, where every thread evaluates a row
//     int num_threads = blockDim.x;
//     // int blocks_per_row = (N + num_threads - 1) / num_threads;
//     int block_id = blockIdx.x;

//     // int row = block_id / blocks_per_row + row_start;

//     int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
//     int num_threads_total = blockDim.x * gridDim.x;

//     for (int idx = tid_total; idx < counts_this_rank; idx += num_threads_total)
//     {
//         int j;
//         double local_potential = 0.0;
//         int i = idx + displ_this_rank;

//         for (int j_idx = 0; j_idx < N_cutoff; j_idx++)
//         {
//             // long int idx_next = (long int)(i-row_start)*(long int)N_cutoff + (long int)j_idx;
//             long int idx_next = (size_t)idx*(size_t)N_cutoff + (size_t)j_idx;
//             j = cutoff_idx[idx_next];
//             // j = cutoff_idx[i*N_cutoff + j_idx];

//             if (j >= 0 && i != j && charge[j] != 0) {           // the padding in the cutoff list is -1
//                 double dist = 1e-10 * site_dist_gpu(posx[i], posy[i], posz[i], 
//                                                     posx[j], posy[j], posz[j]);

//                 local_potential += v_solve_gpu(dist, charge[j], sigma, k);
//             }
//         }
//         potential[i] = local_potential;
//     }
    
// }


// template <int NTHREADS>
// __global__ void calculate_pairwise_interaction_indexed(const double* posx, const double* posy, const double* posz,
//                                                        const double *lattice, const int pbc,
//                                                        const int N, const double *sigma, const double *k,
//                                                        const int *charge, double* potential,
//                                                        const int row_start, const int row_end,
//                                                        const int *cutoff_idx, const double *cutoff_dists, const int N_cutoff) {

//     // Version with reduction, where every thread evaluates site-site interaction term
//     int num_threads = blockDim.x;
//     int blocks_per_row = (N + num_threads - 1) / num_threads;
//     int block_id = blockIdx.x;

//     int row = block_id / blocks_per_row + row_start;
//     int scol = (block_id % blocks_per_row) * num_threads;
//     int lcol = min(N_cutoff, scol + num_threads);

//     int tid = threadIdx.x;

//     __shared__ double buf[NTHREADS];
//     double dist;
//     int i, j, j_idx;

//     for (int ridx = row; ridx < row_end; ridx += gridDim.x/blocks_per_row) {

//         buf[tid] = 0.0;
//         if (tid + scol < lcol) {

//             i = ridx;
//             j = scol+tid;

//             j_idx = cutoff_idx[i*N_cutoff + j];
//             dist = 1e-10 * site_dist_gpu(posx[i], posy[i], posz[i], 
//                                          posx[j_idx], posy[j_idx], posz[j_idx], 
//                                          lattice[0], lattice[1], lattice[2], pbc);
//             buf[tid] = v_solve_gpu(dist, charge[j_idx], sigma, k);
//         }

//         int width = num_threads / 2;
//         while (width != 0) {
//             __syncthreads();
//             if (tid < width) {
//                 buf[tid] += buf[tid + width];
//             }
//             width /= 2;
//         }

//         if (tid == 0) {
//             atomicAdd(potential + ridx, buf[0]);
//         }
    
//     }
// }

void poisson_gridless_gpu(const int num_atoms_contact, const int pbc, const int N, const double *lattice, 
                          const double *sigma, const double *k,
                          const double *posx, const double *posy, const double *posz, 
                          const int *site_charge, double *site_potential_charge,
                          const int rank, const int size, const int *count, const int *displ, 
                          const int *cutoff_window, const int *cutoff_idx, const int N_cutoff){

    int num_threads = NUM_THREADS;
    int blocks_per_row = (N + NUM_THREADS - 1) / NUM_THREADS; 
    //TODO: change to variable number of blocks and not fixed
    // overflow problem, but for small devices sm bigger than 10 would be good
    int num_blocks = blocks_per_row * 10;
    //int num_blocks = blocks_per_row * 1;

    // set the inhomogenous poisson solution to zero before populating it
    gpuErrchk( cudaMemset(site_potential_charge, 0, N * sizeof(double)) ); 
    gpuErrchk( cudaDeviceSynchronize() );

    auto t1 = std::chrono::steady_clock::now();

    // this num_threads should be equal to NUM_THREADS

    // // naive implementation, all-to-all
    // calculate_pairwise_interaction<NUM_THREADS><<<num_blocks, NUM_THREADS, NUM_THREADS * sizeof(double)>>>(posx, posy, posz, lattice,
    //     pbc, N, sigma, k, site_charge, site_potential_charge, displ[rank], displ[rank] + count[rank]);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // gpuErrchk( cudaPeekAtLastError() );

    // only checks sites within an index window
    // calculate_pairwise_interaction_windowed<<<num_blocks, num_threads>>>(posx, posy, posz, lattice,
    //     pbc, N, sigma, k, site_charge, site_potential_charge, displ[rank], displ[rank] + count[rank], cutoff_window);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // gpuErrchk( cudaPeekAtLastError() );


    // row-wise: only checks sites which were precomputed to be within the cutoff radius
    calculate_pairwise_interaction_indexed<<<num_blocks, num_threads>>>(posx, posy, posz, lattice,
        pbc, N, sigma, k, site_charge, site_potential_charge, displ[rank], displ[rank] + count[rank], cutoff_idx, N_cutoff);
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );



    // double* host_site_potential_charge = new double[N];
    // cudaMemcpy(host_site_potential_charge, site_potential_charge, N * sizeof(double), cudaMemcpyDeviceToHost);
    // double sum = 0.0;
    // for (int i = 0; i < N; ++i) {
    //     sum += host_site_potential_charge[i];
    // }
    // std::cout << "Sum of site_potential_charge: " << sum << std::endl;
    // exit(1);
}