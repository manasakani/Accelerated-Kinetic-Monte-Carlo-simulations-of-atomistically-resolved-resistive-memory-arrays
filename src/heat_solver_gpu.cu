#include "gpu_solvers.h"
#define NUM_THREADS 512

//reduces the array into the value 
template <typename T, int NTHREADS>
__global__ void reduce(const T* array_to_reduce, T* value, const int N){

    __shared__ T buf[NTHREADS];
    
    int num_threads = blockDim.x;                           // number of threads in this block
    int blocks_per_row = (N-1)/num_threads + 1;             // number of blocks to fit in this array
    int block_id = blockIdx.x;                              // id of the block
    int tid = threadIdx.x;                                  // local thread id to this block
    int row = block_id / blocks_per_row;                    // which 'row' of the array to work on, rows are the overflow

    buf[tid] = 0;

    for (int ridx = row; ridx < N/(blocks_per_row*num_threads) + 1; ridx++){
    
        if (ridx*blocks_per_row*num_threads + block_id * num_threads + tid < N){
            buf[tid] = array_to_reduce[ridx*blocks_per_row*num_threads + block_id * num_threads + tid];
        }
       
        int width = num_threads / 2;

        while (width != 0){
            __syncthreads();
            if (tid < width){
                buf[tid] += buf[tid+width];
            }
            width /= 2;
        }

        if (tid == 0){
           atomicAdd(value, buf[0]);
        }
    }
}


//called by a single gpu-thread
__global__ void update_temp_global(double *P_tot, double* T_bg, const double a_coeff, const double b_coeff, const double number_steps, const double C_thermal, const double small_step)
{
    double c_coeff = b_coeff + *P_tot/C_thermal * small_step;
    double T_intermediate = *T_bg;
    int step = number_steps;
    *T_bg = c_coeff*(1.0-pow(a_coeff, (double) step)) / (1.0-a_coeff) + pow(a_coeff, (double) step)* T_intermediate;
}


//Global temperature update
void update_temperatureglobal_gpu(const double *site_power, double *T_bg, const int N, const double a_coeff, const double b_coeff, const double number_steps, const double C_thermal, const double small_step){

    int num_threads = 512;
    int num_blocks = (N - 1) / num_threads + 1;

    double *P_tot;
    gpuErrchk( cudaMalloc((void**)&P_tot, 1 * sizeof(double)) );
    gpuErrchk( cudaMemset(P_tot, 0, 1 * sizeof(double)) );
    gpuErrchk( cudaDeviceSynchronize() );

    //collect site_power
    reduce<double, NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS*sizeof(double)>>>(site_power, P_tot, N);

    //update the temperature
    update_temp_global<<<1, 1>>>(P_tot, T_bg, a_coeff, b_coeff, number_steps, C_thermal, small_step);

    cudaFree(P_tot);
}

__global__ void calc_nnz_per_row_L(const double *posx, const double *posy, const double *posz, double nn_dist,
                                   int N_boundary_left, int N_boundary_right, int matrix_size,
                                   int *nnz_per_row_d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int row = idx; row < matrix_size; row += blockDim.x * gridDim.x){
        int nnz_row = 0;

        int i = row + N_boundary_left;

        for(int col = 0; col < matrix_size; col++){

            int j = col + N_boundary_left;

            double dist = site_dist_gpu(posx[i], posy[i], posz[i],
                                        posx[j], posy[j], posz[j]);

            if(dist < nn_dist){
                nnz_row++;
            }

        }

        atomicAdd(&nnz_per_row_d[row], nnz_row);
    }

}

__global__ void assemble_L_col_indices(const double *posx, const double *posy, const double *posz, double nn_dist,
                                       int N_boundary_left, int N_boundary_right, int matrix_size,
                                       int *nnz_per_row_d, int *L_row_ptr_d, int *L_col_indices_d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int row = idx; row < matrix_size; row += blockDim.x * gridDim.x){
        int nnz_row = 0;
        int i = row + N_boundary_left;

        for(int col = 0; col < matrix_size; col++){

            int j = col + N_boundary_left;

            double dist = site_dist_gpu(posx[i], posy[i], posz[i],
                                        posx[j], posy[j], posz[j]);

            if(dist < nn_dist){
                L_col_indices_d[L_row_ptr_d[row] + nnz_row] = col;
                nnz_row++;
            }
        }
    }
}

__global__ void populate_L(const double *posx, const double *posy, const double *posz, const ELEMENT *element, double nn_dist,
                           int N_boundary_left, int N_boundary_right, int matrix_size,
                           const ELEMENT *metals, int num_metals,
                           int *nnz_per_row_d, int *L_row_ptr_d, int *L_col_indices_d, double *L_values_d, double gamma, double L_char)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // populate off-diagonals
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        
        for(int j = L_row_ptr_d[i]; j < L_row_ptr_d[i+1]; j++){
            
            int row = i + N_boundary_left;
            int col = L_col_indices_d[j] + N_boundary_left;

            double dist = site_dist_gpu(posx[row], posy[row], posz[row],
                                        posx[col], posy[col], posz[col]);

            if(i != L_col_indices_d[j]){
                // L_values_d[j] = 1.0; 
                L_values_d[j] = dist / (L_char*(1e10)); 
            }
        }
    }

    // sync before adding dissipative terms
    __syncthreads();

    // handle dissipative boundary connections
    int N = N_boundary_left + matrix_size + N_boundary_right;

    // iterate over the diagonal sites
    for(int diag_site = idx; diag_site < matrix_size; diag_site += blockDim.x * gridDim.x){

        // check if there is a boundary site neighboring this one
        int i = diag_site + N_boundary_left;
        bool connected_to_heat_bath = false;

        // iterate over all other sites to search for metal neighbors
        for(int j = 0; j < N; j++){

            double dist = site_dist_gpu(posx[i], posy[i], posz[i],
                                        posx[j], posy[j], posz[j]);

            bool metal_neighbor = is_in_array_gpu(metals, element[j], num_metals);
            // bool contact_neighbor = j < N_boundary_left || j >= N_boundary_left + matrix_size;
            // contact neighboring instead of metal site neighboring
            
            if(dist < nn_dist && metal_neighbor){
                // site i is connected to a metal site, 
                // that is the term (i - left_boundary) or (diag_site) in the L matrix
                connected_to_heat_bath = true;
            }
        }

        // add -gamma to the diagonal if the site is connected to a heat bath
        if(connected_to_heat_bath){
            for(int j = L_row_ptr_d[diag_site]; j < L_row_ptr_d[diag_site+1]; j++)
            {
                if(diag_site == L_col_indices_d[j]){
                    L_values_d[j] = -gamma;
                }
            }
        }
    }

    // matrix needs to be populated before row sums
    __syncthreads();

    // row sum to diagonals
    for (int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        double sum = 0.0;

        // gather sum of off-diagonals
        for (int j = L_row_ptr_d[i]; j < L_row_ptr_d[i+1]; j++)
        {
            if (i != L_col_indices_d[j]){
                sum += L_values_d[j];
            }
        }

        // added sum to diagonal
        for (int j = L_row_ptr_d[i]; j < L_row_ptr_d[i+1]; j++)
        {
            if (L_col_indices_d[j] == i)
            {
                L_values_d[j] += -sum;
            }
        }
    }

}


__global__ void add_I_to_sparse_matrix(int *L_row_ptr_d, int *L_col_indices_d, double *L_values_d, int matrix_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < matrix_size; i += blockDim.x * gridDim.x)
    {
        for (int j = L_row_ptr_d[i]; j < L_row_ptr_d[i+1]; j++)
        {
            if (L_col_indices_d[j] == i)
            {
                L_values_d[j] += 1.0;
            }
        }
    }
}


void build_laplacian_gpu(cublasHandle_t handle_cublas, cusolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf, 
                         int N, int N_boundary_left, int N_boundary_right, int N_center, double nn_dist, double gamma, double step_time, double L_char)
{

    int matrix_size = N_center;

    // Compute the number of nonzeros per row of the matrix (Nsub x Nsub)
    int *nnz_per_row_d;
    gpuErrchk( cudaMalloc((void **)&nnz_per_row_d, matrix_size * sizeof(int)) );
    gpuErrchk( cudaMemset(nnz_per_row_d, 0, matrix_size * sizeof(int)) );
    
    gpuErrchk( cudaMalloc((void **)&(gpubuf.L_row_ptr_d), (matrix_size + 1) * sizeof(int)));
    gpuErrchk( cudaMemset(gpubuf.L_row_ptr_d, 0, (matrix_size + 1) * sizeof(int)) );

    int threads = 1024;
    int blocks = (matrix_size - 1) / threads + 1;
    calc_nnz_per_row_L<<<blocks, threads>>>(gpubuf.site_x, gpubuf.site_y, gpubuf.site_z, nn_dist, 
                                            N_boundary_left, N_boundary_right, matrix_size, nnz_per_row_d);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();

    void     *temp_storage_d = NULL;
    size_t   temp_storage_bytes = 0;

    // determines temporary device storage requirements for inclusive prefix sum
    cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, gpubuf.L_row_ptr_d+1, matrix_size);

    // Allocate temporary storage for inclusive prefix sum
    gpuErrchk(cudaMalloc(&temp_storage_d, temp_storage_bytes));

    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, gpubuf.L_row_ptr_d+1, matrix_size);
    
    // nnz is the same as (*row_ptr_d)[matrix_size]
    int nnz;
    gpuErrchk( cudaMemcpy(&nnz, gpubuf.L_row_ptr_d + matrix_size, sizeof(int), cudaMemcpyDeviceToHost) );
    gpubuf.L_nnz = nnz;

    gpuErrchk( cudaMalloc((void **)&(gpubuf.L_col_indices_d), nnz * sizeof(int)) );
    gpuErrchk( cudaMalloc((void **)&(gpubuf.L_values_d), nnz * sizeof(double)) );
    gpuErrchk( cudaMemset(gpubuf.L_values_d, 0, nnz * sizeof(double)) );

    // assemble the indices of L
    assemble_L_col_indices<<<blocks, threads>>>(gpubuf.site_x, gpubuf.site_y, gpubuf.site_z, nn_dist, 
                                                N_boundary_left, N_boundary_right, matrix_size, 
                                                nnz_per_row_d, gpubuf.L_row_ptr_d, gpubuf.L_col_indices_d);
    cudaDeviceSynchronize();

    // populate L
    populate_L<<<blocks, threads>>>(gpubuf.site_x, gpubuf.site_y, gpubuf.site_z, gpubuf.site_element, nn_dist, 
                                    N_boundary_left, N_boundary_right, matrix_size, 
                                    gpubuf.metal_types, gpubuf.num_metal_types_,
                                    nnz_per_row_d, gpubuf.L_row_ptr_d, gpubuf.L_col_indices_d, gpubuf.L_values_d, gamma, L_char);
    cudaDeviceSynchronize();

    // copy gpubuf.L_values_d to gpubuf.Lss_values_d for the steady-state solution
    gpuErrchk( cudaMalloc((void **)&(gpubuf.Lss_values_d), nnz * sizeof(double)) );
    gpuErrchk( cudaMemcpy(gpubuf.Lss_values_d, gpubuf.L_values_d, nnz * sizeof(double), cudaMemcpyDeviceToDevice) );

    // scale L by the step time and subtract I to get I - delta_t * L
    double factor = -step_time;
    cublasDscal(handle_cublas, nnz, &factor, gpubuf.L_values_d, 1);
    add_I_to_sparse_matrix<<<blocks, threads>>>(gpubuf.L_row_ptr_d, gpubuf.L_col_indices_d, gpubuf.L_values_d, matrix_size);
    cudaDeviceSynchronize();

    // dump_csr_matrix_txt(matrix_size, nnz, gpubuf.L_row_ptr_d, gpubuf.L_col_indices_d, gpubuf.L_values_d, 0);
    // std::cout << "dumped L\n";
    // exit(1);

    cudaFree(nnz_per_row_d);
    cudaFree(temp_storage_d);

}


double update_temperature_local_gpu(cublasHandle_t handle_cublas, cusolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf, 
                                  int N, int N_boundary_left, int N_boundary_right, int N_center, double background_temp, 
                                  double t, double tau, double k_th_interface, double k_th_vacancies, 
                                  double kmc_step_time, double nn_dist, double T_1, double L_char)
{

    double T_0 = background_temp;                                   // [K] Temperature scale
    double one = 1.0;

    // Calculate constants
    double step_time = t * tau;                                                                                                       // [a.u.]                                                               // [a.u.]
    const double p_transfer_vacancies = 1 / ((L_char * k_th_interface) * (T_1 - background_temp));                         // [1/W]
    const double p_transfer_non_vacancies = 1 / ((L_char * k_th_vacancies) * (T_1 - background_temp));                     // [1/W]

    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);
    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_DEVICE);

    // scale gpubuf.site_power 
    double factor = 1e10 * p_transfer_vacancies * step_time;
    cublasDscal(handle_cublas, N_center, &factor, gpubuf.site_power + N_boundary_left, 1);
    // note: does not treat p_transfer_vacancies and p_transfer_non_vacancies seperately

    double *d_sum_buffer, *ones;
    gpuErrchk( cudaMalloc((void **)&d_sum_buffer, N_center * sizeof(double)) );
    gpuErrchk( cudaMalloc((void **)&ones, N_center * sizeof(double)) );
    gpuErrchk( cudaMemset(d_sum_buffer, 0, N_center * sizeof(double)) );
    thrust::fill(thrust::device_pointer_cast(ones), thrust::device_pointer_cast(ones) + N_center, 1.0);

    // Transform background temperatures - (site_temperature[i] - T_0) / (T_1 - T_0);
    factor = -T_0;
    cublasDaxpy(handle_cublas, N_center, &factor, ones, 1, gpubuf.site_temperature + N_boundary_left, 1); // Add -T_0 to each element 
    factor = 1.0 / (T_1 - T_0);
    cublasDscal(handle_cublas, N_center, &factor, gpubuf.site_temperature + N_boundary_left, 1); // Scale T_vec by  1.0 / (T_1 - T_0)

    // Temperature update timestep loop
    for (int i = 0; i <= int(kmc_step_time / t); ++i)
    {
        // sum ([site_power]_Nx1 + [T_vec]_Nx1)  into d_sum_buffer
        cublasDcopy(handle_cublas, N_center, gpubuf.site_power + N_boundary_left, 1, d_sum_buffer, 1);
        cublasDaxpy(handle_cublas, N_center, &one, gpubuf.site_temperature + N_boundary_left, 1, d_sum_buffer, 1);      

        // solve the system of linear equations [laplacian]_NxN * [T_temp]_Nx1 = ([site_power]_Nx1  + [T_vec]_Nx1) 
        // not preconditioning because it changes the laplacian matrix. 
        solve_sparse_CG(handle_cublas, cusparseHandle, gpubuf.L_values_d, gpubuf.L_row_ptr_d, gpubuf.L_col_indices_d, gpubuf.L_nnz, N_center, d_sum_buffer, gpubuf.site_temperature + N_boundary_left);

    } // for (int i = 0; i <= int(step_time / p.delta_t); ++i)

    // rescale temperatures - T_temp[i] * (T_1 - T_0) + T_0;
    factor = T_1 - T_0;
    cublasDscal(handle_cublas, N_center, &factor, gpubuf.site_temperature + N_boundary_left, 1);
    factor = T_0;
    cublasDaxpy(handle_cublas, N_center, &factor, ones, 1, gpubuf.site_temperature + N_boundary_left, 1);

    std::cout << "num steps for temperature: " << int(kmc_step_time / t) << "\n"; 

    // sum all the elements in site_temperature 
    thrust::device_ptr<double> dev_ptr(gpubuf.site_temperature + N_boundary_left);
    double T_tot = thrust::reduce(dev_ptr, dev_ptr + N_center, 0.0, thrust::plus<double>());

    cusparseDestroy(cusparseHandle);
    cudaFree(d_sum_buffer);
    cudaFree(ones);
    return T_tot/N_center;

}


double update_temperature_local_steadystate_gpu(cublasHandle_t handle_cublas, cusolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf, 
                                                int N, int N_boundary_left, int N_boundary_right, int N_center, double background_temp, 
                                                double t, double tau, double k_th_interface, double k_th_vacancies, 
                                                double kmc_step_time, double nn_dist, double T_1, double L_char)
{

    double T_0 = background_temp;                                   // [K] Temperature scale
    double one = 1.0;

    // Calculate constants --> this should be Lc, not nn_dist (now these are different)
    double step_time = t * tau;                                                                                                       // [a.u.]                                                               // [a.u.]
    const double p_transfer_vacancies = 1 / ((L_char * k_th_interface) * (T_1 - background_temp));                         // [1/W]
    const double p_transfer_non_vacancies = 1 / ((L_char * k_th_vacancies) * (T_1 - background_temp));                     // [1/W]

    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);
    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_DEVICE);

    // scale gpubuf.site_power 
    double factor = 1e10 * p_transfer_vacancies;
    cublasDscal(handle_cublas, N_center, &factor, gpubuf.site_power + N_boundary_left, 1);
    // note: does not treat p_transfer_vacancies and p_transfer_non_vacancies seperately

    double *d_T_vec, *d_T_temp, *d_sum_buffer, *ones;
    gpuErrchk( cudaMalloc((void **)&d_sum_buffer, N_center * sizeof(double)) );
    gpuErrchk( cudaMalloc((void **)&ones, N_center * sizeof(double)) );
    gpuErrchk( cudaMemset(d_sum_buffer, 0, N_center * sizeof(double)) );
    thrust::fill(thrust::device_pointer_cast(ones), thrust::device_pointer_cast(ones) + N_center, 1.0);

    // Transform background temperatures - (site_temperature[i] - T_0) / (T_1 - T_0);
    factor = -T_0;
    cublasDaxpy(handle_cublas, N_center, &factor, ones, 1, gpubuf.site_temperature + N_boundary_left, 1); // Add -T_0 to each element 
    factor = 1.0 / (T_1 - T_0);
    cublasDscal(handle_cublas, N_center, &factor, gpubuf.site_temperature + N_boundary_left, 1); // Scale T_vec by  1.0 / (T_1 - T_0)

    // Solve the system of equations [laplacian_ss]_NxN * [T_temp]_Nx1 = [site_power]_Nx1
    // not preconditioning because it changes the laplacian matrix. 
    cublasDcopy(handle_cublas, N_center, gpubuf.site_power + N_boundary_left, 1, d_sum_buffer, 1);
    solve_sparse_CG(handle_cublas, cusparseHandle, gpubuf.Lss_values_d, gpubuf.L_row_ptr_d, gpubuf.L_col_indices_d, gpubuf.L_nnz, N_center, d_sum_buffer, gpubuf.site_temperature + N_boundary_left);
    cudaDeviceSynchronize();

    // rescale temperatures - -T_temp[i] * (T_1 - T_0) + T_0;
    factor = -1*(T_1 - T_0);
    cublasDscal(handle_cublas, N_center, &factor, gpubuf.site_temperature + N_boundary_left, 1);
    factor = T_0;
    cublasDaxpy(handle_cublas, N_center, &factor, ones, 1, gpubuf.site_temperature + N_boundary_left, 1);

    // sum all the elements in site_temperature 
    thrust::device_ptr<double> dev_ptr(gpubuf.site_temperature + N_boundary_left);
    double T_tot = thrust::reduce(dev_ptr, dev_ptr + N_center, 0.0, thrust::plus<double>());

    cusparseDestroy(cusparseHandle);
    cudaFree(d_sum_buffer);
    cudaFree(ones);
    return T_tot/N_center;

}