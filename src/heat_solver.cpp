#include "Device.h"

void Device::constructLaplacian(KMCParameters &p, cublasHandle_t handle_cublas, cusolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf)
{
    std::cout << "Constructing graph Laplacian..." << std::endl;

    int N_boundary_left = p.num_atoms_contact;                                          // # atoms in the right boundary 
    int N_boundary_right = p.num_atoms_contact;                                         // # atoms in the left contact (excluding Ti)
    N_center = N - N_boundary_left - N_boundary_right;                                  // # atoms in the center device

    // Constants
    double gamma = 1 / (p.delta * ((p.k_th_interface / p.k_th_metal) + 1)); // [a.u.] 
    double step_time = p.delta_t * p.tau;                                   // [a.u.]

    gpubuf.sync_HostToGPU(*this); // this one is needed, it's done before the first hostToGPU sync for a given bias point

    build_laplacian_gpu(handle_cublas, handle_cusolver, gpubuf, 
                        N, N_boundary_left, N_boundary_right, N_center, 
                        nn_dist, gamma, step_time, p.L_char);

    gpubuf.sync_GPUToHost(*this); 

    std::cout << "Constructed."  << std::endl;

}

// construct laplacian and steady state laplacian
void Device::constructLaplacian(KMCParameters &p)
{
    print("Constructing graph Laplacian...");

    int N_boundary_left = p.num_atoms_contact;                                          // # atoms in the right boundary 
    int N_boundary_right = p.num_atoms_contact;                                         // # atoms in the left contact (excluding Ti)
    N_center = N - N_boundary_left - N_boundary_right;                                  // # atoms in the center device

    // Initialize the laplacian and inverse of the laplacian
    laplacian.resize(N_center * N_center);                                              // Laplacian (connectivity) matrix 
    laplacian_ss.resize(N_center * N_center);                                           // Laplacian modified for steady state

    // Inverse laplacian
    int *ipiv_L_T = (int *)malloc(N_center * sizeof(int));
    int lwork = N_center;
    double *work = (double *)malloc(N_center * sizeof(double));
    int N_test = N_center;

    // Inverse steady state laplacian
    int *ipiv_L_ss_T = (int *)malloc(N_center * sizeof(int));
    int lwork_ss = N_center;
    double *work_ss = (double *)malloc(N_center * sizeof(double));

    double *L = (double *)calloc(N_center * N_center, sizeof(double));                  // Laplacian
    double *L_ss = (double *)calloc(N_center * N_center, sizeof(double));               // Steady state lapalacian
    double *L_inv = (double *)calloc(N_center * N_center, sizeof(double));              // Inverse of the laplacian

    // Initialize B for to calculate the inverse
    double *B_L = (double *)calloc(N_center * N_center, sizeof(double));

    // Initialize B for to calculate the inverse
    double *B_L_ss = (double *)calloc(N_center * N_center, sizeof(double));

    for (int i = 0; i < N_center; i++)
    {
        B_L[i * N_center + i] = 1.0;    // setting the diagonal elements to 1 to make it an identity matrix
        B_L_ss[i * N_center + i] = 1.0; // setting the diagonal elements to 1 to make it an identity matrix
    }

    int info;

    // Calculate constants
    double gamma = 1 / (p.delta * ((p.k_th_interface / p.k_th_metal) + 1)); // [a.u.] 
    double step_time = p.delta_t * p.tau;                                   // [a.u.]

// Build laplacian matrix - the matrix indexes the atoms which are not in the contacts

#pragma omp parallel for
    for (int i = N_boundary_left; i < N - N_boundary_right; i++)
    {
        int index_i, index_j;
        index_i = i - N_boundary_left;

        for (int j : site_neighbors.l[i])
        {

            index_j = j - N_boundary_left;

            if ( (i != j) && (j >= N_boundary_left) && (j < (N - N_boundary_right)) ) // Neighbouring site not in the contacts
            {
                L[N_center * index_i + index_j] = 1;
            }

            // if connected to the contacts, add a dissipating term to the diagonal
            bool connected_to_heat_bath = is_in_vector<ELEMENT>(p.metals, site_element[j]);
            if (connected_to_heat_bath) 
            {
                L[N_center * index_i + index_i] = -gamma;
            }
            
        } // j

    } // i

// sum matrix rows into diagonals
#pragma omp parallel for
    for (int i = 0; i < N_center; i++)
    {
        for (int j = 0; j < N_center; j++)
        {
            if (i != j)
            {
                L[N_center * i + i] += -L[N_center * i + j];
            }
        }
    }

// Construct (I-L*time_step)

// Prepare L_T to solve for the inverse of the [unity - laplacian] matrix (I-delta_t*L)
#pragma omp parallel for 
    for (int i = 0; i < N_center; i++)
    {
        for (int j = 0; j < N_center; j++)
        {
            L_inv[i * N_center + j] = -step_time * L[i * N_center + j];
        }
        L_inv[N_center * i + i] += 1; // subtract from identity matrix
    }

// Prepare Lss to solve for the inverse of the laplacian matrix (Lss)
#pragma omp parallel for collapse(2) num_threads(1)
    for (int i = 0; i < N_center; i++)
    {
        for (int j = 0; j < N_center; j++)
        {
            L_ss[i * N_center + j] = L[i * N_center + j];
        }
    }

#ifdef USE_CUDA

    gesv(&N_center, &N_center, L_inv, &N_center, ipiv_L_T, B_L, &N_center, &info);      // B_L = L^-1
    gesv(&N_center, &N_center, L_ss, &N_center, ipiv_L_ss_T, B_L_ss, &N_center, &info); // B_L_ss = L_ss^-1

#else
    //  LU factorization of (I-L) (overwrite L_T with the factorization)
    dgetrf_(&N_center, &N_center, L_inv, &N_center, ipiv_L_T, &info);

    // LU factorization of (L) (overwrite L_T with the factorization)
    dgetrf_(&N_center, &N_center, L_ss, &N_center, ipiv_L_ss_T, &info);

    // Compute the inverse of the matrix L_T using the LU factorization (overwrite A with the factorization)
    dgetri_(&N_center, L_inv, &N_center, ipiv_L_T, work, &lwork, &info);

    // Prepare Lss to solve for the inverse of the laplacian matrix (Lss)
#pragma omp parallel for collapse(2) num_threads(1)
    for (int i = 0; i < N_center; i++)
    {
        for (int j = 0; j < N_center; j++)
        {
            B_L[i * N_center + j] = L_inv[i * N_center + j];
        }
    }

    // Compute the inverse of the matrix L_T using the LU factorization (overwrite A with the factorization)
    dgetri_(&N_center, L_ss, &N_center, ipiv_L_ss_T, work_ss, &lwork_ss, &info);

    // Prepare Lss to solve for the inverse of the laplacian matrix (Lss)
#pragma omp parallel for collapse(2) num_threads(1)
    for (int i = 0; i < N_center; i++)
    {
        for (int j = 0; j < N_center; j++)
        {
            B_L_ss[i * N_center + j] = L_ss[i * N_center + j];
        }
    }
    print("Assembling Laplacian on the CPU");

#endif

    // Update the inverse of the laplacian and steady state laplacian
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N_center; i++)
    {
        for (int j = 0; j < N_center; j++)
        {
            laplacian[i * N_center + j] = B_L[i * N_center + j];
        }
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < N_center; i++)
    {
        for (int j = 0; j < N_center; j++)
        {
            laplacian_ss[i * N_center + j] = B_L_ss[i * N_center + j];
        }
    }

    free(ipiv_L_T);
    free(L);
    free(ipiv_L_ss_T);
    free(work_ss);
    free(work);
    free(L_ss);
    free(B_L);
    free(B_L_ss);
}


// update temperature on the CPU
std::map<std::string, double> Device::updateTemperature(cublasHandle_t handle, cusolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf, KMCParameters &p, double step_time)
{
#ifdef USE_CUDA

    // gpubuf.sync_HostToGPU(*this); // remove eventually
    if (p.solve_heating_global)
    {
        std::cout << "note: global temperature model is currently computed on the cpu\n";
        std::map<std::string, double> result;

        // get the site power vector from the gpu
        gpubuf.copy_power_fromGPU(site_power);

        // compute the global temperature on the cpu
        updateTemperatureGlobal(step_time, p.small_step, p.dissipation_constant,
                                p.background_temp, p.t_ox, p.A, p.c_p, result);
        
        // update the global temperature in gpu memory
        gpubuf.copy_Tbg_toGPU(this->T_bg);
        result["Global temperature [K]"] = this->T_bg;
        return result;
        
        
        // double C_thermal = p.A * p.t_ox * p.c_p * (1e6); // [J/K]
        // double number_steps = step_time / p.small_step;
        // double a_coeff = -p.dissipation_constant*1/C_thermal*p.small_step + 1;
        // double b_coeff = p.dissipation_constant*1/C_thermal*p.small_step*p.background_temp; 
        // // call CUDA implementation
        // update_temperatureglobal_gpu(gpubuf.site_power, gpubuf.T_bg, gpubuf.N_, a_coeff, b_coeff, number_steps, C_thermal, p.small_step);
    }
    else if (p.solve_heating_local)
    {
        auto t0 = std::chrono::steady_clock::now();

        std::map<std::string, double> TemperatureMap;
        int N_boundary_left = p.num_atoms_contact;
        int N_boundary_right = p.num_atoms_contact;         // # atoms in the left contact (excluding Ti)
        N_center = N - N_boundary_left - N_boundary_right; 

        // Evaluate steady-state condition
        if (step_time > 1e3 * p.delta_t)                // steady state
        {
            this->T_bg = update_temperature_local_steadystate_gpu(handle, handle_cusolver, gpubuf, N, N_boundary_left, N_boundary_right, N_center, 
                                                                  p.background_temp, p.delta_t, p.tau, p.k_th_interface,
                                                                  p.k_th_vacancies, step_time, nn_dist, T_1, p.L_char);
                                                        
            std::cout << std::fixed << std::setprecision(16) << "Global (ss) temperature [K]: " << this->T_bg << "\n";
        } 
        else 
        {
            this->T_bg = update_temperature_local_gpu(handle, handle_cusolver, gpubuf, N, N_boundary_left, N_boundary_right, N_center, 
                                                    p.background_temp, p.delta_t, p.tau, p.k_th_interface,
                                                    p.k_th_vacancies, step_time, nn_dist, T_1, p.L_char);

            std::cout << std::fixed << std::setprecision(16) << "Global (non-ss) temperature [K]: " << this->T_bg << "\n";
        }

        auto t1 = std::chrono::steady_clock::now();
        std::chrono::duration<double> dt = t1 - t0;
        TemperatureMap["Z - calculation time - local temperature [s]"] = dt.count();
        TemperatureMap["Global temperature [K]"] = this->T_bg;

        return TemperatureMap;
    }
    // gpubuf.sync_GPUToHost(*this); // remove eventually

#else

    if (p.solve_heating_global)
    {
        std::map<std::string, double> result;

        // get the site power vector from the gpu
        gpubuf.copy_power_fromGPU(site_power);

        // compute the global temperature on the cpu
        updateTemperatureGlobal(step_time, p.small_step, p.dissipation_constant,
                                p.background_temp, p.t_ox, p.A, p.c_p, result);
        
        // update the global temperature in gpu memory
        gpubuf.copy_Tbg_toGPU(this->T_bg);
        result["Global temperature [K]"] = this->T_bg;
        return result;
    }
    else if (p.solve_heating_local)
    {
        auto t0 = std::chrono::steady_clock::now();

        std::map<std::string, double> TemperatureMap;

        gpubuf.sync_GPUToHost(*this);                  // remove once full while loop is completed

        int N_boundary_left = p.num_atoms_contact;
        int N_boundary_right = p.num_atoms_contact;         // # atoms in the left contact (excluding Ti)
        N_center = N - N_boundary_left - N_boundary_right; 

        // Evaluate steady-state condition
        if (step_time > 1e3 * p.delta_t)                // steady state
        {
            TemperatureMap = updateLocalTemperatureSteadyState(handle, p.background_temp, p.delta_t, p.tau, p.power_adjustment_term, p.k_th_interface,
                                                               p.k_th_vacancies, p.num_atoms_contact, p.metals, N_boundary_left, N_boundary_right, N_center);
            std::cout << std::fixed << std::setprecision(16) << "Global (ss) temperature [K]: " << this->T_bg << "\n";
        }
        else                                            // non-steady state
        {
            TemperatureMap = updateLocalTemperature(handle, p.background_temp, p.delta_t, p.tau, p.power_adjustment_term, p.k_th_interface,
                                                    p.k_th_vacancies, p.num_atoms_contact, p.metals, step_time, N_boundary_left, N_boundary_right, N_center);
            std::cout << std::fixed << std::setprecision(16) << "Global (non-ss) temperature [K]: " << this->T_bg << "\n";
        }

        gpubuf.sync_HostToGPU(*this); // remove once full while loop is completed

        auto t1 = std::chrono::steady_clock::now();
        std::chrono::duration<double> dt = t1 - t0;
        TemperatureMap["Z - calculation time - local temperature [s]"] = dt.count();

        return TemperatureMap;
    }

#endif
}


// update the global temperature using the global temperature model
void Device::updateTemperatureGlobal(double event_time, double small_step, double dissipation_constant,
                                     double background_temp, double t_ox, double A, double c_p, std::map<std::string, double> result)
{
    
    // std::map<std::string, double> result;

    double C_thermal = A * t_ox * c_p * (1e6);                                              // [J/K]
    double P_tot = 0.0;                                                                     // total dissipated power
    #pragma omp parallel for reduction(+ : P_tot)
    for (int i = 0; i < N; i++)
    {
        P_tot += site_power[i];
    }

    // // Solve the capacitative heat equation (analytical) - does not work mathematically
    // double a_coeff = dissipation_constant/C_thermal;                      
    // double c_coeff = (dissipation_constant/C_thermal)*T_bg + (1/C_thermal) * P_tot; 
    // double T_intermediate = (c_coeff/a_coeff) + (T_bg - c_coeff/a_coeff)*exp(-a_coeff * event_time);
    // this->T_bg = T_intermediate;

    // Solve the capacitative heat equation numerically (geometric sum)
    double number_steps = event_time / small_step;                                          // number of discrete steps within the KMC event time
    double a_coeff = -dissipation_constant*1/C_thermal*small_step + 1;                      // geometric series coefficients
    double b_coeff = dissipation_constant*1/C_thermal*small_step*background_temp; 
    double c_coeff = b_coeff + P_tot/C_thermal * small_step;
    double T_intermediate = T_bg;
    double T_bg_test = c_coeff * (1.0 - pow(a_coeff, number_steps)) / (1.0-a_coeff) + pow(a_coeff, number_steps) * T_intermediate;
    this->T_bg = T_bg_test;

    // result["Global temperature [K]"] = T_bg;
    result["Total dissipated power [mW]"] = P_tot*1e3;
    std::cout << std::fixed << std::setprecision(16) << "Global temperature [K]: " << T_bg << "\n";

    // return result;
}


// update the local and global temperature
std::map<std::string, double> Device::updateLocalTemperature(cublasHandle_t handle, double background_temp, double t, double tau, double power_adjustment_term, double k_th_interface,
                                                             double k_th_vacancies, double num_atoms_contact, std::vector<ELEMENT> metals, double kmc_step_time, int N_boundary_left, int N_boundary_right, int N_center)
{

    // Map
    std::map<std::string, double> result;

    double T_tot = 0.0;                                             // [K] Background temperature
    double T_0 = background_temp;                                   // [K] Temperature scale
    double *T_vec = (double *)calloc(N, sizeof(double));            // Normalized temperatures
    double *T_temp = (double *)calloc(N_center, sizeof(double)); 
    double T_transf;
    
    // gemv and axpy parameters
    int incx = 1;                        // Increment for the elements of vector x
    int incy = 1;                        // Increment for the elements of vector y
    double alpha = 1.0;                  // Scaling factor for the matrix-vector product
    double beta = 0.0;                   // Scaling factor for vector y
    double one = 1.0;

    // Calculate constants
    double step_time = t * tau;                                                                                                       // [a.u.]                                                               // [a.u.]
    const double p_transfer_vacancies = 1 / ((nn_dist * (1e-10) * k_th_interface) * (T_1 - background_temp));                         // [1/W]
    const double p_transfer_non_vacancies = 1 / ((nn_dist * (1e-10) * k_th_vacancies) * (T_1 - background_temp));                     // [1/W]

// scale the power vector
#pragma omp parallel for
    for (int i = N_boundary_left; i < N - N_boundary_right; i++)
    {
        if (site_element[i] == VACANCY)
        {
            site_power[i] *= 10000000000 * p_transfer_vacancies * step_time;
        }
        else
        {
            site_power[i] *= 10000000000 * p_transfer_non_vacancies * step_time;
        }
    }

    double *d_laplacian, *d_T_vec, *d_site_power, *d_T_temp, *d_sum_buffer;
    cudaMalloc(&d_laplacian, N_center * N_center * sizeof(double));
    cudaMalloc(&d_T_vec, N_center * sizeof(double));
    cudaMalloc(&d_site_power, N_center * sizeof(double));
    cudaMalloc(&d_T_temp, N_center * sizeof(double));
    cudaMalloc(&d_sum_buffer, N_center * sizeof(double));
    cudaMemcpy(d_laplacian, laplacian.data(), N_center * N_center * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_site_power, site_power.data() + N_boundary_left, N_center * sizeof(double), cudaMemcpyHostToDevice);

    for (int i = 0; i <= int(kmc_step_time / t); ++i)
    {

        // Transform background temperatures
    #pragma omp parallel for
        for (int i = N_boundary_left; i < N - N_boundary_right; i++)
        {
            T_vec[i] = (site_temperature[i] - T_0) / (T_1 - T_0);
        }

        // Compute the matrix-vector product [T_temp]_Nx1 = [laplacian_inv]_NxN * ( [site_power]_Nx1  + [T_vec]_Nx1) - CUDA version
        cudaMemcpy(d_T_vec, T_vec + N_boundary_left, N_center * sizeof(double), cudaMemcpyHostToDevice);
        CheckCublasError( cublasDcopy(handle, N_center, d_T_vec, 1, d_sum_buffer, 1) );             
        CheckCublasError( cublasDaxpy(handle, N_center, &one, d_site_power, 1, d_sum_buffer, 1) );   
        CheckCublasError( cublasDgemv(handle, CUBLAS_OP_T, N_center, N_center, &alpha, d_laplacian, N_center, d_sum_buffer, incx, &beta, d_T_temp, incy) );
        cudaDeviceSynchronize();
        cudaMemcpy(T_temp, d_T_temp, N_center * sizeof(double), cudaMemcpyDeviceToHost);

        // Update the temperature of the internal nodes (transform back to normal temperature scale)
    #pragma omp parallel for
        for (int i = 0; i < N_center; i++)
        {
            site_temperature[i + N_boundary_left] = T_temp[i] * (T_1 - T_0) + T_0;
        }

    } // for (int i = 0; i <= int(step_time / p.delta_t); ++i)

    std::cout << "num steps for temperature: " << int(kmc_step_time / t) << "\n"; 

    // Update the global temperature
#pragma omp parallel
   {
#pragma omp for reduction(+ : T_tot)
        for (int i = 0; i < N; i++)     // for (int i = num_atoms_contact; i < N - num_atoms_contact; i++)
        {
            T_tot += site_temperature[i];
        }
   }

    this->T_bg = T_tot / N; // this->T_bg = T_tot / (N - 2*num_atoms_contact);
    result["Global (non-ss) temperature [K]"] = this->T_bg;
    free(T_vec);
    free(T_temp);

    cudaFree(d_laplacian);
    cudaFree(d_T_vec);
    cudaFree(d_site_power);
    cudaFree(d_T_temp);
    cudaFree(d_sum_buffer);

    return result;
}


// update the local and global temperature in steady state
std::map<std::string, double> Device::updateLocalTemperatureSteadyState(cublasHandle_t handle, double background_temp, double delta_t, double tau, double power_adjustment_term, double k_th_interface,
                                                                        double k_th_vacancies, double num_atoms_contact, std::vector<ELEMENT> metals, int N_boundary_left, int N_boundary_right, int N_center)
{
    std::map<std::string, double> result;

    double T_tot = 0.0;           // [K] Background temperature
    double T_0 = background_temp; // [K] Temperature scale
    double T_transf;

    // Calculate constants
    double step_time = delta_t * tau;                                                                                                 // [a.u.]                                                               // [a.u.]
    const double p_transfer_vacancies = 1 / ((nn_dist * (1e-10) * k_th_interface) * (T_1 - background_temp));                         // [a.u.]
    const double p_transfer_non_vacancies = 1 / ((nn_dist * (1e-10) * k_th_vacancies) * (T_1 - background_temp));                     // [a.u.]

    // scale the power vector
#pragma omp parallel for
    for (int i = N_boundary_left; i < N - N_boundary_right; i++)
    {
        if (site_element[i] == VACANCY)
        {
            site_power[i] *= 10000000000 * p_transfer_vacancies;
        }
        else
        {
            site_power[i] *= 10000000000 * p_transfer_non_vacancies;
        }
    }

    double *T_temp = (double *)calloc(N_center, sizeof(double)); // Normalized temperatures

    // Compute the matrix-vector product [T_temp]_Nx1 = [laplacian_ss_inv]_NxN * [site_power]_Nx1
    double *d_laplacian_ss, *d_site_power_offset, *d_T_temp;
    gpuErrchk( cudaMalloc(&d_laplacian_ss, N_center * N_center * sizeof(double)) );
    gpuErrchk( cudaMalloc(&d_site_power_offset, N_center * sizeof(double)) );
    gpuErrchk( cudaMalloc(&d_T_temp, N_center * sizeof(double)) );
    gpuErrchk( cudaMemcpy(d_laplacian_ss, laplacian_ss.data(), N_center * N_center * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_site_power_offset, site_power.data() + N_boundary_left, N_center * sizeof(double), cudaMemcpyHostToDevice) );

    int incx = 1;                        // Increment for the elements of vector x
    int incy = 1;                        // Increment for the elements of vector y
    double alpha = 1.0;                  // Scaling factor for the matrix-vector product
    double beta = 0.0;                   // Scaling factor for vector y

    CheckCublasError( cublasDgemv(handle, CUBLAS_OP_T, N_center, N_center, &alpha,
                                 d_laplacian_ss, N_center,
                                 d_site_power_offset, incx,
                                 &beta, d_T_temp, incy) );
    cudaDeviceSynchronize();
    cudaMemcpy(T_temp, d_T_temp, N_center * sizeof(double), cudaMemcpyDeviceToHost);

    // Update the temperature of the internal nodes (transform back to normal temperature scale)
#pragma omp parallel for
    for (int i = 0; i < N_center; i++)
    {
        site_temperature[i + N_boundary_left] = -T_temp[i] * (T_1 - T_0) + T_0;
    }

    // Update the global temperature
#pragma omp parallel
   {
#pragma omp for reduction(+ : T_tot)
        for (int i = 0; i < N; i++)  
        {
            T_tot += site_temperature[i];
        }
   }

    T_bg = T_tot / N;
    result["Global (ss) temperature [K]"] = T_bg;

    free(T_temp);
    cudaFree(d_laplacian_ss);
    cudaFree(d_site_power_offset);
    cudaFree(d_T_temp);

    return result;
}

// used, but can repurpose for simliar tasks
// do not use, because the laplacian (which is actually the inverse) is dense
std::map<std::string, double> Device::updateLocalTemperature_sparse(cublasHandle_t handle, double background_temp, double t, double tau, double power_adjustment_term, double k_th_interface,
                                                             double k_th_vacancies, double num_atoms_contact, std::vector<ELEMENT> metals, double kmc_step_time, int N_boundary_left, int N_boundary_right, int N_center)
{

    std::map<std::string, double> result;

    double T_tot = 0.0;                                             // [K] Background temperature
    double T_0 = background_temp;                                   // [K] Temperature scale
    double *T_vec = (double *)calloc(N, sizeof(double));            // Normalized temperatures
    double *T_temp = (double *)calloc(N_center, sizeof(double)); 
    double T_transf;
    
    // gemv and axpy parameters
    int incx = 1;                        // Increment for the elements of vector x
    int incy = 1;                        // Increment for the elements of vector y
    double alpha = 1.0;                  // Scaling factor for the matrix-vector product
    double beta = 0.0;                   // Scaling factor for vector y

    // Calculate constants
    double step_time = t * tau;                                                                                                       // [a.u.]                                                               // [a.u.]
    const double p_transfer_vacancies = 1 / ((nn_dist * (1e-10) * k_th_interface) * (T_1 - background_temp));                         // [1/W]
    const double p_transfer_non_vacancies = 1 / ((nn_dist * (1e-10) * k_th_vacancies) * (T_1 - background_temp));                     // [1/W]

// scale the power vector
#pragma omp parallel for
    for (int i = N_boundary_left; i < N - N_boundary_right; i++)
    {
        if (site_element[i] == VACANCY)
        {
            site_power[i] *= 10000000000 * p_transfer_vacancies * step_time;
        }
        else
        {
            site_power[i] *= 10000000000 * p_transfer_non_vacancies * step_time;
        }
    }

    double *d_laplacian, *d_T_vec, *d_site_power, *d_T_temp, *d_sum_buffer;
    gpuErrchk( cudaMalloc((void**)&d_laplacian, N_center * N_center * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_T_vec, N_center * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_site_power, N_center * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_T_temp, N_center * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_sum_buffer, N_center * sizeof(double)) );
    gpuErrchk( cudaMemcpy(d_laplacian, laplacian.data(), N_center * N_center * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_site_power, site_power.data() + N_boundary_left, N_center * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemset(d_sum_buffer, 0, N_center * sizeof(double)) );
    gpuErrchk( cudaMemset(d_T_temp, 0, N_center * sizeof(double)) );

    cusparseHandle_t cusparse_handle;
    cusparseCreate(&cusparse_handle);
    cusparseStatus_t status;

    cusparseDnMatDescr_t dnMatA;
    cusparseSpMatDescr_t spMatA;
    int *csrRowPtr;
    int *csrColInd;
    double *csrVal;
    cudaMalloc((void**)&csrRowPtr, (N_center + 1) * sizeof(int));

    // Create dense and sparse matrices
    status = cusparseCreateDnMat(&dnMatA, N_center, N_center, N_center, d_laplacian, CUDA_R_64F, CUSPARSE_ORDER_ROW);
    status = cusparseCreateCsr(&spMatA, N_center, N_center, 0,
                               csrRowPtr, NULL, NULL,
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                               CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    // Get buffer for csr conversion
    void* buffer = NULL;
    size_t bufferSize = 0;
    status = cusparseDenseToSparse_bufferSize(cusparse_handle, dnMatA, spMatA,
                                              CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                              &bufferSize);
    gpuErrchk( cudaMalloc((void**)&buffer, sizeof(double) * bufferSize) );

    // do analysis of sparse matrix
    status = cusparseDenseToSparse_analysis(cusparse_handle, dnMatA, spMatA,
                                            CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                            buffer);

    // Get sparsity pattern:
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    status = cusparseSpMatGetSize(spMatA, &num_rows_tmp, &num_cols_tmp, &nnz);
    nnz = static_cast<int>(nnz);
    // std::cout << " nnz: " << nnz << "\n";

    // set pointers:
    cudaMalloc((void**)&csrColInd, nnz * sizeof(int));
    cudaMalloc((void**)&csrVal,  nnz * sizeof(double));
    status = cusparseCsrSetPointers(spMatA, csrRowPtr, csrColInd, csrVal);

    // execute Sparse to Dense conversion
    status = cusparseDenseToSparse_convert(cusparse_handle, dnMatA, spMatA,
                                           CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                           buffer);

    dump_csr_matrix_txt(N_center, nnz, csrRowPtr, csrColInd, csrVal, 0);
    std::cout << "dumped sparse laplacian matrix\n";

    size_t MVBufferSize;
    void *MVBuffer;
    double *one_d, *zero_d;
    double one = 1.0;
    double zero = 0.0;
    gpuErrchk( cudaMalloc((void**)&one_d, sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&zero_d, sizeof(double)) );
    gpuErrchk( cudaMemcpy(one_d, &one, sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(zero_d, &zero, sizeof(double), cudaMemcpyHostToDevice) );
    
    // Create dense vectors for SpMV
    cusparseDnVecDescr_t vec_sum_buffer, vec_T_temp;
    status = cusparseCreateDnVec(&vec_sum_buffer, N_center, d_sum_buffer, CUDA_R_64F);
    status = cusparseCreateDnVec(&vec_T_temp, N_center, d_T_temp, CUDA_R_64F);

    // Create buffer for matrix-vector multiplication
    status = cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, spMatA, 
                                     vec_sum_buffer, zero_d, vec_T_temp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &MVBufferSize);  
    gpuErrchk( cudaMalloc((void**)&MVBuffer, sizeof(double) * MVBufferSize) );

    // temperature update loop
    for (int i = 0; i <= int(kmc_step_time / t); ++i)
    {
        // Transform background temperatures
    #pragma omp parallel for
        for (int i = N_boundary_left; i < N - N_boundary_right; i++)
        {
            T_vec[i] = (site_temperature[i] - T_0) / (T_1 - T_0);
        }

        // Compute the matrix-vector product [T_temp]_Nx1 = [laplacian]_NxN * ( [site_power]_Nx1  + [T_vec]_Nx1) - CUDA version
        cudaMemcpy(d_T_vec, T_vec + N_boundary_left, N_center * sizeof(double), cudaMemcpyHostToDevice);
        CheckCublasError( cublasDcopy(handle, N_center, d_T_vec, 1, d_sum_buffer, 1) );             
        CheckCublasError( cublasDaxpy(handle, N_center, &one, d_site_power, 1, d_sum_buffer, 1) ); 

        cudaDeviceSynchronize();
        status = cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, spMatA,                         
                              vec_sum_buffer, zero_d, vec_T_temp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, MVBuffer);   

        if (status != CUSPARSE_STATUS_SUCCESS)
        {
            std::cerr << "cusparseSpMV_bufferSize failed in local temperature solver." << std::endl;
        }
        cudaDeviceSynchronize();

        cudaMemcpy(T_temp, d_T_temp, N_center * sizeof(double), cudaMemcpyDeviceToHost);

        // Update the temperature of the internal nodes (transform back to normal temperature scale)
        // can remove the HostToDevice copy and use the device pointer directly
    #pragma omp parallel for
        for (int i = 0; i < N_center; i++)
        {
            site_temperature[i + N_boundary_left] = T_temp[i] * (T_1 - T_0) + T_0;
        }

    } // for (int i = 0; i <= int(step_time / p.delta_t); ++i)

    // Destroy cuSPARSE handle and descriptors
    cusparseDestroy(cusparse_handle);
    cusparseDestroyDnMat(dnMatA);
    cusparseDestroySpMat(spMatA);
    cudaFree(buffer);

//
    std::cout << "num steps for temperature: " << int(kmc_step_time / t) << "\n"; 

    // Update the global temperature
#pragma omp parallel
   {
#pragma omp for reduction(+ : T_tot)
        for (int i = 0; i < N; i++)     // for (int i = num_atoms_contact; i < N - num_atoms_contact; i++)
        {
            T_tot += site_temperature[i];
        }
   }

    this->T_bg = T_tot / N; // this->T_bg = T_tot / (N - 2*num_atoms_contact);
    result["Global (non-ss) temperature [K]"] = this->T_bg;
    free(T_vec);
    free(T_temp);
    cudaFree(d_laplacian);
    return result;
}
