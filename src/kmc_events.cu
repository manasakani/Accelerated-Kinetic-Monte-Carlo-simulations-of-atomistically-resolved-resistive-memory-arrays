#include "gpu_solvers.h"
#include <omp.h>

// Constants needed:
constexpr double kB = 8.617333262e-5;           // [eV/K]

#define NUM_THREADS 512
#define MAX_NUM_LAYERS 5

// in GPU cache
__constant__ double E_gen_const[MAX_NUM_LAYERS];
__constant__ double E_rec_const[MAX_NUM_LAYERS];
__constant__ double E_Vdiff_const[MAX_NUM_LAYERS];
__constant__ double E_Odiff_const[MAX_NUM_LAYERS];

void get_gpu_info(char *gpu_string, int dev){
    struct cudaDeviceProp dprop;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(dev);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error or exit the program
        exit(EXIT_FAILURE);
    }

    cudaGetDeviceProperties(&dprop, dev);
    strcpy(gpu_string,dprop.name);
}

void set_gpu(int dev){
 cudaSetDevice(dev);
}

// builds only part of the event list
// from start_i to start_i + size_i
__global__ void build_event_list_split(const int N, const int size_i, const int start_i,
                                       const int nn, const int *neigh_idx, 
                                       const int *layer, const double *lattice, const int pbc, 
                                       const double *T_bg, const double *freq, const double *sigma, const double *k, 
                                       const double *posx, const double *posy, const double *posz,
                                       const double *potential_charge, const double *temperature,
                                       const ELEMENT *element, const int *charge, EVENTTYPE *event_type, double *event_prob)
{
    int total_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int id = total_tid; id < size_i * nn; id += total_threads) {
        EVENTTYPE event_type_ = NULL_EVENT;
        double P = 0.0;

        int idx = id + start_i * nn;
        int i = idx / nn;
        int j = neigh_idx[idx];

        double epsilon = 1e-200; // for exponential overflow

        // condition for neighbor existing
        if (j >= 0 && j < N) {
            double dist = 1e-10 * site_dist_gpu(posx[i], posy[i], posz[i], 
                                                posx[j], posy[j], posz[j], 
                                                lattice[0], lattice[1], lattice[2], pbc);

            // Generation
            if (element[i] == DEFECT && element[j] == O_EL)
            {

                double E = 2 * ((potential_charge[i]) - (potential_charge[j]));
                double zero_field_energy = E_gen_const[layer[j]]; 
                event_type_ = VACANCY_GENERATION;
                double Ekin = 1.5 * kB * (temperature[j] - (*T_bg)); //kB * (temperature[j] - temperature[i]);
                double EA = zero_field_energy - E - Ekin;
                // P = exp(-1 * EA / (kB * (*T_bg))) * (*freq);
                P = (*freq) * (1 / (exp(EA / (kB * (*T_bg))) + epsilon) );
            }

            // Recombination
            if (element[i] == OXYGEN_DEFECT && element[j] == VACANCY) 
            {
                int charge_abs = 2;
                double self_int_V = v_solve_gpu(dist, charge_abs, sigma, k);

                int charge_state = charge[i] - charge[j];
                double E = charge_state * ((potential_charge[i]) - (potential_charge[j]) + (charge_state / 2) * self_int_V);
                double zero_field_energy = E_rec_const[layer[j]];

                event_type_ = VACANCY_RECOMBINATION;
                double Ekin = 1.5 * kB * (temperature[i] - (*T_bg));
                double EA = zero_field_energy - E - Ekin;
                P = (*freq) * (1 / (exp(EA / (kB * (*T_bg))) + epsilon) );
            }

            // Vacancy diffusion
            if (element[i] == VACANCY && element[j] == O_EL)
            {

                double self_int_V = 0.0;
                if (charge[i] != 0)
                {
                    self_int_V = v_solve_gpu(dist, charge[i], sigma, k);
                }

                event_type_ = VACANCY_DIFFUSION;
                double E = (charge[i] - charge[j]) * ((potential_charge[i]) - (potential_charge[j]) + self_int_V);
                double zero_field_energy = E_Vdiff_const[layer[j]];  
                double Ekin = 1.5 * kB * (temperature[i] - (*T_bg));
                double EA = zero_field_energy - E - Ekin;
                P = (*freq) * (1 / (exp(EA / (kB * (*T_bg))) + epsilon) );
            }

            // Ion diffusion
            if (element[i] == OXYGEN_DEFECT && element[j] == DEFECT)
            {
                int charge_abs = 2;
                double self_int_V = 0.0;
                if (charge[i] != 0)
                {                    
                    self_int_V = v_solve_gpu(dist, charge_abs, sigma, k);
                }

                // double E = (charge[i] - charge[j]) * ((potential_charge[i]) - (potential_charge[j]) - self_int_V);
                double E = (charge[i] - charge[j]) * ((potential_charge[i]) - (potential_charge[j]) + self_int_V);
                double zero_field_energy = E_Odiff_const[layer[j]];

                event_type_ = ION_DIFFUSION;
                double Ekin = 1.5 * kB * (temperature[i] - (*T_bg));
                double EA = zero_field_energy - E - Ekin;
                P = (*freq) * (1 / (exp(EA / (kB * (*T_bg))) + epsilon) );
            }
        }
        event_type[id] = event_type_;
        event_prob[id] = P;
    }
}


__global__ void zero_out_events(EVENTTYPE *event_type, double *event_prob, const int *neigh_idx, int N, int nn, int i_to_delete, int j_to_delete){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    int j;
    for (int id = idx; id < N * nn; id += blockDim.x * gridDim.x){
        i = id / nn;
        j = neigh_idx[id];

        if (i == i_to_delete || j == j_to_delete || i == j_to_delete || j == i_to_delete){
            event_type[id] = NULL_EVENT;
            event_prob[id] = 0.0;
        }
    }

}

__global__ void zero_out_events_split(EVENTTYPE *event_type, double *event_prob, const int *neigh_idx,
        const int size_i, const int start_i,
        int nn, int i_to_delete, int j_to_delete)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    int j;
    for (int id = idx; id < size_i * nn; id += blockDim.x * gridDim.x){
        i = id / nn + start_i;
        j = neigh_idx[id+start_i*nn];

        if ( j >=0 && (i == i_to_delete || j == j_to_delete || i == j_to_delete || j == i_to_delete)){
            event_type[id] = NULL_EVENT;
            event_prob[id] = 0.0;
        }
    }

}

double execute_kmc_step_gpu(const int N, const int nn, const int *neigh_idx, const int *site_layer,
                            const double *lattice, const int pbc, const double *T_bg, 
                            const double *freq, const double *sigma, const double *k,
                            const double *posx, const double *posy, const double *posz, 
                            const double *site_potential_charge, const double *site_temperature,
                            ELEMENT *site_element, int *site_charge, RandomNumberGenerator &rng, const int *neigh_idx_host){

    // **************************
    // **** Build Event List ****
    // **************************

    double time_event_list = 0.0;


    // the KMC event list arrays only exist in gpu memory
    EVENTTYPE *event_type; 
    double    *event_prob; 
    gpuErrchk( cudaMalloc((void**)&event_type, N * nn * sizeof(EVENTTYPE)) );
    gpuErrchk( cudaMalloc((void**)&event_prob, N * nn * sizeof(double)) );
                
    int num_threads = 512;
    int num_blocks = (N * nn - 1) / num_threads + 1;

    // populate the event_type and event_prob arrays:
    build_event_list_split<<<num_blocks, num_threads>>>(N, N, 0, nn, neigh_idx, 
                                                  site_layer, lattice, pbc,
                                                  T_bg, freq, sigma, k,
                                                  posx, posy, posz, 
                                                  site_potential_charge, site_temperature, 
                                                  site_element, site_charge, event_type, event_prob);

    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );
    // **************************
    // ** Event Execution Loop **
    // **************************

    // helper variables:
    // NOTE: INITIALIZE THESE ON GPU AND USE MEMCPY DEVICETODEVICE INSTEAD
    int two_host = 2;
    int two_neg_host = -2;
    int zero_host = 0;
    ELEMENT defect_element_host = DEFECT;
    ELEMENT O_defect_element_host = OXYGEN_DEFECT;
    ELEMENT vacancy_element_host = VACANCY;
    ELEMENT O_element_host = O_EL;

    int *two_d;
    gpuErrchk( cudaMalloc((void**)&two_d, 1 * sizeof(int)) );
    gpuErrchk( cudaMemcpy(two_d, &two_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );
    int *two_neg_d;
    gpuErrchk( cudaMalloc((void**)&two_neg_d, 1 * sizeof(int)) );
    gpuErrchk( cudaMemcpy(two_neg_d, &two_neg_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );
    int *zero_d;
    gpuErrchk( cudaMalloc((void**)&zero_d, 1 * sizeof(int)) );
    gpuErrchk( cudaMemcpy(zero_d, &zero_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );
    ELEMENT *defect_element_d;
    gpuErrchk( cudaMalloc((void**)&defect_element_d, 1 * sizeof(ELEMENT)) );
    gpuErrchk( cudaMemcpy(defect_element_d, &defect_element_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
    ELEMENT *O_defect_element_d;
    gpuErrchk( cudaMalloc((void**)&O_defect_element_d, 1 * sizeof(ELEMENT)) );
    gpuErrchk( cudaMemcpy(O_defect_element_d, &O_defect_element_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
    ELEMENT *vacancy_element_d;
    gpuErrchk( cudaMalloc((void**)&vacancy_element_d, 1 * sizeof(ELEMENT)) );
    gpuErrchk( cudaMemcpy(vacancy_element_d, &vacancy_element_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
    ELEMENT *O_element_d;
    gpuErrchk( cudaMalloc((void**)&O_element_d, 1 * sizeof(ELEMENT)) );
    gpuErrchk( cudaMemcpy(O_element_d, &O_element_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );


    double *event_prob_cum;
    gpuErrchk( cudaMalloc((void**)&event_prob_cum, N * nn * sizeof(double)) );
 
    double freq_host;
    gpuErrchk( cudaMemcpy(&freq_host, freq, 1 * sizeof(double), cudaMemcpyDeviceToHost) );


    double time_incl_sum = 0.0;
    double time_upper_bound = 0.0;
    double time_which_event = 0.0;
    double time_zero_prob = 0.0;

    double event_time = 0.0;
    int event_counter = 0;
    while (event_time < 1 / freq_host) {
        event_counter++;  
        // get the cumulative sum of the probabilities
        thrust::inclusive_scan(thrust::device, event_prob, event_prob + N * nn, event_prob_cum);

        // select an event
        double Psum_host;
        gpuErrchk( cudaMemcpy(&Psum_host, event_prob_cum + N * nn - 1, sizeof(double), cudaMemcpyDeviceToHost) );

        //TODO: cuda random number
        double number = rng.getRandomNumber() * Psum_host;
        int event_idx = thrust::upper_bound(thrust::device, event_prob_cum, event_prob_cum + N * nn, number) - event_prob_cum;
        // std::cout << "selected event: " << event_idx << "\n";

        EVENTTYPE sel_event_type = NULL_EVENT;
        gpuErrchk( cudaMemcpy(&sel_event_type, event_type + event_idx, sizeof(EVENTTYPE), cudaMemcpyDeviceToHost) );

        // test output:
        // double sel_event_prob;
        // gpuErrchk( cudaMemcpy(&sel_event_prob, event_prob + event_idx, sizeof(double), cudaMemcpyDeviceToHost) );
        // std::cout << "Selected event index: " << event_idx << " with type "
        //           << sel_event_type << " and probability " << sel_event_prob << std::endl;

        // get attributes of the sites involved:
        int i_host = static_cast<int>(floorf(event_idx / nn));
        int j_host;
        ELEMENT element_i_host, element_j_host;
        int charge_i_host, charge_j_host;

        gpuErrchk( cudaMemcpy(&j_host, neigh_idx + event_idx, sizeof(int), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(&element_i_host, site_element + i_host, sizeof(ELEMENT), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(&element_j_host, site_element + j_host, sizeof(ELEMENT), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(&charge_i_host, site_charge + i_host, sizeof(int), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(&charge_j_host, site_charge + j_host, sizeof(int), cudaMemcpyDeviceToHost) );

        // ELEMENT *element_i_d = site_element + i_host;
        // ELEMENT *element_j_d = site_element + j_host;
        // int *charge_i_d = site_charge + i_host;
        // int *charge_j_d = site_charge + j_host;



        // Event execution loop
        switch (sel_event_type)
        {
        case VACANCY_GENERATION:
        {
            gpuErrchk( cudaMemcpy(site_element + i_host, O_defect_element_d, 1 * sizeof(ELEMENT), cudaMemcpyDeviceToDevice) );
            gpuErrchk( cudaMemcpy(site_element + j_host, vacancy_element_d, 1 * sizeof(ELEMENT), cudaMemcpyDeviceToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + i_host, two_neg_d, 1 * sizeof(int), cudaMemcpyDeviceToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + j_host, two_d, 1 * sizeof(int), cudaMemcpyDeviceToDevice) );



            break;
        }
        case VACANCY_RECOMBINATION:
        {
            gpuErrchk( cudaMemcpy(site_element + i_host, defect_element_d, 1 * sizeof(ELEMENT), cudaMemcpyDeviceToDevice) );
            gpuErrchk( cudaMemcpy(site_element + j_host, O_element_d, 1 * sizeof(ELEMENT), cudaMemcpyDeviceToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + i_host, zero_d, 1 * sizeof(int), cudaMemcpyDeviceToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + j_host, zero_d, 1 * sizeof(int), cudaMemcpyDeviceToDevice) );       

            break;
        }
        case VACANCY_DIFFUSION:
        {

            // problem: TODO swap needed (naive overwrite one first)
            // make swap kernel
            gpuErrchk( cudaMemcpy(site_element + i_host, &element_j_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_element + j_host, &element_i_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + i_host, &charge_j_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + j_host, &charge_i_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );

            break;
        }
        case ION_DIFFUSION:
        {
            // gpuErrchk( cudaMemcpy(site_element + i_host, element_j_d, 1 * sizeof(ELEMENT), cudaMemcpyDeviceToDevice) );
            // gpuErrchk( cudaMemcpy(site_element + j_host, element_i_d, 1 * sizeof(ELEMENT), cudaMemcpyDeviceToDevice) );
            // gpuErrchk( cudaMemcpy(site_charge + i_host, charge_j_d, 1 * sizeof(int), cudaMemcpyDeviceToDevice) );
            // gpuErrchk( cudaMemcpy(site_charge + j_host, charge_i_d, 1 * sizeof(int), cudaMemcpyDeviceToDevice) );

            gpuErrchk( cudaMemcpy(site_element + i_host, &element_j_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_element + j_host, &element_i_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + i_host, &charge_j_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + j_host, &charge_i_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );


            break;
        }
        default:
            print("error: unidentified event key found: ");
            print(sel_event_type);
        }

        int threads = 1024;
        int blocks = (N * nn + threads - 1) / threads;
        zero_out_events<<<blocks, threads>>>(event_type, event_prob,
            neigh_idx, N, nn, i_host, j_host);
        event_time = -log(rng.getRandomNumber()) / Psum_host;
    }


    std::cout << "Number of KMC events: " << event_counter << "\n";
    std::cout << "Event time: " << event_time << "\n";

    gpuErrchk( cudaFree(event_prob_cum) );
    gpuErrchk( cudaFree(event_type) );
    gpuErrchk( cudaFree(event_prob) );

    return event_time;    
}

template <typename T>
__device__ void swap(
    T *a, T *b
){
    T temp = *a;
    *a = *b;
    *b = temp;
}

__global__ void execute_event(
    ELEMENT *site_element,
    int *site_charge,
    int *ijevent_to_delete
){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx == 0){
        int i_host = ijevent_to_delete[0];
        int j_host = ijevent_to_delete[1];
        EVENTTYPE sel_event_type = EVENTTYPE(ijevent_to_delete[2]);

        if(sel_event_type == VACANCY_GENERATION){
            site_element[i_host] = OXYGEN_DEFECT;
            site_element[j_host] = VACANCY;
            site_charge[i_host] = -2;
            site_charge[j_host] = 2;
        }
        //TODO generalizable
        else if(sel_event_type == VACANCY_RECOMBINATION){
            site_element[i_host] = DEFECT;
            site_element[j_host] = O_EL;
            site_charge[i_host] = 0;
            site_charge[j_host] = 0;

        }
        else if(sel_event_type == VACANCY_DIFFUSION){
            // swap
            swap<ELEMENT>(site_element + i_host, site_element + j_host);
            swap<int>(site_charge + i_host, site_charge + j_host);
        }
        else if(sel_event_type == ION_DIFFUSION){
            // swap
            swap<ELEMENT>(site_element + i_host, site_element + j_host);
            swap<int>(site_charge + i_host, site_charge + j_host);
        }

    }
} 


__global__ void read_out_event(
    int *ijevent_to_delete,
    const int i,
    const int *j,
    const EVENTTYPE *type
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx == 0){
        ijevent_to_delete[0] = i;
        ijevent_to_delete[1] = *j;
        ijevent_to_delete[2] = int(*type);
    }
}


double execute_kmc_step_mpi(
        MPI_Comm comm,
        const int N,
        const int *count,
        const int *displs,
        const int nn, const int *neigh_idx, const int *site_layer,
        const double *lattice, const int pbc, const double *T_bg, 
        const double *freq, const double *sigma, const double *k,
        const double *posx, const double *posy, const double *posz, 
        const double *site_potential_charge, const double *site_temperature,
        ELEMENT *site_element, int *site_charge, RandomNumberGenerator &rng, const int *neigh_idx_host)
{


    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // **************************
    // **** Build Event List ****
    // **************************

    double time_event_list = 0.0;

    // the KMC event list arrays only exist in gpu memory
    EVENTTYPE *event_type_local_d; 
    double    *event_prob_local_d; 
    gpuErrchk( cudaMalloc((void**)&event_type_local_d, count[rank] * nn * sizeof(EVENTTYPE)) );
    gpuErrchk( cudaMalloc((void**)&event_prob_local_d, count[rank] * nn * sizeof(double)) );
    double *event_prob_cum_local_d;
    gpuErrchk( cudaMalloc((void**)&event_prob_cum_local_d, count[rank] * nn * sizeof(double)) );
    double *event_prob_cum_global_h;
    gpuErrchk(cudaMallocHost((void**)&event_prob_cum_global_h, size * sizeof(double)));
       

    int num_threads = 1024;
    int num_blocks = (count[rank] * nn - 1) / num_threads + 1;

    // populate the event_type and event_prob arrays:
    // only your part of the event list
    // TODO use COO
    build_event_list_split<<<num_blocks, num_threads>>>(N,
                                                count[rank], displs[rank],
                                                nn, neigh_idx, 
                                                site_layer, lattice, pbc,
                                                T_bg, freq, sigma, k,
                                                posx, posy, posz, 
                                                site_potential_charge, site_temperature, 
                                                site_element, site_charge,
                                                event_type_local_d, event_prob_local_d);

    // **************************
    // ** Event Execution Loop **
    // **************************

    int threads_single_block = 64;

    int ijevent_to_delete[3];
    int *ijevent_to_delete_d;
    gpuErrchk( cudaMalloc((void**)&ijevent_to_delete_d, 3 * sizeof(int)) );

    double event_time = 0.0;
    int event_counter = 0;

    double *Psum_host;
    gpuErrchk(cudaMallocHost((void**)&Psum_host, 1 * sizeof(double)));


    double freq_h;
    gpuErrchk( cudaMemcpy(&freq_h, freq, 1 * sizeof(double), cudaMemcpyDeviceToHost) );
    while (event_time < 1 / freq_h) {
        event_counter++;  

        // get the cumulative sum of the probabilities
        thrust::inclusive_scan(thrust::device, event_prob_local_d, event_prob_local_d + count[rank] * nn, event_prob_cum_local_d);

        
        // select an event

        gpuErrchk( cudaMemcpy(Psum_host, event_prob_cum_local_d + count[rank] * nn - 1, sizeof(double), cudaMemcpyDeviceToHost) );
        
        MPI_Allgather(Psum_host, 1, MPI_DOUBLE, event_prob_cum_global_h, 1, MPI_DOUBLE, comm);
        
        for (int i = 1; i < size; i++){
            event_prob_cum_global_h[i] += event_prob_cum_global_h[i-1];
        }

        //TODO: cuda random number
        double number = rng.getRandomNumber() * event_prob_cum_global_h[size-1];
        // figure out which rank has the number
        int source_rank;
        for (int i = 0; i < size; i++){
            if (number < event_prob_cum_global_h[i]){
                source_rank = i;
                break;
            }
        }

        if(rank == source_rank){
            // shift random number to the correct range
            if(rank > 0){
                number -= event_prob_cum_global_h[rank-1];
            }
        
            int event_idx = thrust::upper_bound(thrust::device, event_prob_cum_local_d, event_prob_cum_local_d + count[rank] * nn, number) - event_prob_cum_local_d;
            // std::cout << "selected event: " << event_idx << "\n";

            read_out_event<<<1,threads_single_block>>>(
                ijevent_to_delete_d,
                event_idx / nn + displs[rank],
                neigh_idx + displs[rank]*nn + event_idx,
                event_type_local_d + event_idx);
            
            gpuErrchk( cudaMemcpy(ijevent_to_delete,
                ijevent_to_delete_d, 3 * sizeof(int), cudaMemcpyDeviceToHost) );

            MPI_Bcast(ijevent_to_delete, 3, MPI_INT, source_rank, comm);
        }
        else{
            MPI_Bcast(ijevent_to_delete, 3, MPI_INT, source_rank, comm);
            // memcpy to device
            gpuErrchk( cudaMemcpy(ijevent_to_delete_d, ijevent_to_delete, 3 * sizeof(int), cudaMemcpyHostToDevice) );
        }


        // execute the event on the SoA

        execute_event<<<1, threads_single_block>>>(site_element, site_charge, ijevent_to_delete_d);
        
        // TODO ijevent_to_delete only on device
        int i_host = ijevent_to_delete[0];
        int j_host = ijevent_to_delete[1];

        int threads = 1024;
        int blocks = (count[rank] * nn + threads - 1) / threads;
        zero_out_events_split<<<blocks, threads>>>(event_type_local_d, event_prob_local_d, neigh_idx,
            count[rank], displs[rank],
            nn, i_host, j_host);
        event_time = -log(rng.getRandomNumber()) / event_prob_cum_global_h[size-1];
    }

    if(rank == 0){
        std::cout << "Number of KMC events: " << event_counter << "\n";
        std::cout << "Event time: " << event_time << "\n";
    }


    gpuErrchk( cudaFree(event_prob_cum_local_d) );
    gpuErrchk( cudaFree(event_type_local_d) );
    gpuErrchk( cudaFree(event_prob_local_d) );
    gpuErrchk(cudaFreeHost(event_prob_cum_global_h));
    return event_time;    
}

#ifndef COMPILE_WITH_TESTS
void copytoConstMemory(std::vector<double> E_gen, std::vector<double> E_rec, std::vector<double> E_Vdiff, std::vector<double> E_Odiff)
{   
    gpuErrchk( cudaMemcpyToSymbol(E_gen_const, E_gen.data(), E_gen.size() * sizeof(double)) );
    gpuErrchk( cudaMemcpyToSymbol(E_rec_const, E_rec.data(), E_rec.size() * sizeof(double)) );
    gpuErrchk( cudaMemcpyToSymbol(E_Vdiff_const, E_Vdiff.data(), E_Vdiff.size() * sizeof(double)) );
    gpuErrchk( cudaMemcpyToSymbol(E_Odiff_const, E_Odiff.data(), E_Odiff.size() * sizeof(double)) );
}
#endif
