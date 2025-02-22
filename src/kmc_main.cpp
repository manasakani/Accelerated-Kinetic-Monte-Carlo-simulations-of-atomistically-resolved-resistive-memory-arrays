/*
@author: [Anonymized for review]
@date: 2023-08
Copyright 2023 [Anonymized for review]. All rights reserved.
*/

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstddef>
#include <stdlib.h>
#include <chrono>
#include <map>
#include <iomanip>
#include <mpi.h>

#include "KMCProcess.h"
#include "utils.h"
#include "Device.h"
#include "gpu_buffers.h"
#include "input_parser.h"
#include "KMC_comm.h"

#include "rocm_smi/rocm_smi.h"
#include "gpu_solvers.h"


std::string getHipErrorString(hipError_t error) {
    switch (error) {
        case hipSuccess:
            return "hipSuccess";
        case hipErrorInvalidValue:
            return "hipErrorInvalidValue";
        case hipErrorOutOfMemory:
            return "hipErrorOutOfMemory";
        // Add more cases as needed
        default:
            return "Unknown HIP error";
    }
}

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}


int main(int argc, char **argv)
{

    //***********************************
    // Initialize MPI
    //***********************************
    MPI_Init(&argc, &argv);

    int size_global, rank_global;
    MPI_Comm_size(MPI_COMM_WORLD, &size_global);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_global);

    //***********************************
    // Setup accelerators (GPU)
    //***********************************

    char* slurm_localid = getenv("SLURM_LOCALID");
    int localid = -1;
    if (slurm_localid != nullptr) {
        localid = atoi(slurm_localid);
        std::cout << "Rank " << rank_global << " has SLURM_LOCALID " << localid << std::endl;
    } else {
        std::cerr << "Rank " << rank_global << " cannot access SLURM_LOCALID" << std::endl;
        exit(1);
    }

    char* rocr_visible_devices = getenv("ROCR_VISIBLE_DEVICES");
    if (rocr_visible_devices != nullptr) {
        std::cout << "Rank " << rank_global << " ROCR_VISIBLE_DEVICES: " << rocr_visible_devices << std::endl;
    } else {
        std::cerr << "Rank " << rank_global << " ROCR_VISIBLE_DEVICES not set" << std::endl; exit(1);
    }

    hipError_t hipStatus;
    int device_id = 0; 
    hipStatus = hipSetDevice(device_id);

    hipDeviceProp_t dprop;
    hipGetDeviceProperties(&dprop, device_id);
    if (hipStatus == hipSuccess)
    {
        std::cout << "Rank " << rank_global << " successfully set device " << device_id << std::endl;
    } else {
        std::cerr << "Rank " << rank_global << " failed to set device " << device_id << std::endl;
        std::cerr << "Error: " << getHipErrorString(hipStatus) << std::endl;
    }

#pragma omp parallel
{
    #pragma omp single
    {
        int num_threads = omp_get_num_threads();
        if (!rank_global)
        std::cout << "Number of OMP threads: " << num_threads << std::endl;
    }
}

    //***************************************
    // Parse inputs and setup output logging
    //***************************************
    
    KMCParameters p(argv[1]);                                                       // stores simulation parameters
    std::ostringstream outputBuffer;                                                // holds output data to dump into a txt file
    std::string output_filename = "output" + std::to_string(size_global) + "_" + std::to_string(rank_global) + ".txt";
    std::remove(output_filename.c_str());
    std::ofstream outputFile(output_filename, std::ios_base::app);

    //*******************************************************
    // Initialize the Atomistic simulation domain (the Device)
    //*******************************************************

    std::vector<std::string> xyz_files;
    if (p.restart)
    {
        const bool file_exists = location_exists(p.restart_xyz_file);
        if (!file_exists)
        {
            outputBuffer << "ERROR: Restart file " << p.restart_xyz_file << " not found!\n";
            outputFile << outputBuffer.str();
            outputBuffer.str(std::string());
            return 0;
        }
        else
        {
            outputBuffer << "Restarting from " << p.restart_xyz_file << "\n";
            xyz_files.push_back(p.restart_xyz_file);
        }
    }
    else
    {   
        xyz_files.push_back(p.atom_xyz_file);
        xyz_files.push_back(p.interstitial_xyz_file);
    }

    if (!rank_global)
    std::cout << "Constructing device...\n"; 
    Device device(xyz_files, p);                                                    // contains the simulation domain and field solver functions
    
    if (p.pristine)                                                                // convert an initial percentage of oxygen atoms to vacancies
        device.makeSubstoichiometric(p.initial_vacancy_concentration);

    //******************************
    // Initialize the KMC Comm
    //******************************

    bool split = false;
    int ratio[2] = {8, 24};
    // std::string name = "hybrid_split";

    KMC_comm kmc_comm(MPI_COMM_WORLD,
        device.N - 2*p.num_atoms_first_layer,
        device.N_atom + 1,
        device.N,
        device.N,
        split,
        ratio);

    //******************************
    // Initialize the KMC Simulation
    //******************************
    std::cout << "Rank: " << kmc_comm.rank_events << ", Constructing KMC process" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    KMCProcess sim(device, p.freq);                                                // stores the division of the device into KMC 'layers' with different EA

    //*****************************
    // Setup GPU memory management
    //*****************************

    std::cout << "Rank: " << kmc_comm.rank_events << ", Constructing GPU buffers" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    GPUBuffers gpubuf(sim.layers, sim.site_layer, sim.freq,                         
                      device.N, device.N_atom, device.site_element, device.site_x, device.site_y, device.site_z,
                      device.max_num_neighbors, device.sigma, device.k, 
                      device.lattice, p.metals, p.metals.size(),
                      MPI_COMM_WORLD, p.num_atoms_first_layer);
    
    //******************************
    // Setup Neighbor Lists
    //******************************

    if (kmc_comm.comm_events != MPI_COMM_NULL) {
        std::cout << "Rank: " << kmc_comm.rank_events << ", Initialized neighbor lists" << std::endl; fflush(stdout);
        compute_neighbor_list(kmc_comm.comm_events, kmc_comm.counts_events, kmc_comm.displs_events, device, gpubuf, p);
    }
    if (p.solve_potential)
    {
        if (kmc_comm.comm_pairwise != MPI_COMM_NULL) {
            std::cout << "Rank: " << kmc_comm.rank_pairwise << ", Initialized cutoff lists" << std::endl; fflush(stdout);
            compute_cutoff_list(kmc_comm.comm_pairwise, kmc_comm.counts_pairwise, kmc_comm.displs_pairwise, device, gpubuf, p);
        }
    }
    
    gpubuf.sync_HostToGPU(device);                                                                  // initialize the device attributes in gpu memory

    if (p.solve_potential)
    {
        if (kmc_comm.comm_K != MPI_COMM_NULL) {
            std::cout << "Rank: " << kmc_comm.rank_K << ", Initialized sparsity K" << std::endl;                                                             // [1] fraction of power dissipated as heat
            initialize_sparsity_K(
                gpubuf, p.pbc, p.nn_dist, p.num_atoms_first_layer, kmc_comm);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    if (p.solve_current && kmc_comm.comm_T != MPI_COMM_NULL)   
    { 
        std::cout << "Rank: " << kmc_comm.rank_T << ", Initialized sparsity T" << std::endl;                                                             // [1] fraction of power dissipated as heat
        initialize_sparsity_CB(gpubuf, p.pbc, p.nn_dist, p.num_atoms_first_layer);

        std::cout << "Initialized sparsity pattern of K\n";
    }

    // make layer arrays and copy them to const memory
    std::vector<double> E_gen_host, E_rec_host, E_Vdiff_host, E_Odiff_host;
    for (auto l : sim.layers){
        E_gen_host.push_back(l.E_gen_0);
        E_rec_host.push_back(l.E_rec_1);
        E_Vdiff_host.push_back(l.E_diff_2); 
        E_Odiff_host.push_back(l.E_diff_3);
    }
    int num_layers = sim.layers.size();
    copytoConstMemory(E_gen_host, E_rec_host, E_Vdiff_host, E_Odiff_host); 

    // Create hip library handles to pass into the gpu_Device functions
    hipblasHandle_t handle;
    hipblasCreate(&handle);
    hipsolverHandle_t handle_cusolver;
    hipsolverCreate(&handle_cusolver);                           

    //***********************************
    // loop over V_switch and t_switch
    double Vd, t, kmc_time, I_macro, T_kmc, V_vcm;                                                  // KMC loop variables
    int kmc_step_count;                                                                             // tracks the number of KMC steps per bias point
    // std::map<std::string, double> resultMap;                                                     // dictionary of output quantities which are dumped to output.log
    std::chrono::duration<double> diff, diff_pot, diff_power, diff_temp, diff_perturb;              // track computation time of the different modules

    // std::cout << "Rank: " << kmc_comm.rank_events << ", Starting simulation" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    auto tcode_start = std::chrono::steady_clock::now();
    for (int vt_counter = 0; vt_counter < p.V_switch.size(); vt_counter++)
    {

        Vd = p.V_switch[vt_counter];                                                                // [V] applied voltage at this bias point
        t = p.t_switch[vt_counter];                                                                 // [s] physical duration of the applied voltage
        V_vcm = Vd;                                                                                 // [V] voltage dropped across rram (different only if there is a series resistor)
        I_macro = 0.0;                                                                              // [uA] net current leaving the device
        
        outputBuffer << "--------------------------------\n";
        outputBuffer << "Applied Voltage = " << Vd << " V\n";
        outputBuffer << "--------------------------------\n";

        // solve the Laplace Equation to get the CB edge energy at this voltage
        if (p.solve_current && kmc_comm.comm_T != MPI_COMM_NULL)
        {
            device.setLaplacePotential(handle, handle_cusolver, gpubuf, p, Vd);                     // homogenous poisson equation with contact BC
            initialize_sparsity_T(gpubuf, p.pbc, p.nn_dist, p.num_atoms_first_layer, p.num_atoms_first_layer, p.num_layers_contact, kmc_comm);
            std::cout << "Initialized sparsity of T\n";
        }

        // setup output folder
        const std::string folder_name = "Results_" + std::to_string(Vd);
        if(rank_global == 0){
            const bool folder_already_exists = location_exists(folder_name);
            if (folder_already_exists)
            {
                std::remove(folder_name.c_str());
            }
            const int error = mkdir(folder_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            outputBuffer << "Created folder: " << folder_name << '\n';
            std::string file_name = "snapshot_init.xyz";
            device.writeSnapshot(file_name, folder_name);            
        }

        //***********************************
        // constants from parameterization:
        
        double loop_G = p.high_G*10000000;                                                      // 'conductance' of the driver term for the NESS (converge to high value)
        double high_G = p.high_G*100000;                                                        // 'conductance' between metallic connections
        double low_G = p.low_G;                                                                 // 'conductance' between non-metallic connections
        double scale = 1e-5;                                                                    // scaled to avoid numerical issues
        double G0 = 2 * 3.8612e-5 * scale;                                                      // G0 = (q^2 / h_bar), G = G0 * Tij
        double tol = p.q * 0.01;                                                                // [eV] tolerance after which the barrier slope is considered
        int num_source_inj = p.num_atoms_first_layer;                                           // number of injection points (tied to injection node)
        int num_ground_ext = p.num_atoms_first_layer;                                           // number of extraction points (tied to extraction node)
        double alpha = 1;                                                                       // [1] fraction of power dissipated as heat
            
        kmc_time = 0.0;
        kmc_step_count = 0;
        
        // ********************************************************
        // ***************** MAIN KMC LOOP ************************
        // **** Update fields and execute events on structure *****
        // ********************************************************

        gpubuf.sync_HostToGPU(device);                                                         // initialize the device attributes in gpu memory
        gpuErrchk( hipDeviceSynchronize() ); //debug
        
        // timing/benchmarking setups
        double t_charge_update_start, t_charge_update_end, 
               t_boundary_start, t_boundary_end, 
               t_charge_start, t_charge_end, 
               t_current_start, t_current_end, 
               t_events_start, t_events_end,
               t_superstep_start, t_superstep_end;

        double time_loop;
        hipDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        auto time_start = std::chrono::high_resolution_clock::now();

        while (kmc_time < t)
        {
            MPI_Barrier(kmc_comm.comm_events);
            t_superstep_start = MPI_Wtime();

            // Update site-resolved charges
            if (p.solve_potential)
            {
                if (kmc_comm.comm_events != MPI_COMM_NULL) {

                    MPI_Barrier(kmc_comm.comm_events);

                    t_charge_update_start = MPI_Wtime();

                    update_charge_gpu(gpubuf.site_element,
                        gpubuf.site_charge,
                        gpubuf.neigh_idx,
                        gpubuf.N_, gpubuf.nn_, gpubuf.metal_types, gpubuf.num_metal_types_,
                        kmc_comm.counts_events, kmc_comm.displs_events, kmc_comm.comm_events);

                    t_charge_update_end = MPI_Wtime();
                }
            }

            // Update potential
            if (p.solve_potential)
            {

                // Update site-resolved potential from boundary
                
                if (kmc_comm.comm_K != MPI_COMM_NULL) {
                    MPI_Barrier(kmc_comm.comm_K);

                    t_boundary_start = MPI_Wtime();
                    // auto time_start = std::chrono::high_resolution_clock::now();

                    background_potential_gpu_sparse(handle, handle_cusolver, gpubuf, device.N, p.num_atoms_first_layer, p.num_atoms_first_layer,
                                            Vd, p.pbc, p.high_G, p.low_G, device.nn_dist, p.metals.size(), kmc_step_count);
                    
                    if(kmc_comm.rank_K == 0){
                        MPI_Gatherv(MPI_IN_PLACE, NULL, NULL,
                            gpubuf.site_potential_boundary + p.num_atoms_first_layer,
                            kmc_comm.counts_K,
                            kmc_comm.displs_K,
                            MPI_DOUBLE,
                            0, kmc_comm.comm_K);
                    }
                    else{
                        MPI_Gatherv(gpubuf.site_potential_boundary + p.num_atoms_first_layer + kmc_comm.displs_K[kmc_comm.rank_K],
                            kmc_comm.counts_K[kmc_comm.rank_K],
                            MPI_DOUBLE,
                            NULL,
                            kmc_comm.counts_K,
                            kmc_comm.displs_K,
                            MPI_DOUBLE,
                            0, kmc_comm.comm_K);                        
                    }
                    t_boundary_end = MPI_Wtime();

                    hipDeviceSynchronize();
                    MPI_Barrier(kmc_comm.comm_K);
                    // auto time_end = std::chrono::high_resolution_clock::now();
                    // time_K += std::chrono::duration<double>(time_end - time_start).count(); 

                }
                
                if (kmc_comm.comm_pairwise != MPI_COMM_NULL) {
                    // int measurements = 1;
                    // double time_pairwise[measurements];
                    // for(int j = 0; j < measurements; j++){

                        // Update site-resolved potential from charges
                        hipDeviceSynchronize();
                        MPI_Barrier(kmc_comm.comm_pairwise);
                        t_charge_start = MPI_Wtime();

                        // auto time_start = std::chrono::high_resolution_clock::now();
                        poisson_gridless_gpu(p.num_atoms_contact, p.pbc, gpubuf.N_, gpubuf.lattice, gpubuf.sigma, gpubuf.k,
                                gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
                                gpubuf.site_charge, gpubuf.site_potential_charge,
                                kmc_comm.rank_pairwise, kmc_comm.size_pairwise, kmc_comm.counts_pairwise, kmc_comm.displs_pairwise, 
                                gpubuf.cutoff_window, gpubuf.cutoff_idx, gpubuf.N_cutoff_); 
                        hipDeviceSynchronize();
                        if(kmc_comm.rank_pairwise == 0){
                            MPI_Gatherv(MPI_IN_PLACE, NULL, NULL,
                                gpubuf.site_potential_charge + kmc_comm.displs_pairwise[kmc_comm.rank_pairwise],
                                kmc_comm.counts_pairwise,
                                kmc_comm.displs_pairwise,
                                MPI_DOUBLE,
                                0, kmc_comm.comm_pairwise);
                        }
                        else{
                            MPI_Gatherv(gpubuf.site_potential_charge + kmc_comm.displs_pairwise[kmc_comm.rank_pairwise],
                                kmc_comm.counts_pairwise[kmc_comm.rank_pairwise],
                                MPI_DOUBLE, NULL,
                                kmc_comm.counts_pairwise,
                                kmc_comm.displs_pairwise,
                                MPI_DOUBLE,
                                0, kmc_comm.comm_pairwise);
                        }
                        t_charge_end = MPI_Wtime();
                        hipDeviceSynchronize();
                        MPI_Barrier(kmc_comm.comm_pairwise);

                        // auto time_end = std::chrono::high_resolution_clock::now();
                        // time_pairwise_tot += std::chrono::duration<double>(time_end - time_start).count();
                        // time_pairwise[j] = std::chrono::duration<double>(time_end - time_start).count();
                        // if(kmc_comm.rank_pairwise == 0){
                        //     std::cout << "Rank: " << kmc_comm.rank_pairwise << " " << time_pairwise[j] << std::endl;
                        // }
                    // }

                    // if(kmc_comm.rank_pairwise == 0){
                    //     std::string base_path = "final_reordered/";
                    //     std::string file_name = base_path + "time_pairwise_" + std::to_string(kmc_comm.size_pairwise) + ".txt";
                    //     std::ofstream time_file;
                    //     time_file.open(file_name);
                    //     for(int i = 0; i < measurements; i++){
                    //         time_file << time_pairwise[i] << std::endl;
                    //     }
                    //     time_file.close();                            
                    // }


                    outputBuffer << "Z - calculation time - charge [s]" <<  t_charge_update_end - t_charge_update_start << "\n";
                    outputBuffer << "Z - calculation time - potential from boundaries [s]" <<  t_boundary_end - t_boundary_start << "\n";
                    outputBuffer << "Z - calculation time - potential from charges [s]" << t_charge_end - t_charge_start << "\n";
                }


            }

            // Update current and joule heating
            if (p.solve_current)
            {   
                if (kmc_comm.comm_T != MPI_COMM_NULL) {

                    MPI_Barrier(kmc_comm.comm_T);
                    t_current_start = MPI_Wtime();
                    update_power_gpu_sparse_dist(handle, handle_cusolver, gpubuf, num_source_inj, num_ground_ext, p.num_layers_contact,
                                            Vd, high_G, low_G, loop_G, G0, tol,
                                            device.nn_dist, p.m_e, p.V0, p.metals.size(), &device.imacro,
                                            p.solve_heating_local, p.solve_heating_global, alpha);
                    t_current_end = MPI_Wtime();
                    outputBuffer << "Z - calculation time - potential from charges [s]" << t_current_end - t_current_start << "\n";
                }
            }

            if (p.solve_potential)
            {
                // sum potential terms into charge potential buffer
                sum_and_gather_potential(gpubuf, p.num_atoms_first_layer, kmc_comm);
            }


            // Execute events and update kmc_time
            if (p.perturb_structure){     
                if (kmc_comm.comm_events != MPI_COMM_NULL) {
                    std::cout << "Rank: " << kmc_comm.rank_events << ", Start events" << std::endl;   

                    MPI_Barrier(kmc_comm.comm_events);
                    t_events_start = MPI_Wtime();
                    // auto time_start = std::chrono::high_resolution_clock::now();
                    double event_time = execute_kmc_step_mpi(kmc_comm.comm_events,
                                                device.N, kmc_comm.counts_events,kmc_comm.displs_events,
                                                device.max_num_neighbors, gpubuf.neigh_idx, gpubuf.site_layer,
                                                gpubuf.lattice, device.pbc, gpubuf.T_bg, 
                                                gpubuf.freq, gpubuf.sigma, gpubuf.k,
                                                gpubuf.site_x, gpubuf.site_y, gpubuf.site_z, 
                                                gpubuf.site_potential_charge, gpubuf.site_temperature,
                                                gpubuf.site_element, gpubuf.site_charge, sim.random_generator);  
                    kmc_time += event_time; 
                    t_events_end = MPI_Wtime();
                    MPI_Barrier(kmc_comm.comm_events);
                    // auto time_end = std::chrono::high_resolution_clock::now();
                    // time_events += std::chrono::duration<double>(time_end - time_start).count();
                    outputBuffer << "Z - calculation time - kmc events [s]" <<  t_events_end - t_events_start << "\n";
                }

            } else {           
                if (kmc_step_count > 0)                                                           // run in IV or field-solver testing
                {
                    kmc_time = t;
                }
            }

            t_superstep_end = MPI_Wtime();
            
            // ********************************************************
            // ******************** Log results ***********************
            // ********************************************************

            outputBuffer << "KMC time is: " << kmc_time << "\n";
            
            // dump print buffer into the output file
            if (!(kmc_step_count % p.output_freq))
            {
                outputFile << outputBuffer.str();
                outputBuffer.str(std::string());
            }
            kmc_step_count++;

            outputBuffer << "Z - calculation time - KMC superstep [s]: " << t_superstep_end - t_superstep_start  << "\n";
            outputBuffer << "--------------------------------------\n";

            // DEBUG
            // if (kmc_step_count > 1000)
            // {
            //     std::cout << "kmc step count limit for debug\n";
            //     break;
            // }
            
        } // while (kmc_time < t)
        hipDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        auto time_end = std::chrono::high_resolution_clock::now();
        time_loop = std::chrono::duration<double>(time_end - time_start).count();

        // if(rank_global == 0){
        //     // append to file
        //     std::cout << "Time loop " << time_loop << std::endl;
        //     std::string file_name = "final_reordered/time_full_application" + name +"_" + std::to_string(size_global) + "_" + std::to_string(Vd) + ".txt";
        //     std::ofstream time_file;
        //     time_file.open(file_name, std::ios_base::app);
        //     time_file << time_loop << std::endl;
        //     time_file.close();
        // }

// Get device attributes from GPU memory
#ifdef USE_CUDA
        gpubuf.sync_GPUToHost(device);
#endif
        if (!rank_global)
        {
            const std::string file_name = "snapshot_" + std::to_string(kmc_step_count) + ".xyz";
            std::cout << "KMC step count: " << kmc_step_count << "\n";
            device.writeSnapshot(file_name, folder_name);
        }

        // // TODO: remove the reset inside each bias point
        // // Re-fix the boundary (for changes in applied potential across an IV sweep)
        // thrust::device_ptr<double> left_boundary = thrust::device_pointer_cast(gpubuf.site_potential_boundary);
        // thrust::fill(left_boundary, left_boundary + N_left_tot, -Vd/2);
        // thrust::device_ptr<double> right_boundary = thrust::device_pointer_cast(gpubuf.site_potential_boundary + N_left_tot + N_interface);
        // thrust::fill(right_boundary, right_boundary + N_right_tot, Vd/2);
        // std::cout << "updated boundary inside K - remove this later\n";

    } // for (int vt_counter = 0; vt_counter < p.V_switch.size(); vt_counter++)

    auto tcode_ends = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_code = tcode_ends - tcode_start;
    if (!rank_global)
    {
        std::cout << "Total code execution time: " << diff_code.count() << " s\n";
    }

    //****************************************************************************************
    // Finalize MPI and Cleanup
    // Note: the destructors need to be called in the correct order, this is curently omitted
    //****************************************************************************************    

    // gpubuf.freeGPUmemory();
    CheckCublasError(hipblasDestroy(handle));

    // close logger
    outputFile << outputBuffer.str();
    outputFile.close();

    std::cout << "Rank: " << kmc_comm.rank_events << ", Finished simulation" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // kmc_comm.~KMC_comm();
    
    MPI_Finalize();
    return 0;
}
