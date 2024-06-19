/*
@author: Manasa Kaniselvan 
@date: 2023-08
Copyright 2023 ETH Zurich and the Computational Nanoelectronics Group. All rights reserved.
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

#ifdef USE_CUDA
#include "gpu_solvers.h"
#endif


int main(int argc, char **argv)
{

    //***********************************
    // Initialize MPI
    //***********************************
    MPI_Init(&argc, &argv);

    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    //***********************************
    // Setup accelerators (GPU)
    //***********************************


#ifdef USE_CUDA
    char gpu_string[1000];
    get_gpu_info(gpu_string, 0);
    if (!mpi_rank)
    printf("Will use this GPU: %s\n", gpu_string);
    set_gpu(0);
#else
    if (!mpi_rank)
    std::cout << "Simulation will not use any accelerators.\n";
#endif

#pragma omp parallel
{
    #pragma omp single
    {
        int num_threads = omp_get_num_threads();
        if (!mpi_rank)
        std::cout << "Number of OMP threads: " << num_threads << std::endl;
    }
}

    //***************************************
    // Parse inputs and setup output logging
    //***************************************
    
    KMCParameters p(argv[1]);                                                       // stores simulation parameters
    std::ostringstream outputBuffer;                                                // holds output data to dump into a txt file
    std::string output_filename = "output" + std::to_string(mpi_size) + "_" + std::to_string(mpi_rank) + ".txt";
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

    if (!mpi_rank)
    std::cout << "Constructing device...\n"; 
    Device device(xyz_files, p);                                                                    // contains the simulation domain and field solver functions

    if (p.pristine)                                                                                 // convert an initial percentage of oxygen atoms to vacancies
        device.makeSubstoichiometric(p.initial_vacancy_concentration);

    //******************************
    // Initialize the KMC Simulation
    //******************************

    KMCProcess sim(device, p.freq, p.reservoir_layer_start);                                        // stores the division of the device into KMC 'layers' with different EA
    std::cout << "KMC process initialized\n";

    //*****************************
    // Setup GPU memory management
    //*****************************

#ifdef USE_CUDA
    GPUBuffers gpubuf(sim.layers, sim.site_layer, sim.freq,                         
                      device.N, device.N_atom, device.site_x, device.site_y, device.site_z,
                      device.max_num_neighbors, device.sigma, device.k, 
                      device.lattice, device.neigh_idx, device.cutoff_window, device.cutoff_idx, p.metals, p.metals.size(),
                      MPI_COMM_WORLD, p.num_atoms_first_layer);
    gpubuf.sync_HostToGPU(device);                                                                  // initialize the device attributes in gpu memory
    initialize_sparsity_K(gpubuf, p.pbc, p.nn_dist, p.num_atoms_first_layer);
#else
    GPUBuffers gpubuf;
#endif
    std::cout << "GPU buffers initialized\n";

    // Create CUDA library handles to pass into the gpu_Device functions
    cublasHandle_t handle = CreateCublasHandle(0);
    cusolverDnHandle_t handle_cusolver = CreateCusolverDnHandle(0);                                   

    //***********************************
    std::chrono::duration<double> diff_laplacian, diff, diff_pot, diff_power, diff_temp, diff_perturb;   // track computation time of the different modules

    if (p.solve_heating_local)                                                                      // build the Laplacian to solve for the local temperature distribution
    {
        auto t_lap0 = std::chrono::steady_clock::now();
        device.constructLaplacian(p, handle, handle_cusolver, gpubuf);
        auto t_lap1 = std::chrono::steady_clock::now();
        diff_laplacian = t_lap1 - t_lap0;
        outputBuffer << "**Calculation time for the laplacian:**\n";
        outputBuffer << "Laplacian update: " << diff_laplacian.count() << "\n";
        outputBuffer.str(std::string());
    }

    // loop over V_switch and t_switch
    double Vd, t, kmc_time, step_time, I_macro, T_kmc, V_vcm;                                       // KMC loop variables
    int kmc_step_count;                                                                             // tracks the number of KMC steps per bias point
    std::string file_name;                                                                          // name of the xyz snapshot file
    std::map<std::string, double> resultMap;                                                        // dictionary of output quantities which are dumped to output.log
    
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
        if (p.solve_current)
        {
            device.setLaplacePotential(handle, handle_cusolver, gpubuf, p, Vd);                     // homogenous poisson equation with contact BC
        }

        // setup output folder
        const std::string folder_name = "Results_" + std::to_string(Vd);
        make_folder(folder_name);
        outputBuffer << "Created folder: " << folder_name << '\n';

        kmc_time = 0.0;
        kmc_step_count = 0;
        step_time = 1/p.freq;                                                                       // the initial step time is the inverse of the global attempt frequency

        // ********************************************************
        // ***************** MAIN KMC LOOP ************************
        // **** Update fields and execute events on structure *****
        // ********************************************************

#ifdef USE_CUDA
        gpubuf.sync_HostToGPU(device);                                                              // initialize the device attributes in gpu memory
#endif
        while (kmc_time < t)
        {
            outputBuffer << "--------------\n";
            outputBuffer << "KMC step count: " << kmc_step_count << "\n";
            auto t0 = std::chrono::steady_clock::now();

            // handle any input IR drop:
            V_vcm = Vd - I_macro * p.Rs;
            outputBuffer << "V_vcm: " << V_vcm << "\n";

            // 1. Update potential
            if (p.solve_potential)
            {
                // update site-resolved charge
                std::map<std::string, double> chargeMap = device.updateCharge(gpubuf, p.metals);           
                resultMap.insert(chargeMap.begin(), chargeMap.end());   
                
                 // update site-resolved potential
                std::map<std::string, double> potentialMap = device.updatePotential(handle, handle_cusolver, gpubuf, p, Vd, kmc_step_count);
                resultMap.insert(potentialMap.begin(), potentialMap.end());                                   
            }

            // 2. Update current 
            if (p.solve_current)
            {
                std::map<std::string, double> powerMap = device.updatePower(handle, handle_cusolver,    // update site-resolved dissipated power
                                                                            gpubuf, p, Vd);
                resultMap.insert(powerMap.begin(), powerMap.end());
                I_macro = device.imacro;

            }  

            // 3. Update Temperature
            if (p.solve_current && (p.solve_heating_global || p.solve_heating_local))                                     // update site-resolved heat
            {
                std::map<std::string, double> temperatureMap = device.updateTemperature(handle, handle_cusolver, gpubuf, p, step_time);
                resultMap.insert(temperatureMap.begin(), temperatureMap.end());
            }

            // generate xyz snapshot
            if (!(kmc_step_count % p.log_freq))
            {
#ifdef USE_CUDA
        gpubuf.sync_GPUToHost(device);
#endif
                std::string file_name = "snapshot_" + std::to_string(kmc_step_count) + ".xyz";
                device.writeSnapshot(file_name, folder_name);
            }

            // Compliance check
            if (I_macro >= p.Icc*(1e-6)) // I_macro is in A, Icc is in uA
            {
                outputBuffer << "I_macro > Icc, compliance current reached.\n";
                I_macro = p.Icc;
                resultMap["Current [uA]"] = I_macro; 
                break;
            }

            // 4. Execute events and update kmc_time
            if (p.perturb_structure){                                  
                std::map<std::string, double> kmcMap = sim.executeKMCStep(gpubuf, device, &step_time);   
                if (kmc_time + step_time < t)
                {
                    kmc_time += step_time;
                }
                else
                {
                    kmc_time = t;
                }
                resultMap.insert(kmcMap.begin(), kmcMap.end());                                   
            } else {           
                step_time = p.delta_t;
                kmc_time += step_time;
            }

            auto tfield = std::chrono::steady_clock::now();
            std::chrono::duration<double> dt_field = tfield - t0;

            // ********************************************************
            // ******************** Log results ***********************
            // ********************************************************

            outputBuffer << "KMC time is: " << kmc_time << "\n";
            
            // load step results into print buffer
            for (const auto &pair : resultMap)
            {
                outputBuffer << pair.first << ": " << pair.second << std::endl;
            }
            resultMap.clear();

            // dump print buffer into the output file
            if (!(kmc_step_count % p.output_freq))
            {
                outputFile << outputBuffer.str();
                outputBuffer.str(std::string());
            }
            kmc_step_count++;

            // condition to skip over stuck bias points in a sweep (disable this)
            if(p.V_switch.size() > 1 && kmc_step_count > 50000)
            {
                kmc_time = t;
            }

            auto tlog = std::chrono::steady_clock::now();
            std::chrono::duration<double> dt_log = tlog - tfield;

            // Log timing info
            auto t1 = std::chrono::steady_clock::now();
            std::chrono::duration<double> dt = t1 - t0;
            outputBuffer << "Z - calculation time - all fields [s]: " << dt_field.count() << "\n";
            outputBuffer << "Z - calculation time - logging results [s]: " << dt_log.count() << "\n";
            outputBuffer << "Z - calculation time - KMC superstep [s]: " << dt.count() << "\n";
            outputBuffer << "--------------------------------------";

        } // while (kmc_time < t)
            
        // Last dump into output file
        for (const auto &pair : resultMap)
        {
            outputBuffer << pair.first << ": " << pair.second << std::endl;
        }
        resultMap.clear();
        if (!(kmc_step_count % p.output_freq))
        {
            outputFile << outputBuffer.str();
            outputBuffer.str(std::string());
        }

// Get device attributes from GPU memory
#ifdef USE_CUDA
        gpubuf.sync_GPUToHost(device);
#endif
        file_name = "snapshot_" + std::to_string(kmc_step_count) + ".xyz";
        device.writeSnapshot(file_name, folder_name);

    } // for (int vt_counter = 0; vt_counter < p.V_switch.size(); vt_counter++)

// #ifdef USE_CUDA
//     gpubuf.freeGPUmemory();
//     CheckCublasError(cublasDestroy(handle));
// #endif

    // close logger
    outputFile << outputBuffer.str();
    outputFile.close();

    //***********************************
    // Finalize MPI
    //***********************************

    MPI_Finalize();

    return 0;
}
