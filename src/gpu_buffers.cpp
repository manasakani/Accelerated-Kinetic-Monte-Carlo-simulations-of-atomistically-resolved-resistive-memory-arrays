#include "gpu_buffers.h"
#include "Device.h" 
#include "utils.h"
#include <cassert>

#ifdef USE_CUDA
#include <cuda.h>
#endif

void GPUBuffers::sync_HostToGPU(Device &device){

    assert(N_ > 0);
    assert(nn_ > 0);

    size_t dataSize = N_ * sizeof(ELEMENT);
    if (dataSize != device.site_element.size() * sizeof(ELEMENT)) {
        std::cout << "N_: " << N_ << "\n";
        std::cout << device.site_element.size() << "\n";
        std::cout << "nn: " << nn_ << "\n";
        fprintf(stderr, "ERROR: Size mismatch in GPU memory copy.\n");
        exit(EXIT_FAILURE);
    }

#ifdef USE_CUDA
    cudaDeviceSynchronize();
    gpuErrchk( cudaMemcpy(site_element, device.site_element.data(), N_ * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(site_charge, device.site_charge.data(), N_ * sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(site_power, device.site_power.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(site_CB_edge, device.site_CB_edge.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(site_potential_boundary, device.site_potential_boundary.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(site_potential_charge, device.site_potential_charge.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(site_temperature, device.site_temperature.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(atom_CB_edge, device.atom_CB_edge.data(), N_atom_ * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(T_bg, &device.T_bg, 1 * sizeof(double), cudaMemcpyHostToDevice) );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#endif
}

void GPUBuffers::sync_GPUToHost(Device &device){

#ifdef USE_CUDA
    cudaDeviceSynchronize();
    gpuErrchk( cudaMemcpy(device.site_element.data(), site_element, N_ * sizeof(ELEMENT), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_charge.data(), site_charge, N_ * sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_power.data(), site_power, N_ * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_CB_edge.data(), site_CB_edge, N_ * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_potential_boundary.data(), site_potential_boundary, N_ * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_potential_charge.data(), site_potential_charge, N_ * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_temperature.data(), site_temperature, N_ * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.atom_CB_edge.data(), atom_CB_edge, N_atom_ * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(&device.T_bg, T_bg, 1 * sizeof(double), cudaMemcpyDeviceToHost) );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError()); 
#endif
}

// copy back just the site_power into the power vector
void GPUBuffers::copy_power_fromGPU(std::vector<double> &power){
    power.resize(N_);
#ifdef USE_CUDA
    gpuErrchk( cudaMemcpy(power.data(), site_power, N_ * sizeof(double), cudaMemcpyDeviceToHost) );
#endif

}

// copy the background temperature TO the gpu buffer
void GPUBuffers::copy_Tbg_toGPU(double new_T_bg){
#ifdef USE_CUDA
    gpuErrchk( cudaMemcpy(T_bg, &new_T_bg, 1 * sizeof(double), cudaMemcpyHostToDevice) );
#endif

}

void GPUBuffers::copy_charge_toGPU(std::vector<int> &charge){
    charge.resize(N_);
#ifdef USE_CUDA
    gpuErrchk( cudaMemcpy(site_charge, charge.data(), N_ * sizeof(int), cudaMemcpyHostToDevice) );
#endif
}


void GPUBuffers::freeGPUmemory(){
#ifdef USE_CUDA
    cudaFree(site_element);
    cudaFree(site_x);
    cudaFree(site_y);
    cudaFree(site_z);
    cudaFree(neigh_idx);
    cudaFree(site_layer);
    cudaFree(site_charge);
    cudaFree(site_power);
    cudaFree(site_potential_boundary);
    cudaFree(site_potential_charge);
    cudaFree(site_temperature);
    cudaFree(T_bg);
    cudaFree(metal_types);
    cudaFree(sigma);
    cudaFree(k);
    cudaFree(lattice);
    cudaFree(freq);
    //... FREE THE REST OF THE MEMORY !!! ...
#endif

//destroy handles!
}
