#include "gpu_solvers.h"


// computes number of nonzeros per row of the specified subblock of matrix K
__global__ void calc_nnz_per_row_K(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double cutoff_radius,
    int block_size_i,
    int block_size_j,
    int block_start_i,
    int block_start_j,
    int *nnz_per_row_d
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO optimize this with a 2D grid instead of 1D
    for(int row = idx; row < block_size_i; row += blockDim.x * gridDim.x){
        int nnz_row = 0;
        for(int col = 0; col < block_size_j; col++){
            int i = block_start_i + row;
            int j = block_start_j + col;
            double dist = site_dist_gpu(posx_d[i], posy_d[i], posz_d[i],
                                        posx_d[j], posy_d[j], posz_d[j]);
            if(dist < cutoff_radius){
                nnz_row++;
            }
        }
        nnz_per_row_d[row] = nnz_row;
    }

}


__global__ void assemble_col_inds_K(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double cutoff_radius,
    int block_size_i,
    int block_size_j,
    int block_start_i,
    int block_start_j,
    int *row_ptr_d,
    int *col_indices_d)
{
    // row ptr is already calculated with the exclusive scan of nnz_per_row

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int row = idx; row < block_size_i; row += blockDim.x * gridDim.x){
        int nnz_row = 0;
        for(int col = 0; col < block_size_j; col++){
            int i = block_start_i + row;
            int j = block_start_j + col;
            double dist = site_dist_gpu(posx_d[i], posy_d[i], posz_d[i],
                                        posx_d[j], posy_d[j], posz_d[j]);
            if(dist < cutoff_radius){
                col_indices_d[row_ptr_d[row] + nnz_row] = col;
                nnz_row++;
            }
        }
    }
}



void create_boundary_sparsity(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    int block_size_i,
    int block_size_j,
    int block_start_i,
    int block_start_j,
    int **col_indices_d,
    int **row_ptr_d,
    int *nnz
)
{
    // parallelize over rows
    int threads = 512;
    int blocks = (block_size_i + threads - 1) / threads;

    int *nnz_per_row_d;
    gpuErrchk( cudaMalloc((void **)row_ptr_d, (block_size_i + 1) * sizeof(int)) );
    gpuErrchk( cudaMalloc((void **)&nnz_per_row_d, block_size_i * sizeof(int)) );
    gpuErrchk(cudaMemset((*row_ptr_d), 0, (block_size_i + 1) * sizeof(int)) );

    // calculate the nnz per row
    calc_nnz_per_row_K<<<blocks, threads>>>(posx_d, posy_d, posz_d, cutoff_radius,
        block_size_i, block_size_j, block_start_i, block_start_j, nnz_per_row_d);

    void     *temp_storage_d = NULL;
    size_t   temp_storage_bytes = 0;

    // determines temporary device storage requirements for inclusive prefix sum
    cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*row_ptr_d)+1, block_size_i);

    // Allocate temporary storage for inclusive prefix sum
    gpuErrchk(cudaMalloc(&temp_storage_d, temp_storage_bytes));

    // Run inclusive prefix sum
    // inclusive sum starting at second value to get the row ptr
    // which is the same as inclusive sum starting at first value and last value filled with nnz
    cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*row_ptr_d)+1, block_size_i);
    
    // nnz is the same as (*row_ptr_d)[block_size_i]
    gpuErrchk( cudaMemcpy(nnz, (*row_ptr_d) + block_size_i, sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMalloc((void **)col_indices_d, nnz[0] * sizeof(int)) );

    // assemble the indices of K
    assemble_col_inds_K<<<blocks, threads>>>(
        posx_d, posy_d, posz_d,
        cutoff_radius,
        block_size_i,
        block_size_j,
        block_start_i,
        block_start_j,
        (*row_ptr_d),
        (*col_indices_d)
    );

    cudaFree(temp_storage_d);
    cudaFree(nnz_per_row_d);
}


__global__ void assemble_K_indices_gpu(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    int matrix_size,
    int *nnz_per_row_d,
    int *row_ptr_d,
    int *col_indices_d)
{
    // row ptr is already calculated
    // exclusive scam of nnz_per_row

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    //TODO can be optimized with a 2D grid instead of 1D
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        int nnz_row = 0;
        for(int j = 0; j < matrix_size; j++){
        
            double dist = site_dist_gpu(posx_d[i], posy_d[i], posz_d[i],
                                        posx_d[j], posy_d[j], posz_d[j],
                                        lattice_d[0], lattice_d[1], lattice_d[2], pbc);
            if(dist < cutoff_radius){
                col_indices_d[row_ptr_d[i] + nnz_row] = j;
                nnz_row++;
            }
        }
    }
}

void create_device_submatrix_sparsity(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    const int matrix_size,
    int **col_indices_d,
    int **row_ptr_d,
    int *nnz
)
{
    // parallelize over rows
    int threads = 512;
    int blocks = (matrix_size + threads - 1) / threads;

    int *nnz_per_row_d;
    gpuErrchk( cudaMalloc((void **)row_ptr_d, (matrix_size + 1) * sizeof(int)) );
    gpuErrchk( cudaMalloc((void **)&nnz_per_row_d, matrix_size * sizeof(int)) );
    gpuErrchk(cudaMemset((*row_ptr_d), 0, (matrix_size + 1) * sizeof(int)) );

    // calculate the nnz per row - every rank does the full matrix!
    calc_nnz_per_row_K<<<blocks, threads>>>(posx_d, posy_d, posz_d, cutoff_radius, matrix_size, matrix_size, 0, 0, nnz_per_row_d);

    void     *temp_storage_d = NULL;
    size_t   temp_storage_bytes = 0;
    // determines temporary device storage requirements for inclusive prefix sum
    cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*row_ptr_d)+1, matrix_size);

    // Allocate temporary storage for inclusive prefix sum
    gpuErrchk(cudaMalloc(&temp_storage_d, temp_storage_bytes));
    // Run inclusive prefix sum
    // inclusive sum starting at second value to get the row ptr
    // which is the same as inclusive sum starting at first value and last value filled with nnz
    cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*row_ptr_d)+1, matrix_size);
    
    // nnz is the same as (*row_ptr_d)[matrix_size]
    gpuErrchk( cudaMemcpy(nnz, (*row_ptr_d) + matrix_size, sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMalloc((void **)col_indices_d, nnz[0] * sizeof(int)) );

    // assemble the indices of K
    assemble_K_indices_gpu<<<blocks, threads>>>(
        posx_d, posy_d, posz_d,
        lattice_d, pbc,
        cutoff_radius,
        matrix_size,
        nnz_per_row_d,
        (*row_ptr_d),
        (*col_indices_d)
    );

    cudaFree(temp_storage_d);
    cudaFree(nnz_per_row_d);
}

void indices_creation_gpu_off_diagonal_block(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    int block_size_i,
    int block_size_j,
    int block_start_i,
    int block_start_j,
    int **col_indices_d,
    int **row_ptr_d,
    int *nnz
)
{
    // parallelize over rows
    int threads = 512;
    int blocks = (block_size_i + threads - 1) / threads;

    int *nnz_per_row_d;
    gpuErrchk( cudaMalloc((void **)row_ptr_d, (block_size_i + 1) * sizeof(int)) );
    gpuErrchk( cudaMalloc((void **)&nnz_per_row_d, block_size_i * sizeof(int)) );
    gpuErrchk(cudaMemset((*row_ptr_d), 0, (block_size_i + 1) * sizeof(int)) );

    // calculate the nnz per row
    calc_nnz_per_row_K<<<blocks, threads>>>(posx_d, posy_d, posz_d, cutoff_radius,
        block_size_i, block_size_j, block_start_i, block_start_j, nnz_per_row_d);

    void     *temp_storage_d = NULL;
    size_t   temp_storage_bytes = 0;

    // determines temporary device storage requirements for inclusive prefix sum
    cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*row_ptr_d)+1, block_size_i);

    // Allocate temporary storage for inclusive prefix sum
    gpuErrchk(cudaMalloc(&temp_storage_d, temp_storage_bytes));

    // Run inclusive prefix sum
    // inclusive sum starting at second value to get the row ptr
    // which is the same as inclusive sum starting at first value and last value filled with nnz
    cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*row_ptr_d)+1, block_size_i);
    
    // nnz is the same as (*row_ptr_d)[block_size_i]
    gpuErrchk( cudaMemcpy(nnz, (*row_ptr_d) + block_size_i, sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMalloc((void **)col_indices_d, nnz[0] * sizeof(int)) );

    // assemble the indices of K
    assemble_col_inds_K<<<blocks, threads>>>(
        posx_d, posy_d, posz_d,
        cutoff_radius,
        block_size_i,
        block_size_j,
        block_start_i,
        block_start_j,
        (*row_ptr_d),
        (*col_indices_d)
    );

    cudaFree(temp_storage_d);
    cudaFree(nnz_per_row_d);
}


void Assemble_K_sparsity(const double *posx, const double *posy, const double *posz,
                         const double *lattice, const bool pbc,
                         const double cutoff_radius,
                         int system_size, int contact_left_size, int contact_right_size,
                         int **A_row_ptr, int **A_col_indices, int *A_nnz, 
                         int **contact_left_col_indices, int **contact_left_row_ptr, int *contact_left_nnz, 
                         int **contact_right_col_indices, int **contact_right_row_ptr, int *contact_right_nnz){

    // indices of A (the device submatrix)
    create_device_submatrix_sparsity(
        posx + contact_left_size,
        posy + contact_left_size,
        posz + contact_left_size,
        lattice, pbc,
        cutoff_radius,
        system_size,
        A_col_indices,
        A_row_ptr,
        A_nnz
    );

    // indices of the off-diagonal leftcontact-A matrix
    create_boundary_sparsity(
        posx, posy, posz,
        lattice, pbc,
        cutoff_radius,
        system_size,
        contact_left_size,
        contact_left_size,
        0,
        contact_left_col_indices,
        contact_left_row_ptr,
        contact_left_nnz
    );
    // std::cout << "contact_left_nnz " << *contact_left_nnz << std::endl;

    // indices of the off-diagonal A-rightcontact matrix
    create_boundary_sparsity(
        posx, posy, posz,
        lattice, pbc,
        cutoff_radius,
        system_size,
        contact_right_size,
        contact_left_size,
        contact_left_size + system_size,
        contact_right_col_indices,
        contact_right_row_ptr,
        contact_right_nnz
    );
    // std::cout << "contact_right_nnz " << *contact_right_nnz << std::endl;
}


void initialize_sparsity_K(GPUBuffers &gpubuf, int pbc, const double nn_dist, int num_atoms_contact)
{
    
    int rank = gpubuf.rank;
    int size = gpubuf.size;
    int rows_this_rank = gpubuf.count_K_device[rank];
    int disp_this_rank = gpubuf.displ_K_device[rank];

    int threads = 1024;
    int blocks = (rows_this_rank - 1) / threads + 1;

    int N_left_tot = num_atoms_contact;
    int N_right_tot = num_atoms_contact;
    int N_interface = gpubuf.N_ - (N_left_tot + N_right_tot);
    
    int *dist_nnz_h = new int[gpubuf.size];
    int *dist_nnz_d;
    int *dist_nnz_per_row_d;

    gpuErrchk( cudaMalloc((void **)&dist_nnz_d, gpubuf.size * sizeof(int)) );
    gpuErrchk(cudaMemset(dist_nnz_d, 0, gpubuf.size * sizeof(int)));
    gpuErrchk( cudaMalloc((void **)&dist_nnz_per_row_d, gpubuf.size * rows_this_rank * sizeof(int)) );
    gpuErrchk(cudaMemset(dist_nnz_per_row_d, 0, gpubuf.size * rows_this_rank * sizeof(int)));

    // Assemble the sparsity pattern

    // loop over the size to determine neighbours (the first neighbor is the self)
    for(int i = 0; i < size; i++){
        int rows_other = gpubuf.count_K_device[i];
        int displ_other = gpubuf.displ_K_device[i];

        calc_nnz_per_row_K<<<blocks, threads>>>(
            gpubuf.site_x + N_left_tot,
            gpubuf.site_y + N_left_tot,
            gpubuf.site_z + N_left_tot,
            nn_dist,
            rows_this_rank,
            rows_other,
            disp_this_rank,
            displ_other,
            dist_nnz_per_row_d + i * rows_this_rank
        );

        // reduce nnz per row
        void     *temp_storage_d = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(
        temp_storage_d, temp_storage_bytes, 
            dist_nnz_per_row_d + i * rows_this_rank,
            dist_nnz_d + i, rows_this_rank);

        // Allocate temporary storage
        cudaMalloc(&temp_storage_d, temp_storage_bytes);

        // Run sum-reduction
        cub::DeviceReduce::Sum(temp_storage_d, temp_storage_bytes,
            dist_nnz_per_row_d + i * rows_this_rank,
            dist_nnz_d + i, rows_this_rank);
    }


    gpuErrchk( cudaMemcpy(dist_nnz_h, dist_nnz_d, size * sizeof(int), cudaMemcpyDeviceToHost) );
    // counting neighbours
    int neighbor_count = 0;
    for(int i = 0; i < size; i++){
        if(dist_nnz_h[i] > 0){
            neighbor_count++;
        }
    }

    // get the indices of the neighbours
    int *neighbor_idx = new int[neighbor_count];
    int *neighbor_nnz_h = new int[neighbor_count];
    int *neighbor_nnz_per_row_d;
    gpuErrchk( cudaMalloc((void **)&neighbor_nnz_per_row_d, neighbor_count * rows_this_rank * sizeof(int)) );

    // determine neighbours
    neighbor_count = 0;
    for(int i = 0; i < size; i++){
        int neighbor = (i+rank) % size;
        if(dist_nnz_h[neighbor] > 0){
            neighbor_idx[neighbor_count] = neighbor;
            neighbor_count++;
        }
    }        
    // fill the neighbor nnz
    for(int i = 0; i < neighbor_count; i++){
        neighbor_nnz_h[i] = dist_nnz_h[neighbor_idx[i]];
        gpuErrchk( cudaMemcpy(neighbor_nnz_per_row_d + i * rows_this_rank,
            dist_nnz_per_row_d + neighbor_idx[i] * rows_this_rank,
            rows_this_rank * sizeof(int), cudaMemcpyHostToDevice) );
    }


    // alloc memory
    int **col_indices_d = new int*[neighbor_count];
    int **row_ptr_d = new int*[neighbor_count];
    for(int i = 0; i < neighbor_count; i++){
        gpuErrchk( cudaMalloc((void **)&col_indices_d[i], neighbor_nnz_h[i] * sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&row_ptr_d[i], (rows_this_rank + 1) * sizeof(int)) );
    }
    
    // create row ptr
    for(int i = 0; i < neighbor_count; i++){

        gpuErrchk(cudaMemset(row_ptr_d[i], 0, (rows_this_rank + 1) * sizeof(int)));
        void     *temp_storage_d = NULL;
        size_t   temp_storage_bytes = 0;
        // determines temporary device storage requirements for inclusive prefix sum
        cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes,
            neighbor_nnz_per_row_d + i * rows_this_rank, (row_ptr_d[i])+1, rows_this_rank);

        // Allocate temporary storage for inclusive prefix sum
        gpuErrchk(cudaMalloc(&temp_storage_d, temp_storage_bytes));
        // Run inclusive prefix sum
        // inclusive sum starting at second value to get the row ptr
        // which is the same as inclusive sum starting at first value and last value filled with nnz
        cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes,
            neighbor_nnz_per_row_d + i * rows_this_rank, (row_ptr_d[i])+1, rows_this_rank);

        // Free temporary storage
        gpuErrchk(cudaFree(temp_storage_d)); 

    }


    // column indices
    for(int i = 0; i < neighbor_count; i++){
        int neighbour = neighbor_idx[i];
        int rows_neighbour = gpubuf.count_K_device[neighbour];
        int disp_neighbour = gpubuf.displ_K_device[neighbour];

        blocks = (rows_this_rank + threads - 1) / threads;
        assemble_col_inds_K<<<blocks, threads>>>(
            gpubuf.site_x,
            gpubuf.site_y,
            gpubuf.site_z,
            nn_dist,
            rows_this_rank,
            rows_neighbour,
            N_left_tot + disp_this_rank,
            N_left_tot + disp_neighbour,
            row_ptr_d[i],
            col_indices_d[i]
        );
    }

    // *** Sparsity pattern of the distributed matrix

    gpubuf.K_distributed = new Distributed_matrix(
        N_interface,
        gpubuf.count_K_device,
        gpubuf.displ_K_device,
        neighbor_count,
        neighbor_idx,
        col_indices_d,
        row_ptr_d,
        neighbor_nnz_h,
        gpubuf.comm
    );

    gpubuf.K_p_distributed = new Distributed_vector(
        N_interface,
        gpubuf.count_K_device,
        gpubuf.displ_K_device,
        gpubuf.K_distributed->number_of_neighbours,
        gpubuf.K_distributed->neighbours,
        gpubuf.comm
    );

    // indices of the off-diagonal leftcontact-A matrix
    indices_creation_gpu_off_diagonal_block(
        gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
        gpubuf.lattice, pbc,
        nn_dist,
        rows_this_rank,
        N_left_tot,
        N_left_tot + disp_this_rank,
        0,
        &gpubuf.left_col_indices_d,
        &gpubuf.left_row_ptr_d,
        &gpubuf.left_nnz
    );

    // indices of the off-diagonal A-rightcontact matrix
    indices_creation_gpu_off_diagonal_block(
        gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
        gpubuf.lattice, pbc,
        nn_dist,
        rows_this_rank,
        N_right_tot,
        N_left_tot + disp_this_rank,
        N_left_tot + N_interface,
        &gpubuf.right_col_indices_d,
        &gpubuf.right_row_ptr_d,
        &gpubuf.right_nnz
    );


    for(int i = 0; i < neighbor_count; i++){
        gpuErrchk( cudaFree(col_indices_d[i]) );
        gpuErrchk( cudaFree(row_ptr_d[i]) );
    }   
    delete[] col_indices_d;
    delete[] row_ptr_d;
    delete[] neighbor_idx;
    delete[] dist_nnz_h;
    gpuErrchk( cudaFree(dist_nnz_d) );    
    gpuErrchk( cudaFree(dist_nnz_per_row_d) );
    delete[] neighbor_nnz_h;
    gpuErrchk( cudaFree(neighbor_nnz_per_row_d) );

    // DEBUG
    // dump A into a text file:
    // dump_csr_matrix_txt(N_interface, gpubuf.K_distributed->nnz, gpubuf.K_distributed->row_ptr_d[0],  gpubuf.K_distributed->col_indices_d[0], gpubuf.K_distributed->data_d[0], 0);
    // std::cout << "dumped csr matrix\n";
    // exit(1);
    // DEBUG

    // *** Sparsity pattern of the local matrix
    // This populates the site_CB_edge vector, and runs once at the beginning of a bias point 
    // Is not distributed (solution is computed on every rank)
    Assemble_K_sparsity(gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
                        gpubuf.lattice, pbc, nn_dist,
                        N_interface, N_left_tot, N_right_tot,
                        &gpubuf.Device_row_ptr_d, &gpubuf.Device_col_indices_d, &gpubuf.Device_nnz,
                        &gpubuf.contact_left_col_indices, &gpubuf.contact_left_row_ptr, &gpubuf.contact_left_nnz,
                        &gpubuf.contact_right_col_indices, &gpubuf.contact_right_row_ptr, &gpubuf.contact_right_nnz);
}




// **Sparsity pattern of X/T:


// Compute the number of nonzeros per row of the matrix including the injection, extraction, and device nodes (excluding the ground). 
// Has dimensions of Nsub by Nsub (by the cpu code)
__global__ void calc_nnz_per_row_X( const double *posx_d, const double *posy_d, const double *posz_d,
                                        const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                                        double nn_dist, const double tol,
                                        int num_source_inj, int num_ground_ext, const int num_layers_contact, const int num_atoms_reservoir,
                                        int num_metals, int matrix_size, int *nnz_per_row_d){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Natom = matrix_size - 2; 
    
    // TODO optimize this with a 2D grid instead of 1D
    for(int i = idx; i < Natom - 1; i += blockDim.x * gridDim.x){  // N_atom - 1 to exclude the ground node

        int nnz_row = 0;

        for(int j = 0; j < Natom - 1; j++){ // N_atom - 1 to exclude the ground node

            double dist = site_dist_gpu(posx_d[i], posy_d[i], posz_d[i],
                                        posx_d[j], posy_d[j], posz_d[j]);
            
            // diagonal terms
            if ( i == j )
            {
                nnz_row++;
            }

            // direct terms 
            else if ( i != j && dist < nn_dist )
            {
                nnz_row++;
            }

            // tunneling terms 
            else
            { 
                bool any_vacancy1 = element[i] == VACANCY;
                bool any_vacancy2 = element[j] == VACANCY;

                // contacts, excluding the last layer 
                bool metal1p = is_in_array_gpu(metals, element[i], num_metals) 
                                                && (i > ((num_layers_contact - 1)*num_source_inj))
                                                && (i < (Natom - num_atoms_reservoir - (num_layers_contact - 1)*num_ground_ext)); 

                bool metal2p = is_in_array_gpu(metals, element[j], num_metals)
                                                && (j > ((num_layers_contact - 1)*num_source_inj))
                                                && (j < (Natom - num_atoms_reservoir - (num_layers_contact - 1)*num_ground_ext));  

                // types of tunnelling conditions considered
                bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                bool contact_to_contact = (metal1p && metal2p);
                double local_E_drop = atom_CB_edge[i] - atom_CB_edge[j];                

                if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                {
                    nnz_row++;
                }
            }
        }

        nnz_per_row_d[i+2] = nnz_row;

        // source/ground connections
        if ( i < num_source_inj )
        {
            atomicAdd(&nnz_per_row_d[1], 1);
            nnz_per_row_d[i+2]++;
        }
        if ( i > (Natom - num_ground_ext) )
        {
            atomicAdd(&nnz_per_row_d[0], 1);
            nnz_per_row_d[i+2]++;
        }
        if ( i == 0 )
        {
            atomicAdd(&nnz_per_row_d[0], 2); // loop connection and diagonal element
            atomicAdd(&nnz_per_row_d[1], 2); // loop connection and diagonal element
        }

    }

}


// Compute the number of nonzeros per row of the matrix including the injection, extraction, and device nodes (excluding the ground). 
// Has dimensions of Nsub by Nsub (by the cpu code)
__global__ void calc_nnz_per_row_T( const double *posx_d, const double *posy_d, const double *posz_d,
                                    const ELEMENT *metals, const ELEMENT *element, const double *atom_CB_edge, const double *lattice, bool pbc,
                                    double nn_dist, const double tol,
                                    int num_source_inj, int num_ground_ext, const int num_layers_contact,
                                    int num_metals, int N_sub, int num_atoms_reservoir,
                                    int block_size_i, int block_size_j, int block_start_i, int block_start_j, int *nnz_per_row_d){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Natom = N_sub - 1; 
    
    // TODO optimize this with a 2D grid instead of 1D
    for(int row = idx; row < block_size_i; row += blockDim.x * gridDim.x){  // N_atom - 1 to exclude the ground node

        int nnz_row = 0;

        for(int col = 0; col < block_size_j; col++){

            int i = block_start_i + row; // i indexes the matrix and i-2 indexes the atoms
            int j = block_start_j + col; 

            double dist = site_dist_gpu(posx_d[i-2], posy_d[i-2], posz_d[i-2],
                                        posx_d[j-2], posy_d[j-2], posz_d[j-2]);
            
            if ( i == j ) // all diagonal terms
            {
                nnz_row++;
            }

            if ( (i == 0 && j == 1)  || (i == 1 && j == 0) ) // loop connection
            {
                nnz_row++;
            }

            if ( i == 0 && ( j > (Natom + 2 - num_ground_ext) )) // extraction terms minus ground node
            {
                nnz_row++;
            }

            if ( i == 1 && (j > 1) && (j < num_source_inj+2) ) // injection terms minus ground node
            {
                nnz_row++;
            }

            if (i > 1 && i != j)
            {

                // inj/ext terms
                if ( (j == 0) && ( i > ((N_sub+1) - num_ground_ext) ) )
                // if ( (j == 0) && ( i > (Natom - num_ground_ext) ) )
                {
                    nnz_row++;
                }

                if ( (j == 1) && (i > 1) && (i < num_source_inj + 2) ) 
                // if ( (j == 1) && (i > 1) && (i < num_source_inj) ) 
                {
                    nnz_row++;
                }

                if ( j > 1 && i != j )
                {
                    double dist = site_dist_gpu(posx_d[i-2], posy_d[i-2], posz_d[i-2],
                                                posx_d[j-2], posy_d[j-2], posz_d[j-2]);

                    // direct terms 
                    if ( dist < nn_dist ) 
                    {
                        nnz_row++;
                    }
                }

                // tunneling terms 
                if ( i != j && dist > nn_dist )
                { 
                    bool any_vacancy1 = element[i-2] == VACANCY;
                    bool any_vacancy2 = element[j-2] == VACANCY;

                    // contacts, excluding the last layer 
                    bool metal1p = is_in_array_gpu(metals, element[i-2], num_metals) 
                                                    && (i-2 > ((num_layers_contact - 1)*num_source_inj))
                                                    && (i-2 < (Natom - num_atoms_reservoir - (num_layers_contact - 1)*num_ground_ext)); 

                    bool metal2p = is_in_array_gpu(metals, element[j-2], num_metals)
                                                    && (j-2 > ((num_layers_contact - 1)*num_source_inj))
                                                    && (j-2 < (Natom - num_atoms_reservoir - (num_layers_contact - 1)*num_ground_ext));  

                    // types of tunnelling conditions considered
                    bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                    bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                    bool contact_to_contact = (metal1p && metal2p);
                    double local_E_drop = atom_CB_edge[i-2] - atom_CB_edge[j-2];                

                    if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                    {
                        nnz_row++;
                    }
                }
            }
        }
        __syncthreads();
        atomicAdd(&nnz_per_row_d[row], nnz_row); 
    }
}


__global__ void assemble_col_inds_X(const double *posx_d, const double *posy_d, const double *posz_d,
                                        const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                                        double nn_dist, const double tol,
                                        int num_source_inj, int num_ground_ext, const int num_layers_contact, const int num_atoms_reservoir,
                                        int num_metals, int matrix_size, int *nnz_per_row_d, int *row_ptr_d, int *col_indices_d)
{
    // row ptr is already calculated

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Natom = matrix_size - 2;
    int N_full = matrix_size;
    
    // TODO can be optimized with a 2D grid instead of 1D
    // INDEXED OVER NFULL
    for(int i = idx; i < N_full - 1; i += blockDim.x * gridDim.x){                      // exclude ground node with Nfull - 1

        int nnz_row = 0;

        // loop connection and injection row
        if ( i == 0 )
        {
            for (int j = 0; j < N_full - 1; j++)                                        // exclude ground node with Nfull - 1
            {
                if ( (j < 2) || j > (N_full - num_ground_ext) )
                {
                    col_indices_d[row_ptr_d[i] + nnz_row] = j;
                    nnz_row++;
                }
            }
        }
        // loop connection and extraction row
        if ( i == 1 )
        {
            for (int j = 0; j < num_source_inj + 2; j++)
            {
                col_indices_d[row_ptr_d[i] + nnz_row] = j;
                nnz_row++;
            }
        }

        // inner matrix terms
        if (i >= 2)
        {
            for(int j = 0; j < N_full - 1; j++){                                        // exclude ground node with Nfull - 1

                // add injection term for this row
                if ( (j == 1) && (i < num_source_inj + 2) )
                {
                    col_indices_d[row_ptr_d[i] + nnz_row] = 1;
                    nnz_row++;
                }

                // add extraction term for this row
                if ( (j == 0) && (i > N_full - num_ground_ext) )
                {
                    col_indices_d[row_ptr_d[i] + nnz_row] = 0;
                    nnz_row++;
                }

                if ( j >= 2 ) 
                {
                    double dist = site_dist_gpu(posx_d[i - 2], posy_d[i - 2], posz_d[i - 2],
                                                posx_d[j - 2], posy_d[j - 2], posz_d[j - 2]);
                    
                    // diagonal terms
                    if ( i == j )
                    {
                        col_indices_d[row_ptr_d[i] + nnz_row] = j;
                        nnz_row++;
                    }

                    // direct terms 
                    else if ( i != j && dist < nn_dist )
                    {
                        col_indices_d[row_ptr_d[i] + nnz_row] = j;
                        nnz_row++;
                    }

                    // tunneling terms 
                    else
                    { 
                        bool any_vacancy1 = element[i - 2] == VACANCY;
                        bool any_vacancy2 = element[j - 2] == VACANCY;

                        // contacts, excluding the last layer 
                        bool metal1p = is_in_array_gpu(metals, element[i - 2], num_metals) 
                                                    && ((i - 2) > ((num_layers_contact - 1)*num_source_inj))
                                                    && ((i - 2) < (Natom - num_atoms_reservoir - (num_layers_contact - 1)*num_ground_ext)); 

                        bool metal2p = is_in_array_gpu(metals, element[j - 2], num_metals)
                                                    && ((j - 2) > ((num_layers_contact - 1)*num_source_inj))
                                                    && ((j - 2) < (Natom - num_atoms_reservoir - (num_layers_contact - 1)*num_ground_ext));  

                        // bool metal1p = is_in_array_gpu(metals, element[i - 2], num_metals) 
                        //                             && ((i - 2) > ((num_layers_contact - 1)*num_source_inj))
                        //                             && ((i - 2) < (Natom - (num_layers_contact - 1)*num_ground_ext)); 

                        // bool metal2p = is_in_array_gpu(metals, element[j - 2], num_metals)
                        //                             && ((j - 2) > ((num_layers_contact - 1)*num_source_inj))
                        //                             && ((j - 2) < (Natom - (num_layers_contact - 1)*num_ground_ext));  

                        // types of tunnelling conditions considered
                        bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                        bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                        bool contact_to_contact = (metal1p && metal2p);
                        double local_E_drop = atom_CB_edge[i - 2] - atom_CB_edge[j - 2];                

                        if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                        {
                            col_indices_d[row_ptr_d[i] + nnz_row] = j;
                            nnz_row++;
                        }
                    }
                }
            }
        }

    }
}


__global__ void assemble_T_col_indices(const double *posx_d, const double *posy_d, const double *posz_d,
                                        const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                                        const double *lattice, bool pbc, double nn_dist, const double tol,
                                        int num_source_inj, int num_ground_ext, const int num_layers_contact,
                                        int num_metals, int N_sub, int num_atoms_reservoir, int *row_ptr_d, int *col_indices_d,
                                        int block_size_i, int block_size_j, int block_start_i, int block_start_j)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Natom = N_sub - 1; 

    for(int row = idx; row < block_size_i; row += blockDim.x * gridDim.x){  // Nsub x Nsub matrix divided into blocks by rank
        int nnz_row = 0;
        for(int col = 0; col < block_size_j; col++){

            int i = block_start_i + row; // this is where this rank's rows start, i indexes the matrix and i-2 indexes the atoms
            int j = block_start_j + col; 

            if ( i == j ) // all diagonal terms
            {
                col_indices_d[row_ptr_d[row] + nnz_row] = col;
                nnz_row++;
            }

            if ( (i == 0 && j == 1)  || (i == 1 && j == 0) ) // loop connection
            {
                col_indices_d[row_ptr_d[row] + nnz_row] = col;
                nnz_row++;
            }

            if ( i == 0 && ( j > ((N_sub+1) - num_ground_ext) )) // extraction terms minus ground node
            {
                col_indices_d[row_ptr_d[row] + nnz_row] = col;
                nnz_row++;
            }

            if ( i == 1 && (j > 1) && (j < num_source_inj+2) ) // injection terms minus ground node
            {
                col_indices_d[row_ptr_d[row] + nnz_row] = col;
                nnz_row++;
            }

            if (i > 1 && i != j)
            {
                // source/ground terms
                if ( (j == 1) && (i > 1) && (i < num_source_inj + 2) ) 
                {
                    col_indices_d[row_ptr_d[row] + nnz_row] = col;
                    nnz_row++;
                }
                if ( (j == 0) && ( i > ((N_sub+1) - num_ground_ext) ) )
                {
                    col_indices_d[row_ptr_d[row] + nnz_row] = col;
                    nnz_row++;
                }

                double dist = site_dist_gpu(posx_d[i-2], posy_d[i-2], posz_d[i-2],
                                            posx_d[j-2], posy_d[j-2], posz_d[j-2]);

                if ( j > 1 && i != j )
                {

                    // direct terms 
                    if ( dist < nn_dist )
                    {
                        col_indices_d[row_ptr_d[row] + nnz_row] = col;
                        nnz_row++;
                    }
                }

                // tunneling terms 
                if ( i != j && dist > nn_dist )
                { 
                    bool any_vacancy1 = element[i-2] == VACANCY;
                    bool any_vacancy2 = element[j-2] == VACANCY;

                    // contacts, excluding the last layer 
                    bool metal1p = is_in_array_gpu(metals, element[i-2], num_metals) 
                                                    && (i-2 > ((num_layers_contact - 1)*num_source_inj))
                                                    && (i-2 < (Natom - num_atoms_reservoir - (num_layers_contact - 1)*num_ground_ext)); 

                    bool metal2p = is_in_array_gpu(metals, element[j-2], num_metals)
                                                    && (j-2 > ((num_layers_contact - 1)*num_source_inj))
                                                    && (j-2 < (Natom - num_atoms_reservoir - (num_layers_contact - 1)*num_ground_ext));  

                    // types of tunnelling conditions considered
                    bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                    bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                    bool contact_to_contact = (metal1p && metal2p);
                    double local_E_drop = atom_CB_edge[i-2] - atom_CB_edge[j-2];                

                    if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                    {
                        col_indices_d[row_ptr_d[row] + nnz_row] = col;
                        nnz_row++;
                    }
                }
            }
        }
    }
}

// populates the row pointers and column indices of X
void Assemble_X_sparsity(int Natom, const double *posx, const double *posy, const double *posz,
                         const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                         const double *lattice, bool pbc, double nn_dist, const double tol,
                         int num_source_inj, int num_ground_ext, const int num_layers_contact, const int num_atoms_reservoir,
                         int num_metals, int *nnz_per_row_d, int **X_row_ptr, int **X_row_indices, int **X_col_indices, int *X_nnz)
{

    // number of atoms + ground node + driver nodes 
    int Nfull = Natom + 2;
    int matrix_size = Nfull; 

    // Compute the number of nonzeros per row of the matrix (Nsub x Nsub) - reusing the nnz_per_row buffer
    gpuErrchk( cudaMemset(nnz_per_row_d, 0, matrix_size * sizeof(int)) );

    int threads = 512;
    int blocks = (matrix_size + threads - 1) / threads;
    calc_nnz_per_row_X<<<blocks, threads>>>(posx, posy, posz,
                         metals, element, atom_charge, atom_CB_edge,
                         nn_dist, tol,
                         num_source_inj, num_ground_ext, num_layers_contact, num_atoms_reservoir,
                         num_metals, matrix_size, nnz_per_row_d);
    // cudaDeviceSynchronize();

    // Set the row pointers according to the cumulative sum of the nnz per row (total nnz is the last element of the row pointer)
    gpuErrchk( cudaMalloc((void **)X_row_ptr, (matrix_size + 1 - 1) * sizeof(int)) );   // subtract 1 to ignore the ground node
    gpuErrchk( cudaMemset((*X_row_ptr), 0, (matrix_size + 1 - 1) * sizeof(int)) );      // subtract 1 to ignore the ground node

    auto t1 = std::chrono::steady_clock::now();

    // V1. inclusive scan with CUB:
    // void     *temp_storage_d = NULL;                                                    // determines temporary device storage requirements for inclusive prefix sum
    // size_t   temp_storage_bytes = 0;
    // cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*X_row_ptr)+1, matrix_size - 1); // subtract 1 to ignore the ground node
    // gpuErrchk( cudaMalloc(&temp_storage_d, temp_storage_bytes) );                             // inclusive sum starting at second value to get the row ptr, which is the same as inclusive sum starting at first value and last value filled with nnz
    // cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*X_row_ptr)+1, matrix_size - 1);
    // cudaFree(temp_storage_d);

    // V2. inclusive scan with thrust:
    thrust::device_ptr<int> nnz_per_row_thrust(nnz_per_row_d);
    thrust::device_ptr<int> X_row_ptr_thrust((*X_row_ptr) + 1);
    thrust::inclusive_scan(nnz_per_row_thrust, nnz_per_row_thrust + matrix_size - 1, X_row_ptr_thrust); // subtract 1 to ignore the ground node

    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t2 - t1;
    std::cout << "time to calc row pointers: " << dt.count() << "\n";
    
    // total number of nonzeros
    gpuErrchk( cudaMemcpy(X_nnz, (*X_row_ptr) + matrix_size - 1, sizeof(int), cudaMemcpyDeviceToHost) );
    // std::cout << "\nsparse nnz: " << *X_nnz << std::endl;

    // assemble the column indices from 0 to Nsub (excluding the ground node)
    gpuErrchk( cudaMalloc((void **)X_col_indices, X_nnz[0] * sizeof(int)) );
    assemble_col_inds_X<<<blocks, threads>>>(posx, posy, posz,
                         metals, element, atom_charge, atom_CB_edge,
                         nn_dist, tol,
                         num_source_inj, num_ground_ext, num_layers_contact, num_atoms_reservoir,
                         num_metals, matrix_size, nnz_per_row_d,
                        (*X_row_ptr),
                        (*X_col_indices));
    
    // // get the row indices for COO if it's needed:
    // int *X_row_indices_h = new int[X_nnz[0]];
    // int *X_row_ptr_h = new int[Natom + 2];

    // gpuErrchk( cudaMemcpy(X_row_ptr_h, (*X_row_ptr), (Natom + 2) * sizeof(int), cudaMemcpyDeviceToHost) );
    // for(int i = 0; i < Natom + 1; i++){
    //     for(int j = X_row_ptr_h[i]; j < X_row_ptr_h[i+1]; j++){
    //         X_row_indices_h[j] = i;
    //     }
    // }

    // gpuErrchk( cudaMalloc((void **)X_row_indices, X_nnz[0] * sizeof(int)) );
    // gpuErrchk( cudaMemcpy((*X_row_indices), X_row_indices_h, X_nnz[0] * sizeof(int), cudaMemcpyHostToDevice) );
    // free(X_row_indices_h);
    // free(X_row_ptr_h);

}


// Populate X: CSR format inputs
__global__ void populate_sparse_X_CSR(const double *posx_d, const double *posy_d, const double *posz_d,
                                        const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                                        double nn_dist, const double tol,
                                        const double high_G, const double low_G, const double loop_G, 
                                        const double Vd, const double m_e, const double V0,
                                        int num_source_inj, int num_ground_ext, const int num_layers_contact, const int num_atoms_reservoir,
                                        int num_metals, int matrix_size, int *row_ptr_d, int *col_indices_d, double *data_d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N_full = matrix_size;
    int N_atom = matrix_size - 2;
    
    for(int i = idx; i < N_full - 1; i += blockDim.x * gridDim.x){

        for( int j = row_ptr_d[i]; j < row_ptr_d[i+1]; j++ )
        {
            // col_indices_d[j] is the index of j in the matrix. j is the index of the data vector
            // if dealing with a diagonal element, we add the positive value from i = i and j = N_full to include the ground node

            data_d[j] = 0.0;

            // extraction boundary (row)
            if(i == 0)
            {
                // diagonal element (0, 0) --> add the value from (0, N_full)
                if (col_indices_d[j] == 0)
                {
                    data_d[j] = +high_G;
                }
                // loop connection (0, 1)
                if (col_indices_d[j] == 1)
                {
                    data_d[j] = -loop_G;
                }
                // extraction connections from the device
                if ( col_indices_d[j] > N_full - num_ground_ext )
                {
                    data_d[j] = -high_G;
                } 
            }

            // injection boundary (row)
            if(i == 1)
            {
                // loop connection (1, 0)
                if (col_indices_d[j] == 0)
                {
                    data_d[j] = -loop_G;
                }
                // injection connections to the device
                if ( col_indices_d[j] >= 2 || (col_indices_d[j] > N_full - num_ground_ext) )
                {
                    data_d[j] = -high_G;
                } 
            }

            // inner matrix terms
            if (i >= 2)
            {
                // diagonal elements --> add the value from (i - 2, N_full - 2) if site i - 2 neighbors the ground node
                if (i == col_indices_d[j])
                {
                    double dist_angstrom = site_dist_gpu(posx_d[i - 2], posy_d[i - 2], posz_d[i - 2],
                                                        posx_d[N_atom-1], posy_d[N_atom-1], posz_d[N_atom-1]);                                   
                    bool neighboring_ground = (dist_angstrom < nn_dist);
                    
                    if (neighboring_ground) 
                    {
                        data_d[j] = +high_G;     // assuming all the connections to ground come from the right contact
                    } 
                }

                // extraction boundary (column)
                if ( (col_indices_d[j] == 0) && (i > N_full - num_ground_ext) )
                {
                    data_d[j] = -high_G;
                }

                // injection boundary (column)
                if ( (col_indices_d[j] == 1) && (i < num_source_inj + 2) )
                {
                    data_d[j] = -high_G;
                }

                // off-diagonal inner matrix elements
                if ( (col_indices_d[j] >= 2) && (col_indices_d[j] != i)) 
                {

                    double dist_angstrom = site_dist_gpu(posx_d[i - 2], posy_d[i - 2], posz_d[i - 2],
                                                         posx_d[col_indices_d[j] - 2], posy_d[col_indices_d[j] - 2], posz_d[col_indices_d[j] - 2]);                                       
                        
                    bool neighbor = (dist_angstrom < nn_dist);                                                      

                    // non-neighbor connections
                    if (!neighbor)
                    {
                        bool any_vacancy1 = element[i - 2] == VACANCY;
                        bool any_vacancy2 = element[col_indices_d[j] - 2] == VACANCY;

                        // contacts, excluding the last layer 
                        bool metal1p = is_in_array_gpu(metals, element[i - 2], num_metals) 
                                                    && ((i - 2) > ((num_layers_contact - 1)*num_source_inj))
                                                    && ((i - 2) < (N_full - num_atoms_reservoir - (num_layers_contact - 1)*num_ground_ext)); 

                        bool metal2p = is_in_array_gpu(metals, element[col_indices_d[j] - 2], num_metals)
                                                    && ((col_indices_d[j] - 2) > ((num_layers_contact - 1)*num_source_inj))
                                                    && ((col_indices_d[j] - 2) < (N_full - num_atoms_reservoir - (num_layers_contact - 1)*num_ground_ext));  

                        // types of tunnelling conditions considered
                        bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                        bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                        bool contact_to_contact = (metal1p && metal2p);

                        double local_E_drop = atom_CB_edge[i - 2] - atom_CB_edge[col_indices_d[j] - 2];                // [eV] difference in energy between the two atoms

                        // compute the WKB tunneling coefficients for all the tunnelling conditions
                        if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                        {
                                
                            double prefac = -(sqrt( 2 * m_e ) / h_bar) * (2.0 / 3.0);           // [s/(kg^1/2 * m^2)] coefficient inside the exponential
                            double dist = (1e-10)*dist_angstrom;                                // [m] 3D distance between atoms i and j

                            if (contact_to_trap)
                            {
                                double energy_window = fabs(local_E_drop);                      // [eV] energy window for tunneling from the contacts
                                double dV = 0.01;                                               // [V] energy spacing for numerical integration
                                double dE = eV_to_J * dV;                                       // [eV] energy spacing for numerical integration
                                        
                                // integrate over all the occupied energy levels in the contact
                                double T = 0.0;
                                for (double iv = 0; iv < energy_window; iv += dE)
                                {
                                    double E1 = eV_to_J * V0 + iv;                                  // [J] Energy distance to CB before tunnelling
                                    double E2 = E1 - fabs(local_E_drop);                            // [J] Energy distance to CB after tunnelling

                                    if (E2 > 0)                                                     // trapezoidal potential barrier (low field)                 
                                    {                                                           
                                        T += exp(prefac * (dist / fabs(local_E_drop)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                                    }

                                    if (E2 < 0)                                                      // triangular potential barrier (high field)                               
                                    {
                                        T += exp(prefac * (dist / fabs(local_E_drop)) * ( pow(E1, 1.5) )); 
                                    } 
                                }
                                data_d[j] = -T;
                            } 
                            else 
                            {
                                double E1 = eV_to_J * V0;                                        // [J] Energy distance to CB before tunnelling
                                double E2 = E1 - fabs(local_E_drop);                             // [J] Energy distance to CB after tunnelling
                                        
                                if (E2 > 0)                                                      // trapezoidal potential barrier (low field)
                                {                                                           
                                    double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                                    data_d[j] = -T;
                                }

                                if (E2 < 0)                                                        // triangular potential barrier (high field)
                                {
                                    double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) ));
                                    data_d[j] = -T;
                                }
                            }
                        }
                    }

                    // direct terms
                    if ( neighbor )
                    {
                        // contacts
                        bool metal1 = is_in_array_gpu<ELEMENT>(metals, element[i - 2], num_metals);
                        bool metal2 = is_in_array_gpu<ELEMENT>(metals, element[col_indices_d[j] - 2], num_metals);

                        bool vacancy1 = (element[i - 2] == VACANCY);
                        bool vacancy2 = (element[col_indices_d[j] - 2] == VACANCY);

                        if ((metal1 && metal2) || (vacancy1 && vacancy2) || (metal1 && vacancy2) || (vacancy1 && metal2))
                        {
                            data_d[j] = -high_G;
                        }
                        else
                        {
                            data_d[j] = -low_G;
                        }
                    }

                }
            }
        }
    }

    __syncthreads();

    if (idx < matrix_size - 1){
        //reduce the elements in the row
        double tmp = 0.0;
        for(int j = row_ptr_d[idx]; j < row_ptr_d[idx+1]; j++){
            if(idx != col_indices_d[j]){
                tmp += data_d[j];
            }
        }
        //write the sum of the off-diagonals onto the existing diagonal element
        for(int j = row_ptr_d[idx]; j < row_ptr_d[idx+1]; j++){
            if(idx == col_indices_d[j]){
                data_d[j] += -tmp;
            }
        }
    }
}

// Populate X: COO format inputs
__global__ void populate_sparse_X_COO(const double *posx_d, const double *posy_d, const double *posz_d,
                                        const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                                        const double *lattice, bool pbc, double nn_dist, const double tol,
                                        const double high_G, const double low_G, const double loop_G, 
                                        const double Vd, const double m_e, const double V0,
                                        int num_source_inj, int num_ground_ext, const int num_layers_contact, const int num_atoms_reservoir,
                                        int num_metals, int matrix_size, int *row_indices_d, int *row_ptr_d, int *col_indices_d, double *data_d, int X_nnz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N_full = matrix_size;
    int N_atom = matrix_size - 2;
    
    for(int id = idx; id < X_nnz; id += blockDim.x * gridDim.x){

        int i = row_indices_d[id];
        int j = id;

        if(i >= N_full-1){
            continue;
        }

        data_d[j] = 0.0;

        // col_indices_d[j] is the index of j in the matrix. j is the index of the data vector
        // if dealing with a diagonal element, we add the positive value from i = i and j = N_full to include the ground node

        // extraction boundary (row)
        if(i == 0)
        {
            // diagonal element (0, 0) --> add the value from (0, N_full)
            if (col_indices_d[j] == 0)
            {
                data_d[j] = +high_G;
            }
            // loop connection (0, 1)
            if (col_indices_d[j] == 1)
            {
                data_d[j] = -loop_G;
            }
            // extraction connections from the device
            if ( col_indices_d[j] > N_full - num_ground_ext )
            {
                data_d[j] = -high_G;
            } 
        }

        // injection boundary (row)
        if(i == 1)
        {
            // loop connection (1, 0)
            if (col_indices_d[j] == 0)
            {
                data_d[j] = -loop_G;
            }
            // injection connections to the device
            if ( col_indices_d[j] >= 2 || (col_indices_d[j] > N_full - num_ground_ext) )
            {
                data_d[j] = -high_G;
            } 
        }

        // inner matrix terms
        if (i >= 2)
        {
            // diagonal elements --> add the value from (i - 2, N_full - 2) if site i - 2 neighbors the ground node
            if (i == col_indices_d[j])
            {
                double dist_angstrom = site_dist_gpu(posx_d[i - 2], posy_d[i - 2], posz_d[i - 2],
                                                        posx_d[N_atom-1], posy_d[N_atom-1], posz_d[N_atom-1], 
                                                        lattice[0], lattice[1], lattice[2], pbc);                                   
                bool neighboring_ground = (dist_angstrom < nn_dist);
                
                if (neighboring_ground) 
                {
                    data_d[j] = +high_G;     // assuming all the connections to ground come from the right contact
                } 
            }

            // extraction boundary (column)
            if ( (col_indices_d[j] == 0) && (i > N_full - num_ground_ext) )
            {
                data_d[j] = -high_G;
            }

            // injection boundary (column)
            if ( (col_indices_d[j] == 1) && (i < num_source_inj + 2) )
            {
                data_d[j] = -high_G;
            }

            // off-diagonal inner matrix elements
            if ( (col_indices_d[j] >= 2) && (col_indices_d[j] != i)) 
            {

                double dist_angstrom = site_dist_gpu(posx_d[i - 2], posy_d[i - 2], posz_d[i - 2],
                                                    posx_d[col_indices_d[j] - 2], posy_d[col_indices_d[j] - 2], posz_d[col_indices_d[j] - 2], 
                                                    lattice[0], lattice[1], lattice[2], pbc);                                       
                    
                bool neighbor = (dist_angstrom < nn_dist);                                                      

                // non-neighbor connections
                if (!neighbor)
                {
                    bool any_vacancy1 = element[i - 2] == VACANCY;
                    bool any_vacancy2 = element[col_indices_d[j] - 2] == VACANCY;

                    // contacts, excluding the last layer 
                    bool metal1p = is_in_array_gpu(metals, element[i - 2], num_metals) 
                                                && ((i - 2) > ((num_layers_contact - 1)*num_source_inj))
                                                && ((i - 2) < (N_full - num_atoms_reservoir - (num_layers_contact - 1)*num_ground_ext)); 

                    bool metal2p = is_in_array_gpu(metals, element[col_indices_d[j] - 2], num_metals)
                                                && ((col_indices_d[j] - 2) > ((num_layers_contact - 1)*num_source_inj))
                                                && ((col_indices_d[j] - 2) < (N_full - num_atoms_reservoir - (num_layers_contact - 1)*num_ground_ext));  

                    // types of tunnelling conditions considered
                    bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                    bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                    bool contact_to_contact = (metal1p && metal2p);

                    double local_E_drop = atom_CB_edge[i - 2] - atom_CB_edge[col_indices_d[j] - 2];                // [eV] difference in energy between the two atoms

                    // compute the WKB tunneling coefficients for all the tunnelling conditions
                    if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                    {
                            
                        double prefac = -(sqrt( 2 * m_e ) / h_bar) * (2.0 / 3.0);           // [s/(kg^1/2 * m^2)] coefficient inside the exponential
                        double dist = (1e-10)*dist_angstrom;                                // [m] 3D distance between atoms i and j

                        if (contact_to_trap)
                        {
                            double energy_window = fabs(local_E_drop);                      // [eV] energy window for tunneling from the contacts
                            double dV = 0.01;                                               // [V] energy spacing for numerical integration
                            double dE = eV_to_J * dV;                                       // [eV] energy spacing for numerical integration
                                    
                            // integrate over all the occupied energy levels in the contact
                            double T = 0.0;
                            for (double iv = 0; iv < energy_window; iv += dE)
                            {
                                double E1 = eV_to_J * V0 + iv;                                  // [J] Energy distance to CB before tunnelling
                                double E2 = E1 - fabs(local_E_drop);                            // [J] Energy distance to CB after tunnelling

                                if (E2 > 0)                                                     // trapezoidal potential barrier (low field)                 
                                {                                                           
                                    T += exp(prefac * (dist / fabs(local_E_drop)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                                }

                                if (E2 < 0)                                                      // triangular potential barrier (high field)                               
                                {
                                    T += exp(prefac * (dist / fabs(local_E_drop)) * ( pow(E1, 1.5) )); 
                                } 
                            }
                            data_d[j] = -T;
                        } 
                        else 
                        {
                            double E1 = eV_to_J * V0;                                        // [J] Energy distance to CB before tunnelling
                            double E2 = E1 - fabs(local_E_drop);                             // [J] Energy distance to CB after tunnelling
                                    
                            if (E2 > 0)                                                      // trapezoidal potential barrier (low field)
                            {                                                           
                                double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                                data_d[j] = -T;
                            }

                            if (E2 < 0)                                                        // triangular potential barrier (high field)
                            {
                                double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) ));
                                data_d[j] = -T;
                            }
                        }
                    }
                }

                // direct terms
                if ( neighbor )
                {
                    // contacts
                    bool metal1 = is_in_array_gpu<ELEMENT>(metals, element[i - 2], num_metals);
                    bool metal2 = is_in_array_gpu<ELEMENT>(metals, element[col_indices_d[j] - 2], num_metals);

                    bool vacancy1 = (element[i - 2] == VACANCY);
                    bool vacancy2 = (element[col_indices_d[j] - 2] == VACANCY);

                    if ((metal1 && metal2) || (vacancy1 && vacancy2) || (metal1 && vacancy2) || (vacancy1 && metal2))
                    {
                        data_d[j] = -high_G;
                    }
                    else
                    {
                        data_d[j] = -low_G;
                    }
                }

            }
        }
    }

}


__global__ void calc_diagonal_CSR(
    int *col_indices,
    int *row_ptr,
    double *data,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < matrix_size - 1; i += blockDim.x * gridDim.x){ // MINUS ONE
        //reduce the elements in the row
        double tmp = 0.0;
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i != col_indices[j]){
                tmp += data[j];
            }
        }
        //write the sum of the off-diagonals onto the existing diagonal element
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i == col_indices[j]){
                data[j] += -tmp;
            }
        }
    }
}

void Assemble_X(int Natom, const double *posx, const double *posy, const double *posz,
                const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                const double *lattice, bool pbc, double nn_dist, const double tol, const double Vd, const double m_e, const double V0,
                const double high_G, const double low_G, const double loop_G,
                int num_source_inj, int num_ground_ext, const int num_layers_contact, const int num_atoms_reservoir,
                int num_metals, double **X_data, int **X_row_indices,
                int **X_row_ptr, int **X_col_indices, int *X_nnz){

    auto t0 = std::chrono::steady_clock::now();

    // parallelize over rows
    int Nfull = Natom + 2;
    int threads = 512;

    // allocate the data array and initialize it to zeros
    gpuErrchk(cudaMalloc((void **)X_data, X_nnz[0] * sizeof(double)));
    // gpuErrchk(cudaMemset((*X_data), 0, X_nnz[0] * sizeof(double)));

    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "time to allocate X_data: " << dt.count() << "\n";

    // this COO version does not update the diagonal - also enable COO indices in the sparsity function first!
    // int blocks2 = (X_nnz[0] + threads - 1) / threads;
    // populate_sparse_X_COO<<<blocks2, threads>>>(posx, posy, posz,
    //                                             metals, element, atom_charge, atom_CB_edge,
    //                                             lattice, pbc, nn_dist, tol, high_G, low_G, loop_G,
    //                                             Vd, m_e, V0,
    //                                             num_source_inj, num_ground_ext, num_layers_contact, num_atoms_reservoir,
    //                                             num_metals, Nfull, *X_row_indices, *X_row_ptr, *X_col_indices, *X_data, X_nnz[0]);
    // add the off diagonals onto the diagonal
    // calc_diagonal_CSR<<<blocks2, threads>>>(*X_col_indices, *X_row_ptr, *X_data, Nfull);

    // the CSR version internally updates the diagonal
    int blocks = (Nfull + threads - 1) / threads;
    populate_sparse_X_CSR<<<blocks, threads>>>(posx, posy, posz,
                                               metals, element, atom_charge, atom_CB_edge,
                                               nn_dist, tol, high_G, low_G, loop_G,
                                               Vd, m_e, V0,
                                               num_source_inj, num_ground_ext, num_layers_contact, num_atoms_reservoir,
                                               num_metals, Nfull, *X_row_ptr, *X_col_indices, *X_data);
    
    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt2 = t2 - t1;
    std::cout << "time to populate matrix: " << dt2.count() << "\n";

}


// distributed T:
void Assemble_T_sparsity(GPUBuffers &gpubuf, int pbc, int N_atom, int num_atoms_reservoir, const double nn_dist, int num_source_inj, int num_ground_ext, int num_layers_contact)
{
    int N_sub = N_atom + 1;
    int rank = gpubuf.rank;
    int size = gpubuf.size;
    int rows_this_rank = gpubuf.count_T_device[rank];
    int disp_this_rank = gpubuf.displ_T_device[rank];
    
    int *dist_nnz_h = new int[gpubuf.size];
    int *dist_nnz_d;
    int *dist_nnz_per_row_d;

    gpuErrchk( cudaMalloc((void **)&dist_nnz_d, gpubuf.size * sizeof(int)) );
    gpuErrchk(cudaMemset(dist_nnz_d, 0, gpubuf.size * sizeof(int)));
    gpuErrchk( cudaMalloc((void **)&dist_nnz_per_row_d, gpubuf.size * rows_this_rank * sizeof(int)) );
    gpuErrchk(cudaMemset(dist_nnz_per_row_d, 0, gpubuf.size * rows_this_rank * sizeof(int)));

    // loop over the size to determine neighbours
    for(int i = 0; i < size; i++){
        int rows_other = gpubuf.count_T_device[i];
        int displ_other = gpubuf.displ_T_device[i];

        int threads = 1024;
        //start with self
        int blocks = (rows_this_rank - 1) / threads + 1;

        double tol = eV_to_J * 0.01;                                                                // [eV] tolerance after which the barrier slope is considered
        int num_metals = 2;
        calc_nnz_per_row_T<<<blocks, threads>>>(gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
                           gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_CB_edge, gpubuf.lattice, pbc,
                           nn_dist, tol, num_source_inj, num_ground_ext, num_layers_contact,
                           num_metals, N_sub, num_atoms_reservoir, rows_this_rank, rows_other, disp_this_rank, displ_other, dist_nnz_per_row_d + (size_t)i * (size_t)rows_this_rank);

        // reduce nnz per row
        void     *temp_storage_d = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(
        temp_storage_d, temp_storage_bytes, 
            dist_nnz_per_row_d + i * rows_this_rank,
            dist_nnz_d + i, rows_this_rank);

        // Allocate temporary storage
        cudaMalloc(&temp_storage_d, temp_storage_bytes);

        // Run sum-reduction
        cub::DeviceReduce::Sum(temp_storage_d, temp_storage_bytes,
            dist_nnz_per_row_d + i * rows_this_rank,
            dist_nnz_d + i, rows_this_rank);
    }
 
    gpuErrchk( cudaMemcpy(dist_nnz_h, dist_nnz_d, size * sizeof(int), cudaMemcpyDeviceToHost) );
    // counting neighbours
    int neighbor_count = 0;
    for(int i = 0; i < size; i++){
        if(dist_nnz_h[i] > 0){
            neighbor_count++;
        }
    }    

    // // print dist_nnz_h:
    // for (int i = 0; i < size; i++)
    // {
    //     std::cout << "rank " << gpubuf.rank << "T dist_nnz_h[" << i << "] = " << dist_nnz_h[i] << std::endl;
    // }
    // std::cout << "rank " << gpubuf.rank <<  "T neighbor_count = " << neighbor_count << std::endl;
    // exit(1);

    // get the indices of the neighbours
    int *neighbor_idx = new int[neighbor_count];
    int *neighbor_nnz_h = new int[neighbor_count];
    int *neighbor_nnz_per_row_d;
    gpuErrchk( cudaMalloc((void **)&neighbor_nnz_per_row_d, neighbor_count * rows_this_rank * sizeof(int)) );

    // determine neighbours
    neighbor_count = 0;
    for(int i = 0; i < size; i++){
        int neighbor = (i+rank) % size;
        if(dist_nnz_h[neighbor] > 0){
            neighbor_idx[neighbor_count] = neighbor;
            neighbor_count++;
        }
    }    

    std::cout << "rank " << gpubuf.rank <<  "T neighbor_idx = " << neighbor_count << std::endl;   

    // fill the neighbor nnz
    for(int i = 0; i < neighbor_count; i++){
        neighbor_nnz_h[i] = dist_nnz_h[neighbor_idx[i]];
        gpuErrchk( cudaMemcpy(neighbor_nnz_per_row_d + i * rows_this_rank,
            dist_nnz_per_row_d + neighbor_idx[i] * rows_this_rank,
            rows_this_rank * sizeof(int), cudaMemcpyHostToDevice) );
    }

    // alloc memory
    int **col_indices_d = new int*[neighbor_count];
    int **row_ptr_d = new int*[neighbor_count];
    for(int i = 0; i < neighbor_count; i++){
        gpuErrchk( cudaMalloc((void **)&col_indices_d[i], neighbor_nnz_h[i] * sizeof(int)) );
        gpuErrchk( cudaMalloc((void **)&row_ptr_d[i], (rows_this_rank + 1) * sizeof(int)) );
    }
    
    // create row ptr
    for(int i = 0; i < neighbor_count; i++){

        gpuErrchk(cudaMemset(row_ptr_d[i], 0, (rows_this_rank + 1) * sizeof(int)));
        void     *temp_storage_d = NULL;
        size_t   temp_storage_bytes = 0;
        // determines temporary device storage requirements for inclusive prefix sum
        cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes,
            neighbor_nnz_per_row_d + i * rows_this_rank, (row_ptr_d[i])+1, rows_this_rank);

        // Allocate temporary storage for inclusive prefix sum
        gpuErrchk(cudaMalloc(&temp_storage_d, temp_storage_bytes));
        // Run inclusive prefix sum
        // inclusive sum starting at second value to get the row ptr
        // which is the same as inclusive sum starting at first value and last value filled with nnz
        cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes,
            neighbor_nnz_per_row_d + i * rows_this_rank, (row_ptr_d[i])+1, rows_this_rank);

        // Free temporary storage
        gpuErrchk(cudaFree(temp_storage_d)); 

    }

    // column indices
    for(int i = 0; i < neighbor_count; i++){
        int neighbour = neighbor_idx[i];
        int rows_neighbour = gpubuf.count_T_device[neighbour];
        int disp_neighbour = gpubuf.displ_T_device[neighbour];

        int threads = 1024;
        int blocks = (rows_this_rank + threads - 1) / threads;
       
        double tol = eV_to_J * 0.01;                                                                // [eV] tolerance after which the barrier slope is considered
        int num_metals = 2;
        assemble_T_col_indices<<<blocks, threads>>>(gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
                           gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_charge, gpubuf.atom_CB_edge,
                           gpubuf.lattice, pbc, nn_dist, tol,
                           num_source_inj, num_ground_ext, num_layers_contact,
                           num_metals, N_sub, num_atoms_reservoir, 
                           row_ptr_d[i],
                           col_indices_d[i], rows_this_rank, rows_neighbour, disp_this_rank, disp_neighbour);
    }

    // cudaDeviceSynchronize();
    // auto t3 = std::chrono::steady_clock::now();

    gpubuf.T_distributed = new Distributed_matrix(
        N_sub,
        gpubuf.count_T_device,
        gpubuf.displ_T_device,
        neighbor_count,
        neighbor_idx,
        col_indices_d,
        row_ptr_d,
        neighbor_nnz_h,
        gpubuf.comm
    );

    // cudaDeviceSynchronize();
    // auto t3p = std::chrono::steady_clock::now();
    // std::chrono::duration<double> dt = t3p - t3;
    // std::cout << "time to create distributed matrix: " << dt.count() << "\n"; // 6k sites, 1.06895 s

    gpubuf.T_p_distributed = new Distributed_vector(
        N_sub,
        gpubuf.count_T_device,
        gpubuf.displ_T_device,
        gpubuf.T_distributed->number_of_neighbours,
        gpubuf.T_distributed->neighbours,
        gpubuf.comm
    );

    for(int i = 0; i < neighbor_count; i++){
        gpuErrchk( cudaFree(col_indices_d[i]) );
        gpuErrchk( cudaFree(row_ptr_d[i]) );
    }   

    delete[] col_indices_d;
    delete[] row_ptr_d;
    delete[] neighbor_idx;
    delete[] dist_nnz_h;

    gpuErrchk( cudaFree(dist_nnz_d) );
    gpuErrchk( cudaFree(dist_nnz_per_row_d) );
    delete[] neighbor_nnz_h;
    gpuErrchk( cudaFree(neighbor_nnz_per_row_d) );    

}

// assemble the data for the X matrix - 1D distribution over rows
__global__ void populate_T_dist_neighbor(const double *posx_d, const double *posy_d, const double *posz_d,
                                const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                                double nn_dist, const double tol,
                                const double high_G, const double low_G, const double loop_G, 
                                const double Vd, const double m_e, const double V0,
                                int num_source_inj, int num_ground_ext, const int num_layers_contact,
                                int num_metals, int matrix_size, int num_atoms_reservoir,
                                int *col_indices_d, int *row_ptr_d, double *data_d,
                                int size_i, int size_j, int start_i, int start_j)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Nsub = matrix_size;
    int N_atom = matrix_size - 1;
    int N_full = matrix_size + 1;

    for(int id = idx; id < size_i; id += blockDim.x * gridDim.x){
        for( int jd = row_ptr_d[id]; jd < row_ptr_d[id+1]; jd++ )
        {
            int i = start_i + id;
            int j = start_j + col_indices_d[jd];
            data_d[jd] = 0.0;

            // col_indices_d[j] is the index of j in the matrix. j is the index of the data vector
            // if dealing with a diagonal element, we add the positive value from i = i and j = N_full to include the ground node

            // extraction boundary (row)
            if(i == 0)
            {
                // diagonal element (0, 0) --> add the value from (0, N_full)
                if (j == 0)
                {
                    data_d[jd] = +high_G;
                }
                // loop connection (0, 1)
                else if (j == 1)
                {
                    data_d[jd] = -loop_G;
                }
                // extraction connections from the device
                else
                {
                    data_d[jd] = -high_G;
                }  
            }

            // injection boundary (row)
            if(i == 1)
            {
                // loop connection (1, 0) 
                if (j == 0)
                {
                    data_d[jd] = -loop_G;
                }
                // injection connections to the device
                // else
                if ( j > 1 )
                {
                    data_d[jd] = -high_G;
                } 
            }

            // inner matrix terms
            if (i >= 2)
            {
                // diagonal elements --> add the value from (i - 2, N_full - 2) if site i - 2 neighbors the ground node
                if (i == j)
                {
                    double dist_angstrom = site_dist_gpu(posx_d[i - 2], posy_d[i - 2], posz_d[i - 2],
                                                         posx_d[N_atom-1], posy_d[N_atom-1], posz_d[N_atom-1]);                                   
                    bool neighboring_ground = (dist_angstrom < nn_dist);
                    
                    if (neighboring_ground) 
                    {
                        data_d[jd] = +high_G;     // assuming all the connections to ground come from the right contact
                    } 
                }

                // extraction boundary (column)
                if ( (j == 0) && (i > (Nsub+1) - num_ground_ext) )
                {
                    data_d[jd] = -high_G;
                }

                // injection boundary (column)
                if ( (j == 1) && (i > 1) && (i < num_source_inj + 2) )
                {
                    data_d[jd] = -high_G;
                }

                // off-diagonal inner matrix elements
                if ( (j >= 2) && (j != i)) 
                {
                    double dist_angstrom = site_dist_gpu(posx_d[i - 2], posy_d[i - 2], posz_d[i - 2],
                                                         posx_d[j - 2], posy_d[j - 2], posz_d[j - 2]);                                       
                        
                    bool neighbor = (dist_angstrom < nn_dist);                                                      

                    // // non-neighbor connections
                    // if (!neighbor)
                    // {
                    //     bool any_vacancy1 = element[i - 2] == VACANCY;
                    //     bool any_vacancy2 = element[j - 2] == VACANCY;

                    //     // contacts, excluding the last layer 
                    //     bool metal1p = is_in_array_gpu(metals, element[i-2], num_metals) 
                    //                                     && (i-2 > ((num_layers_contact - 1)*num_source_inj))
                    //                                     && (i-2 < (N_full - num_atoms_reservoir - (num_layers_contact - 1)*num_ground_ext)); 

                    //     bool metal2p = is_in_array_gpu(metals, element[j-2], num_metals)
                    //                                     && (j-2 > ((num_layers_contact - 1)*num_source_inj))
                    //                                     && (j-2 < (N_full - num_atoms_reservoir - (num_layers_contact - 1)*num_ground_ext));  

                    //     // types of tunnelling conditions considered
                    //     bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                    //     bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                    //     bool contact_to_contact = (metal1p && metal2p);

                    //     double local_E_drop = atom_CB_edge[i - 2] - atom_CB_edge[j - 2];                // [eV] difference in energy between the two atoms

                    //     // compute the WKB tunneling coefficients for all the tunnelling conditions
                    //     if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                    //     {
                                
                    //         double prefac = -(sqrt( 2 * m_e ) / h_bar) * (2.0 / 3.0);           // [s/(kg^1/2 * m^2)] coefficient inside the exponential
                    //         double dist = (1e-10)*dist_angstrom;                                // [m] 3D distance between atoms i and j

                    //         if (contact_to_trap)
                    //         {
                    //             double energy_window = fabs(local_E_drop);                      // [eV] energy window for tunneling from the contacts
                    //             double dV = 0.01;                                               // [V] energy spacing for numerical integration
                    //             // double dE = eV_to_J * dV;                                       // [eV] energy spacing for numerical integration
                    //             double dE = eV_to_J * dV * 100000; // NOTE: @Manasa this is a temporary fix to avoid MPI issues!

                    //             // integrate over all the occupied energy levels in the contact
                    //             double T = 0.0;
                    //             for (double iv = 0; iv < energy_window; iv += dE)
                    //             {
                    //                 double E1 = eV_to_J * V0 + iv;                                  // [J] Energy distance to CB before tunnelling
                    //                 double E2 = E1 - fabs(local_E_drop);                            // [J] Energy distance to CB after tunnelling

                    //                 if (E2 > 0)                                                     // trapezoidal potential barrier (low field)                 
                    //                 {                                                           
                    //                     T += exp(prefac * (dist / fabs(local_E_drop)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                    //                 }

                    //             //     if (E2 < 0)                                                      // triangular potential barrier (high field)                               
                    //             //     {
                    //             //         T += exp(prefac * (dist / fabs(local_E_drop)) * ( pow(E1, 1.5) )); 
                    //             //     } 
                    //             }
                    //             data_d[jd] = -T;
                    //         } 
                    //         else 
                    //         {
                    //             double E1 = eV_to_J * V0;                                        // [J] Energy distance to CB before tunnelling
                    //             double E2 = E1 - fabs(local_E_drop);                             // [J] Energy distance to CB after tunnelling
                                        
                    //             if (E2 > 0)                                                      // trapezoidal potential barrier (low field)
                    //             {                                                           
                    //                 double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                    //                 data_d[jd] = -T;
                    //             }

                    //             if (E2 < 0)                                                        // triangular potential barrier (high field)
                    //             {
                    //                 double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) ));
                    //                 data_d[jd] = -T;
                    //             }
                    //         }
                    //     }
                    // }


                    // direct terms
                    if ( neighbor )
                    {
                        // contacts
                        bool metal1 = is_in_array_gpu<ELEMENT>(metals, element[i - 2], num_metals);
                        bool metal2 = is_in_array_gpu<ELEMENT>(metals, element[j - 2], num_metals);

                        // conductive vacancy sites
                        bool vacancy1 = (element[i - 2] == VACANCY);
                        bool vacancy2 = (element[j - 2] == VACANCY);
                        
                        if ((metal1 && metal2) || (vacancy1 && vacancy2) || (metal1 && vacancy2) || (vacancy1 && metal2))
                        {
                            data_d[jd] = -high_G;
                        }
                        else
                        {
                            data_d[jd] = -low_G;
                        }
                    }
                }
            }
        }
    }
}

// assemble the data for the X matrix - 1D distribution over rows
__global__ void populate_T_dist_tunnel(const double *posx_d, const double *posy_d, const double *posz_d,
                                const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                                double nn_dist, const double tol,
                                const double high_G, const double low_G, const double loop_G, 
                                const double Vd, const double m_e, const double V0,
                                int num_source_inj, int num_ground_ext, const int num_layers_contact,
                                int num_metals, int matrix_size, int num_atoms_reservoir,
                                int *col_indices_d, int *row_ptr_d, double *data_d,
                                int size_i, int size_j, int start_i, int start_j)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Nsub = matrix_size;
    int N_atom = matrix_size - 1;
    int N_full = matrix_size + 1;

    for(int id = idx; id < size_i; id += blockDim.x * gridDim.x){
        for( int jd = row_ptr_d[id]; jd < row_ptr_d[id+1]; jd++ )
        {
            int i = start_i + id;
            int j = start_j + col_indices_d[jd];

            // inner matrix terms
            if (i >= 2)
            {
                // off-diagonal inner matrix elements
                if ( (j >= 2) && (j != i)) 
                {
                    double dist_angstrom = site_dist_gpu(posx_d[i - 2], posy_d[i - 2], posz_d[i - 2],
                                                         posx_d[j - 2], posy_d[j - 2], posz_d[j - 2]);                                       
                        
                    bool neighbor = (dist_angstrom < nn_dist);                                                      

                    // non-neighbor connections
                    if (!neighbor)
                    {
                        bool any_vacancy1 = element[i - 2] == VACANCY;
                        bool any_vacancy2 = element[j - 2] == VACANCY;

                        // contacts, excluding the last layer 
                        bool metal1p = is_in_array_gpu(metals, element[i-2], num_metals) 
                                                        && (i-2 > ((num_layers_contact - 1)*num_source_inj))
                                                        && (i-2 < (N_full - num_atoms_reservoir - (num_layers_contact - 1)*num_ground_ext)); 

                        bool metal2p = is_in_array_gpu(metals, element[j-2], num_metals)
                                                        && (j-2 > ((num_layers_contact - 1)*num_source_inj))
                                                        && (j-2 < (N_full - num_atoms_reservoir - (num_layers_contact - 1)*num_ground_ext));  

                        // types of tunnelling conditions considered
                        bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                        bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                        bool contact_to_contact = (metal1p && metal2p);

                        double local_E_drop = atom_CB_edge[i - 2] - atom_CB_edge[j - 2];                // [eV] difference in energy between the two atoms

                        // compute the WKB tunneling coefficients for all the tunnelling conditions
                        if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                        {
                                
                            double prefac = -(sqrt( 2 * m_e ) / h_bar) * (2.0 / 3.0);           // [s/(kg^1/2 * m^2)] coefficient inside the exponential
                            double dist = (1e-10)*dist_angstrom;                                // [m] 3D distance between atoms i and j

                            if (contact_to_trap)
                            {
                                double energy_window = fabs(local_E_drop);                      // [eV] energy window for tunneling from the contacts
                                double dV = 0.01;                                               // [V] energy spacing for numerical integration
                                double dE = eV_to_J * dV;                                       // [eV] energy spacing for numerical integration
                                // double dE = eV_to_J * dV * 100000; // NOTE: @Manasa this is a temporary fix to avoid MPI issues!

                                // integrate over all the occupied energy levels in the contact
                                double T = 0.0;
                                for (double iv = 0; iv < energy_window; iv += dE)
                                {
                                    double E1 = eV_to_J * V0 + iv;                                  // [J] Energy distance to CB before tunnelling
                                    double E2 = E1 - fabs(local_E_drop);                            // [J] Energy distance to CB after tunnelling

                                    if (E2 > 0)                                                     // trapezoidal potential barrier (low field)                 
                                    {                                                           
                                        T += exp(prefac * (dist / fabs(local_E_drop)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                                    }

                                    if (E2 < 0)                                                      // triangular potential barrier (high field)                               
                                    {
                                        T += exp(prefac * (dist / fabs(local_E_drop)) * ( pow(E1, 1.5) )); 
                                    } 
                                }
                                data_d[jd] = -T;
                            } 
                            else 
                            {
                                double E1 = eV_to_J * V0;                                        // [J] Energy distance to CB before tunnelling
                                double E2 = E1 - fabs(local_E_drop);                             // [J] Energy distance to CB after tunnelling
                                        
                                if (E2 > 0)                                                      // trapezoidal potential barrier (low field)
                                {                                                           
                                    double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                                    data_d[jd] = -T;
                                }

                                if (E2 < 0)                                                        // triangular potential barrier (high field)
                                {
                                    double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) ));
                                    data_d[jd] = -T;
                                }
                            }
                        }
                    }

                
                }
            }
        }
    }
}

void Assemble_T(GPUBuffers &gpubuf, const double nn_dist, const double tol, const double high_G, const double low_G, const double loop_G, 
                const double Vd, const double m_e, const double V0, int num_source_inj, int num_ground_ext, int num_layers_contact, int num_metals, int Nsub, int num_atoms_reservoir)
{
    // Distributed_matrix *T_distributed = gpubuf.T_distributed;
    int rows_this_rank = gpubuf.T_distributed->rows_this_rank;
    int disp_this_rank = gpubuf.T_distributed->displacements[gpubuf.T_distributed->rank];

    int threads = 1024;
    int blocks = (gpubuf.T_distributed->rows_this_rank + threads - 1) / threads;   

    for(int i = 0; i < gpubuf.T_distributed->number_of_neighbours; i++){

        // std::cout << "rank " << gpubuf.rank << " T_distributed->nnz_per_neighbour[" << i << "] = " << T_distributed->nnz_per_neighbour[i] << std::endl;

        int rows_neighbour = gpubuf.T_distributed->counts[gpubuf.T_distributed->neighbours[i]];
        int disp_neighbour = gpubuf.T_distributed->displacements[gpubuf.T_distributed->neighbours[i]];

        //check if this memset is needed
        gpuErrchk(cudaMemset(gpubuf.T_distributed->data_d[i], 0,
                             gpubuf.T_distributed->nnz_per_neighbour[i] * sizeof(double)) );

        // the T matrix has the additional terms coming from the last column!
        populate_T_dist_neighbor<<<blocks, threads>>>(gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
                                                    gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_charge, gpubuf.atom_CB_edge,
                                                    nn_dist, tol, high_G, low_G, loop_G,
                                                    Vd, m_e, V0,
                                                    num_source_inj, num_ground_ext, num_layers_contact, num_metals, Nsub, num_atoms_reservoir,
                                                    gpubuf.T_distributed->col_indices_d[i],
                                                    gpubuf.T_distributed->row_ptr_d[i],
                                                    gpubuf.T_distributed->data_d[i], 
                                                    rows_this_rank,
                                                    rows_neighbour,
                                                    disp_this_rank,
                                                    disp_neighbour);

        populate_T_dist_tunnel<<<blocks, threads>>>(gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
                                                gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_charge, gpubuf.atom_CB_edge,
                                                nn_dist, tol, high_G, low_G, loop_G,
                                                Vd, m_e, V0,
                                                num_source_inj, num_ground_ext, num_layers_contact, num_metals, Nsub, num_atoms_reservoir,
                                                gpubuf.T_distributed->col_indices_d[i],
                                                gpubuf.T_distributed->row_ptr_d[i],
                                                gpubuf.T_distributed->data_d[i], 
                                                rows_this_rank,
                                                rows_neighbour,
                                                disp_this_rank,
                                                disp_neighbour);

        gpuErrchk( cudaPeekAtLastError() );
    }

}


__global__ void calc_diagonal_T( 
    int *col_indices,
    int *row_ptr,
    double *data,
    double *diag,
    int matrix_size,
    int this_ranks_block
)
{   // double check data memset
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){ 
        //reduce the elements in the row
        double tmp = 0.0;
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if (i != col_indices[j] || this_ranks_block != 0) // check for multiple nodes
            {
                tmp += data[j];
            }
        }
        diag[i] += -tmp;
    }
}

__global__ void insert_diag_T( 
    int *col_indices,
    int *row_ptr,
    double *data,
    double *diag,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){ 

        // write the sum of the off-diagonals onto the existing diagonal element
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i == col_indices[j]){
                data[j] += diag[i];
                diag[i] = data[j];
            }
        }
    }
}

// reduces the rows of the input sparse matrix into the diagonals, and collects the resulting diagonal vector to be used for preconditioning
void update_diagonal_sparse(GPUBuffers &gpubuf, double *diagonal_local_d)
{
    Distributed_matrix *T_distributed = gpubuf.T_distributed;

    int threads = 1024;
    int blocks = (T_distributed->rows_this_rank + threads - 1) / threads;   

    for(int i = 0; i < T_distributed->number_of_neighbours; i++){
        calc_diagonal_T<<<blocks, threads>>>(T_distributed->col_indices_d[i], 
                                            T_distributed->row_ptr_d[i],
                                            T_distributed->data_d[i], diagonal_local_d, T_distributed->rows_this_rank, 
                                            i);
    }

    // each rank sets its own diagonal (do not set it, the populate-kernel updated the diagonal elements with the last column already)
    insert_diag_T<<<blocks, threads>>>(T_distributed->col_indices_d[0], 
                                    T_distributed->row_ptr_d[0],
                                    T_distributed->data_d[0], diagonal_local_d, T_distributed->rows_this_rank);

    gpuErrchk( cudaPeekAtLastError() );
}

void update_diag_ineg(GPUBuffers &gpubuf, double **ineg_data_d)
{
    Distributed_matrix *T_distributed = gpubuf.T_distributed;
    double *diagonal_local_d;
    gpuErrchk( cudaMalloc((void **)&diagonal_local_d, T_distributed->rows_this_rank* sizeof(double)) );
    gpuErrchk( cudaMemset(diagonal_local_d, 0, T_distributed->rows_this_rank * sizeof(double)) );

    int threads = 1024;
    int blocks = (T_distributed->rows_this_rank + threads - 1) / threads;   

    for(int i = 0; i < T_distributed->number_of_neighbours; i++){
        calc_diagonal_T<<<blocks, threads>>>(T_distributed->col_indices_d[i], 
                                             T_distributed->row_ptr_d[i],
                                             ineg_data_d[i], diagonal_local_d, T_distributed->rows_this_rank, 
                                             i);
    }

    // each rank sets its own diagonal (do not set it, the populate-kernel updated the diagonal elements with the last column already)
    insert_diag_T<<<blocks, threads>>>(T_distributed->col_indices_d[0], 
                                       T_distributed->row_ptr_d[0],
                                       ineg_data_d[0], diagonal_local_d, T_distributed->rows_this_rank);

    gpuErrchk( cudaPeekAtLastError() );
    cudaFree(diagonal_local_d);
}