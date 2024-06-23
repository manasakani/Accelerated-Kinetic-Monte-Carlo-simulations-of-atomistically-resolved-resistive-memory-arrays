import sys
import os
import numpy as np
from scipy.sparse import csr_matrix, csgraph
from scipy.sparse.csgraph import reverse_cuthill_mckee
from itertools import chain
import matplotlib.pyplot as plt

def compute_bandwidth(csr_matrix):
    # Extracting the CSR matrix components
    data = csr_matrix.data
    indices = csr_matrix.indices
    indptr = csr_matrix.indptr

    # Initialize maximum bandwidth to 0
    max_bandwidth = 0

    # Iterate over each row
    for row in range(len(indptr) - 1):
        # Get the start and end indices of the row
        start = indptr[row]
        end = indptr[row + 1]

        # Iterate over the non-zero elements in the row
        for i in range(start, end):
            # Get the column index of the non-zero element
            col = indices[i]

            # Calculate the distance to the diagonal
            distance = col - row

            # Update the maximum bandwidth
            max_bandwidth = max(max_bandwidth, abs(distance))

    return max_bandwidth

def look_at_sparsity(csr_matrix_data, folder_path, step, reorder=False):
    if reorder:
        # Reorder using reverse Cuthill-McKee
        perm = reverse_cuthill_mckee(csr_matrix_data)
        csr_matrix_data = csr_matrix_data[perm, :][:, perm]

    plt.spy(csr_matrix_data, markersize=1)

    filename = os.path.join(folder_path, f"sparsity_pattern_step#{step}")
    if reorder:
        filename += "_reordered"
    filename += ".jpg"
    plt.savefig(filename, dpi=300)


# Read xyz coordinates from a file:
def read_xyz(file_path):
    atoms = []
    coordinates = []

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into words
            words = line.split()

            # Check if the line has at least 4 words (atom and x, y, z coordinates)
            if len(words) >= 4:
                # The first word is the atom
                atom = words[0]

                # The remaining words are x, y, z coordinates
                x, y, z = map(float, words[1:4])

                # Append the atom and coordinates to the respective lists
                atoms.append(atom)
                coordinates.append([x, y, z])

    # Convert the coordinates list to a NumPy array
    coordinates_array = np.array(coordinates)

    return atoms, coordinates, coordinates_array.shape[0]

def read_csr_matrix(folder_path, step):
    filename_values = os.path.join(folder_path, f"csrValues_step#{step}.txt")
    filename_row = os.path.join(folder_path, f"csrRowPtr_step#{step}.txt")
    filename_col = os.path.join(folder_path, f"csrColIndices_step#{step}.txt")

    values = np.loadtxt(filename_values)
    row_ptr = np.loadtxt(filename_row, dtype=int)
    col_indices = np.loadtxt(filename_col, dtype=int)
    
    m = len(row_ptr) - 1
    nnz = len(values)
    csr_matrix_data = csr_matrix((values, col_indices, row_ptr), shape=(m, m))
    return csr_matrix_data

def reorder_xyz(atoms, coordinates, perm):

    # Ensure perm is a 1D numpy array
    perm = np.asarray(perm)
    if perm.ndim != 1:
        raise ValueError("Permutation array must be 1D")

    if isinstance(coordinates, list):
        coordinates = np.array(coordinates)

    # Check the shape of the coordinates array
    if coordinates.shape[0] != len(perm):
        raise ValueError("Number of rows in coordinates must match the length of permutation array")

    # Reorder the atoms and coordinates based on the permutation
    reordered_atoms = [atoms[i] for i in perm]
    reordered_coordinates = coordinates[perm]

    return reordered_atoms, reordered_coordinates

def main():
    # if len(sys.argv) != 2:
    #     print("Usage: python script.py <name of folder with csr txt files>")
    #     sys.exit(1)

    folder_path = './csr_matrix/'

    csr_matrix_data = read_csr_matrix(folder_path, 0)

    # remove the last three rows and columns of the matrix:
    # csr_matrix_data = csr_matrix_data[:-3, :-3]

    plt.spy(csr_matrix_data, markersize=1)
    plt.savefig("sparsity_before.png", dpi=300)
    plt.close()

    print("bandwidth before:")
    bandwidth = compute_bandwidth(csr_matrix_data)
    print(bandwidth)

    # Reorder using reverse Cuthill-McKee
    perm = reverse_cuthill_mckee(csr_matrix_data, symmetric_mode=False)                     # cuthill mckee
    csr_matrix_data_reordered = csr_matrix_data[perm, :][:, perm]
    plt.spy(csr_matrix_data_reordered, markersize=1)
    plt.savefig("sparsity_after.png", dpi=300)

    print("bandwidth after:")
    bandwidth = compute_bandwidth(csr_matrix_data_reordered)
    print(bandwidth)

    # print("bandwidth after:")
    # bandwidth = compute_bandwidth(csr_matrix_data_reordered)
    # print(bandwidth)
    # look_at_sparsity(csr_matrix_data, folder_path, 1)

    matrix_size = csr_matrix_data.shape[0]
    num_nonzeros = csr_matrix_data.nnz
    print(f"Matrix size: {matrix_size}")
    print(f"Number of non-zeros: {num_nonzeros}")

    # Read xyz coordinates
    file_path = "reordered_device_20.xyz"
    atoms, coordinates, num_atoms = read_xyz(file_path)

    # sort the atoms and coordinates by their x-axis values in place
    # atoms, coordinates = zip(*sorted(zip(atoms, coordinates), key=lambda x: x[1][0]))

    num_boundary = 9216
    new_num_atoms = num_atoms - 2*num_boundary 

    # Create a list of tuples containing atom, coordinates, and index in perm
    # atom_coord_perm = [(atoms[i], coordinates[i], perm[i-num_boundary]) for i in range(num_boundary, num_atoms - num_boundary)]
    print(np.shape(perm))
    print(np.shape(atoms[num_boundary:-num_boundary]))
    print("new num atoms ", new_num_atoms)
    reordered_atoms, reordered_coordinates = reorder_xyz(atoms[num_boundary:-num_boundary], coordinates[num_boundary:-num_boundary], perm)

    # Sort the list based on the values of perm
    # sorted_atom_coord_perm = sorted(atom_coord_perm, key=lambda x: x[2])

    # Write out modified xyz file with reordering index
    with open(os.path.join(folder_path, "bwmin_device_20x5.xyz"), 'w') as file:
        file.write(f"{len(reordered_atoms)+ 2*num_boundary}\n")
        file.write("Atoms. Positions (Angstroms)\n")
        # i = 0
        for i, (atom, coord) in enumerate(zip(atoms, coordinates)):
            if i < num_boundary:
                x, y, z = coord
                file.write(f"{atom} {x} {y} {z}\n")
            else:
                break
        
        # i = num_boundary
        for i, (atom, coord) in enumerate(zip(reordered_atoms, reordered_coordinates)):
            x, y, z = coord
            file.write(f"{atom} {x} {y} {z}\n")
            # i+=1

        # i = len(atoms) - num_boundary
        for i, (atom, coord) in enumerate(zip(atoms, coordinates)):
            if i >= (len(atoms) - num_boundary):
                x, y, z = coord
                file.write(f"{atom} {x} {y} {z}\n")

    # # Write out modified xyz file with reordering index
    # with open(os.path.join(folder_path, "bwmin_device_10.xyz"), 'w') as file:
    #     file.write(f"{new_num_atoms}\n")
    #     file.write("Atoms. Positions (Angstroms)\n")
    #     for i, (atom, coord) in enumerate(zip(atoms, coordinates)):
    #         if i < num_boundary or i >= num_atoms - num_boundary:
    #             continue 
    #         else:
    #             x, y, z = coord
    #             reordered_index = perm[i-num_boundary]  
    #             file.write(f"{atom} {x} {y} {z} {reordered_index}\n")

    # Write out modified xyz file with original order
    # with open(os.path.join(folder_path, "reordered_crossbar_40_idx.xyz"), 'w') as file:
    #     file.write(f"{new_num_atoms}\n")
    #     file.write("Atoms. Positions (Angstroms)\n")
    #     for i, (atom, coord) in enumerate(zip(atoms, coordinates)):
    #         if i < num_boundary or i >= num_atoms - num_boundary:
    #             continue 
    #         else:
    #             x, y, z = coord
    #             original_order_index = i - num_boundary 
    #             file.write(f"{atom} {x} {y} {z} {original_order_index}\n")


if __name__ == "__main__":
    main()