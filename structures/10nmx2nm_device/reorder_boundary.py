# This script contains some functionality to re-order the xyz file such that the boundary atoms
# (to which the external potential is applied) are correct at the start and end of the xyz file

import numpy as np

# Write XYZ coordinates to a file
def write_xyz(file_path, atoms, coordinates):
    with open(file_path, 'a') as xyz_file:
        for atom, coord in zip(atoms, coordinates):
            line = "{} {:.6f} {:.6f} {:.6f}\n".format(atom, coord[0], coord[1], coord[2])
            xyz_file.write(line)


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

    return atoms, coordinates


file_path = "device_10_2thick_wrapped.xyz"
file_path_interstitials = "interstitials_device_10_2thick.xyz"

atoms, coordinates = read_xyz(file_path)
print("# Atoms:", len(atoms))
print("Coordinates:", np.shape(coordinates))

# *** Extract all the coordinates and atoms ***

# the number of contact atoms is determined by the left contact
num_contact = 0
left_contact_coords = []
left_contact_atoms = []
for atom, coord in zip(atoms, coordinates):
    if atom in ['Ti', 'N']:
        left_contact_coords.append(coord)
        left_contact_atoms.append(atom)
        num_contact += 1
    if atom in ['Hf', 'O']:
        break

counter = 0
right_contact_coords = []
right_contact_atoms = []
for atom, coord in zip(reversed(atoms), reversed(coordinates)):
    if atom in ['Ti', 'N']:
        right_contact_coords.append(coord)
        right_contact_atoms.append(atom)
        counter += 1
    if atom in ['Hf', 'O'] or counter == num_contact:
        break

right_contact_coords.reverse()
right_contact_atoms.reverse()

print("Num contact: ", num_contact)

# Get the oxide atoms:
oxide = []
for i in range(num_contact, len(atoms)-num_contact):
    oxide.append([atoms[i], coordinates[i]])

# Use this to check how/where the bit/wordlines are split:
# print("Left contact coordinates:", np.shape(left_contact_coords))
# print("x values:", np.unique( [coord[0] for coord in left_contact_coords] ))
# print("y values:", np.sort(np.unique( [coord[1] for coord in left_contact_coords] )))
# print("z values:", np.sort(np.unique( [coord[2] for coord in left_contact_coords] )))

# *** Extract the contacts ***

right_contact = []
for atom, coord in zip(right_contact_atoms, right_contact_coords):
    right_contact.append([atom, coord])
    
print("right_contact #atoms: ", len(right_contact))


left_contact = []
for atom, coord in zip(left_contact_atoms, left_contact_coords):
    left_contact.append([atom, coord])
    
print("left_contact #atoms: ", len(left_contact))

# *** Extract the interstitials:
left_contact_start = 33.727300 # exclude the interstitials which are in the contact
interstitials = []
interstitial_atoms, interstitial_coords = read_xyz(file_path_interstitials)
for atom, coord in zip(interstitial_atoms, interstitial_coords):
    if coord[0] < left_contact_start:
        interstitials.append([atom, coord])


device = []
device.append(left_contact)
device.append(oxide)
device.append(interstitials)
device.append(right_contact)
device = [item for sublist in device for item in sublist]

first_144 = device[:144]
last_144 = device[-144:]

# Sort the elements by x-coordinate while excluding the first and last 144 elements
print("sorting...")
sorted_device = sorted(device[144:-144], key=lambda x: x[1][0])
print("sorted")


# Re-build the xyz file:
xyz_file_path = "reordered_device_10_2thick.xyz"

print("writing to xyz file...")
with open(xyz_file_path, 'w') as xyz_file:
    # Write total number of atoms to the file
    total_atoms = len(device)
    xyz_file.write(str(total_atoms) + '\n\n')
    
    write_xyz(xyz_file_path, *zip(*first_144))
    write_xyz(xyz_file_path, *zip(*sorted_device))
    write_xyz(xyz_file_path, *zip(*last_144))