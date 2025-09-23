from cbed_simulation import contrast_stretch, atomic_potential, parse_xyz, overall_wrapper
from jax import numpy as jnp
import matplotlib.pyplot as plt

import numpy as np

import h5py
import numpy as np
import sys
from tqdm import tqdm
import time
def save_cbed_data(h5_filename, entries):
    """
    Save multiple CBED data entries to an HDF5 file.

    Parameters:
    - h5_filename: str, path to the output h5 file.
    - entries: list of dicts, each dict contains keys:
        - "cbed": np.ndarray (128x128)
        - "filename": str
        - "zone_axis": np.ndarray
        - "position": np.ndarray
        - "rotation": float
        - "crop": list or tuple or dict (any serializable info about cropping)
        - "scale": float
    """

    with h5py.File(h5_filename, "a") as f:
        for i, entry in enumerate(entries):
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            group = f.create_group(f"{current_time}_{i}")

            # Store cbed array
            group.create_dataset("cbed", data=entry["cbed"], dtype='float16')

            # Store strings as attributes (h5py needs strings encoded or use attrs)
            group.attrs["filename"] = entry["filename"]

            # Store arrays as datasets
            group.create_dataset("zone_axis", data=entry["zone_axis"])
            group.create_dataset("position", data=entry["position"])

            # Store scalar floats as attributes
            group.attrs["rotation"] = entry["rotation"]
            group.attrs["scale"] = entry["scale"]

            # Store crop info - if it's array-like, save as dataset; if complex, save as attribute string
            if 'crop' in entry:
                crop = entry["crop"]
                if isinstance(crop, (list, tuple, np.ndarray)):
                    group.create_dataset("crop", data=crop)
                else:
                    # Convert to string if needed
                    group.attrs["crop"] = str(crop)

if __name__== "__main__":
    pixel_size = 0.1 #Angstroms
    element_IDS = set([33,82])
    kirkland_potentials = {}
    for element_ID in element_IDS:
        potential = atomic_potential(
            atom_no=int(element_ID),
            pixel_size=pixel_size,
            sampling=16,
            potential_extent=4,
            datafile="/home/mnt/zwx/Kirkland_Potentials.npy",
        )
        kirkland_potentials[int(element_ID)] = potential


    kirkland_jax = jnp.asarray([kirkland_potentials[int(el)] for el in element_IDS])
    #for ei, element_ID in enumerate(element_IDS):
        #atoms[atoms[:,0] == element_ID, 0] = ei
    h5_filename = sys.argv[1] if len(sys.argv) > 1 else '/home/mnt/zwx/test.h5'
    filenames  = [f'/home/mnt/zwx/New_Bi2Se3_layered/ADHK_Bi2Se3.{i}' for i in range(0, 2600, 4)][:1]
    zone_axes = [jnp.array([0,0,1]), jnp.array([0,1,2]), jnp.array([1, 0, 2]), jnp.array([1, 1, 2]), jnp.array([0, 1, 3]), jnp.array([1, 0, 3]), jnp.array([1, 1, 3]), jnp.array([0, 1, 4]), jnp.array([1, 0, 4]), jnp.array([1, 1, 4])][:1]
    positions = jnp.array([[0,0]]) # these are all in Angstroms, assuming center of the potential slice is at (0,0). if you don't specify poss, it will default to [[0,0]]. (single Cbed at the center)

    for fi, filename in tqdm(enumerate(filenames)):
        entries = []
        atoms, metadata, element = parse_xyz(filename, element_specified=False)
        atoms[atoms[:, 0] == 1, 0] = 82 # Bi
        atoms[atoms[:, 0] == 2, 0] = 33 # Se
        for zone_axis in zone_axes:
            for pos in positions:
                for rot in np.random.uniform(0, 2 * jnp.pi, 2):  # Random rotation

                    cbed_patterns, slices, test_rotated_coords, test_rotated_cells = overall_wrapper(atoms, metadata, zone_hkl = zone_axis, theta = rot, pixel_size = 0.1, kirkland_jax=kirkland_jax, poss = jnp.array([pos]), threshold_A = 5)
                    cbed = np.array(cbed_patterns.data_array[0]) # Assuming you want the first pattern
                    cbed_center = np.array([cbed.shape[0] // 2, cbed.shape[1] // 2])
                    crop_shift = np.random.randint(-5, 5, size=2)  # Random crop shiftm   
                    crop_size = 256
                    cbed_cropped = cbed[cbed_center[0]-crop_size//2 + crop_shift[0]:cbed_center[0]+crop_size//2 + crop_shift[0],
                                        cbed_center[1]-crop_size//2 + crop_shift[1]:cbed_center[1]+crop_size//2 + crop_shift[1]]
                    scale = np.array([cbed_patterns.calib_x, cbed_patterns.calib_y])
                    entry = {
                        "cbed": cbed_cropped.astype(np.float16),
                        "filename": filename,
                        "zone_axis": np.array(zone_axis),
                        "position": np.array(pos),
                        "rotation": rot,  # Store only one rotation angle for simplicity
                        "crop": [int(crop_shift[0]), int(crop_shift[1]), crop_size, crop_size],  # example crop box (x_start, y_start, width, height)
                        "scale": scale
                    }
                    entries.append(entry)

        save_cbed_data(h5_filename, entries)
