#put everything together
from cbed_simulation import contrast_stretch, atomic_potential, parse_xyz, overall_wrapper
from jax import numpy as jnp
import matplotlib.pyplot as plt
import cv2
from model_training import save_cbed_data
import itertools
import numpy as np
import gc
import tqdm
import jax
import jax.profiler

all_possible_zones = [[0,0,1],
                      [0,1,1],
                      [1,1,1],
                      [0,1,2],
                      [1,1,2],
                      [0,1,3],
                      [0,2,3],
                      [1,1,3],
                      [1,2,3],
                      [2,2,3]]
#get all possible permutations for each zone axis
all_possible_zones_permutations = []
for zone in all_possible_zones:
    all_possible_zones_permutations += list(set(itertools.permutations(zone)))
print(len(all_possible_zones_permutations))
print(len(set(all_possible_zones_permutations)))

import glob
structures_path = '/home/mnt/zwx/mp_structures_Bi/BiSe/'
all_structures = glob.glob(structures_path + 'mp-*')
print(all_structures)

import pickle
kirkland_jax_cache = '/home/mnt/zwx/ptyrodactyl/cbed_sim_workflow/kirkland_jax_cached.pkl'
with open(kirkland_jax_cache, 'rb') as f:
    kirkland_jax= pickle.load(f)

import json
with open('atom_numbers.json', 'r') as f:
    atom_numbers = json.load(f)

from cbed_simulation import overall_wrapper_rotate_first
for structure in tqdm.tqdm(all_structures[9:]):
    h5_filename = f'{structure}/simulated_cbed_2.5mrad.h5'
    
    for zone_axis in tqdm.tqdm(all_possible_zones_permutations[:]):
        print(f'Simulating {structure} along zone axis {zone_axis}')
        entries = []
        for perturbation in [0.0, 0.25, 0.5]:  # Different levels of perturbation
            for rot in np.random.uniform(0, jnp.pi, 2):  # Random rotation
            
                atoms, metadata, element = parse_xyz(f'{structure}/xyz.xyz', element_specified=True)
                atoms[:,0] = atoms[:,0] - 1
                poss = jnp.array([[0,0]])
                # 2.5 mrad semiangle at 100 kV has a probe diameter of about 2 nm
                cbed_patterns_2, slices_2, test_rotated_coords_2, test_rotated_cells_2 = overall_wrapper_rotate_first(atoms, metadata, zone_hkl = jnp.array(zone_axis), theta = rot, pixel_size = 0.1, kirkland_jax=kirkland_jax
                                                                                                , poss = poss, threshold_A = 50, perturbation = perturbation, max_slices = 80 )
                
                calibration_length = 2.5 #A-1
                calibration_length_pixels = int(calibration_length / cbed_patterns_2.calib_x[0])
                #fig = plt.figure()
                data = cbed_patterns_2.data_array
                data_center = data.shape[-1] // 2
                cropped_data = np.array(data[0,data_center-calibration_length_pixels//2:data_center+calibration_length_pixels//2, data_center-calibration_length_pixels//2:data_center+calibration_length_pixels//2])
                cropped_data = cv2.resize(cropped_data, (300, 300), interpolation=cv2.INTER_AREA)

                entry = {
                        "cbed": cropped_data.astype(np.float16),
                        "filename": f'{structure}/xyz.xyz',
                        "zone_axis": np.array(zone_axis),
                        "position": np.array(poss[0]),
                        "rotation": rot,  # Store only one rotation angle for simplicity
                        #"crop": [int(crop_shift[0]), int(crop_shift[1]), crop_size, crop_size],  # example crop box (x_start, y_start, width, height)
                        "scale": calibration_length/cropped_data.shape[0]
                    }
                entries.append(entry)

        save_cbed_data(h5_filename, entries)
        jax.clear_caches()
        del cbed_patterns_2, slices_2, test_rotated_coords_2, test_rotated_cells_2, data
        gc.collect()
        jax.profiler.save_device_memory_profile(f"mem_{zone_axis}.prof")
        