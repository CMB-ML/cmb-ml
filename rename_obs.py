from pathlib import Path


datasets_root   = '/shared/data/Datasets/'
dataset_name    = 'I_512_1450'
working_dir_in  = 'Simulation_Working/Simulation_D_No_Noise'
working_dir_out = 'Simulation'

splits = {'Test':200, 
          'Train':1000, 
          'Validation':250}

# detectors = [30, 44, 70, 100, 143, 217, 353, 545, 857]


# in_fn_template = 'obs_{det}_map.fits'
# out_fn_template = 'sky_{det}_no_noise_map.fits'

in_cmb_template = 'cmb_map_fid.fits'
out_cmb_template = 'cmb_map_fid.fits'

for split in splits:
    for sim_num in range(splits[split]):
        in_dir = Path(datasets_root) / dataset_name / working_dir_in / split / f'sim{sim_num:04d}'
        # for det in detectors:
        #     old_fp = in_dir / in_fn_template.format(det=det)
        #     if not old_fp.exists():
        #         print(f"{old_fp} does not exist! Aborting.")
        #         exit()
        old_fp = in_dir / in_cmb_template
        if not old_fp.exists():
            print(f"{old_fp} does not exist! Aborting.")
            exit()


for split in splits:
    for sim_num in range(splits[split]):
        in_dir = Path(datasets_root) / dataset_name / working_dir_in / split / f'sim{sim_num:04d}'
        out_dir = Path(datasets_root) / dataset_name / working_dir_out / split / f'sim{sim_num:04d}'
        # for det in detectors:
        #     old_fp = in_dir / in_fn_template.format(det=det)
        #     new_fp = out_dir / out_fn_template.format(det=det)
        #     new_fp.parent.mkdir(parents=True, exist_ok=True)
        #     print(f"Renaming {old_fp} to {new_fp}")
        #     old_fp.rename(new_fp)
        old_fp = in_dir / in_cmb_template
        new_fp = out_dir / out_cmb_template
        new_fp.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying {old_fp} to {new_fp}")
        old_fp.rename(new_fp)
print("Done renaming files.")
