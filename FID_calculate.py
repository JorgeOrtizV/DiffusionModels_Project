import os
import subprocess
import json
import sys

sample_images_path = "ldm_results_cold_diff/sampled_images"
FIDS = {}
dirs = os.listdir(sample_images_path)
for dir_name in dirs:
    sample_path = os.path.join(sample_images_path, dir_name)
    if os.path.isdir(sample_path):
        print(f"Cal[culating FID for {dir_name}...")
        command = [sys.executable, "-m", "pytorch_fid", sample_path, "test_samples"]
        print(command)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"FID calculation for {dir_name} completed successfully.")
            FID = result.stdout.strip()
            if FID:
                print(f"FID for {dir_name}: {FID}")
                FIDS[dir_name] = FID
            else:
                print(f"No FID value returned for {dir_name}.")
        else:
            print(f"Error calculating FID for {dir_name}.")

            print("ERror: ", result.stderr.strip())


# Store FIDS
fid_file_path = "FID_results_LDM_intermediate.json"
with open(fid_file_path, "w") as fid_file:
    json.dump(FIDS, fid_file, indent=4)
