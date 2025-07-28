import subprocess
import os

script = "experiment.py"

steps_list = [20, 50, 250, 500, 1000]
blur_list = [0]

for steps in steps_list:
    for blur_sigma in blur_list:
        print("############################################")
        print(f"Running with steps: {steps}, blur_sigma: {blur_sigma}")
        print("############################################")
        subprocess.run(
            ["python", script, "--steps", str(steps), "--blur_sigma", str(blur_sigma)]
        )
