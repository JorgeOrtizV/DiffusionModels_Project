import subprocess
import os

script = "ldm_experiment.py"

steps_list = [20, 50, 100, 250, 500, 1000]
latent_channels_list = [1]
blur_sigma = [0.1, 0.2, 0.3, 0.4, 0.5]

for steps in steps_list:
    for latent_channels in latent_channels_list:
        for sigma in blur_sigma:
            print("############################################")
            print(
                f"Running with steps: {steps}, latent_channels: {latent_channels}, blur_sigma: {sigma} --cold_diff"
            )
            print("############################################")
            subprocess.run(
                [
                    "python",
                    script,
                    "--steps",
                    str(steps),
                    "--lc",
                    str(latent_channels),
                    "--blur_sigma",
                    str(sigma),
                    "--cold_diff",
                    "True",
                ]
            )
