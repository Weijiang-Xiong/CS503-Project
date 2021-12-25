""" usage: modify your account first line 118
    conda activate habitat
    python submit_job.py --feature_set <Number>
"""

import os 
import glob 
import argparse
import subprocess

from itertools import product
from pathlib import Path

def format_list(str_list):
    """ convert a list of strings to a string with square brackets "[\"something\", \"something\"]"
        which is suitable for subprocess.run
    """
    list_repr = ['\"{}\"'.format(item).replace("\"", "\\\"") for item in str_list]
    joint_str = ",".join(list_repr)
    return joint_str

repr_short_map = {"keypoints3d":"kp3d", 
                  "normal":"norm", 
                  "denoising":"deno", 
                  "edge_texture":"edge", 
                  "curvature":"curv"}

corruption_short_map = {"brightness":"B", 
                        "defocus_blur":"D", 
                        "motion_blur":"M", 
                        "spatter":"S", 
                        "speckle_noise":"N"}

def form_complete_command(basedir, run_type, encoder_type, representations, corruptions, severity, repeat, idx):
    # python_script = "srun -n 1 --output out/slurm-%A_{}_%a.log python habitat_baselines/run.py".format(idx)
    python_script = "python -u habitat_baselines/run.py"
    yaml_file = "habitat_baselines/config/pointnav/ppo_pointnav_example.yaml"

    ckpt_dir = Path("{}/data/ckpt".format(basedir))
    training_types = sorted([folder.name for folder in ckpt_dir.iterdir() if folder.is_dir()])
    common_part = "{} --exp-config {} --run-type {}".format(python_script, yaml_file, run_type)
    
    encoder_part = " RL.POLICY.NET_CONF.encoder {}".format(encoder_type)

    if encoder_type == "MidLevelEncoder":
        feature_str = format_list(representations)
        repr_part = "RL.POLICY.NET_CONF.representations \"[{}]\"".format(feature_str)
        encoder_part = encoder_part + " " + repr_part 

    save_name = "mid" if encoder_type == "MidLevelEncoder" else "baseline"
    if save_name == "mid":
        short_names = [repr_short_map[feat] for feat in representations]
        save_name = save_name + "_" +"_".join(short_names)
    if run_type == "eval": # make sure the checkpoint is already there 
        assert save_name in training_types 
    ckpt_part = "EVAL_CKPT_PATH_DIR data/ckpt/{} CHECKPOINT_FOLDER data/ckpt/{}".format(save_name, save_name)
    
    if len(corruptions) == 0:
        save_name += "_None"
    else:
        short_names = [corruption_short_map[cor] for cor in corruptions]
        save_name = save_name + "_cor_" +"_".join(short_names)

    video_part = "VIDEO_DIR video/{}_{}".format(save_name, repeat)

    if len(corruptions) > 0:
        cor_str = format_list(corruptions)
        corruption_part = "RL.POLICY.NET_CONF.corruption \"[{}]\"".format(cor_str)
        corruption_part += " RL.POLICY.NET_CONF.severity {}".format(severity)
    else:
        corruption_part = ""
    all_parts = [common_part, ckpt_part, encoder_part, video_part, corruption_part]
    complete_command = " ".join(all_parts)
    complete_command = complete_command + " &" # run in background 
    return complete_command

if __name__ == "__main__":
    
    parser  = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./", help="directory to habitat-lab")
    parser.add_argument("--run_type", type=str, default="eval", help="run type, train or eval")
    parser.add_argument("--feature_set", type=int, default=0, choices=[0,1,2,3,4,5], help="which set of feature to use, from 0 to 5")
    parser.add_argument("--repeat", type=int, default=1, help="how many times to repeat a single experiment")
    args = parser.parse_args() # 
    
    basedir = args.base_dir
    run_type = args.run_type
    
    repeats = list(range(args.repeat))
    
    encoder_type = "MidLevelEncoder"
    representations = [["keypoints3d"], 
                       ["keypoints3d", "normal"], 
                       ["keypoints3d", "normal", "curvature"],
                       ["keypoints3d", "denoising"],
                       ["keypoints3d", "edge_texture"]]
    enc_repr = list(product([encoder_type], representations)) 
    enc_repr.insert(0, ("ResNetEncoder", [])) # raw image baseline + midlevel representations 
    chosen_repr = [enc_repr[args.feature_set]]
    # the first empty list means no corruption
    corruptions = [[], ["brightness"], ["defocus_blur"], ["motion_blur"], ["spatter"], ["speckle_noise"]] 
    # corruptions = [["motion_blur"], ["speckle_noise"], ["motion_blur", "speckle_noise"]]
    # corruptions = [["motion_blur", "speckle_noise"]]
    severity = 1
    
    combinations = list(product(chosen_repr, corruptions, repeats))
    # sort according to the order in corruptions
    combinations = sorted(combinations, key=lambda item: corruptions.index(item[1])) 
    
    all_tasks = []
    for idx, ((encoder, repre), corr, num) in enumerate(combinations):
        complete_command = form_complete_command(basedir, run_type, encoder, repre, corr, severity, num, idx)
        all_tasks.append(complete_command)
        
    
    sbatch_job_specification = """#!/bin/bash
#SBATCH --job-name RUN_TITAN
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time 72:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --output "out/slurm-%A_%a.log"
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

module load gcc/8.4.0-cuda cuda/10.2.89 \n
"""
    all_tasks_str = "\n\n".join(all_tasks)
    suffix = "wait"
    
    file_content = sbatch_job_specification + all_tasks_str + "\n\n" + suffix
    file_name = "temp_{}.sh".format(args.feature_set)
    with open(file_name, "w") as f:
        f.write(file_content)
    
    command = "sbatch {}".format(file_name)
    subprocess.run(command, shell=True)
    os.remove(file_name)
        