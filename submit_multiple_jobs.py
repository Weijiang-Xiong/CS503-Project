import os 
import glob 
import argparse
import subprocess

from itertools import product
from pathlib import Path

def format_list(str_list):
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

def form_complete_command(basedir, run_type, encoder_type, representations, corruptions, severity, repeat):
    python_script = "sbatch python_wrapper.sh habitat_baselines/run.py"
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
    return complete_command

if __name__ == "__main__":
    basedir = "habitat-lab"
    run_type = "eval"
    
    repeats = [1]
    
    encoder_type = "MidLevelEncoder"
    representations = [["keypoints3d"], 
                       ["keypoints3d", "normal"], 
                       ["keypoints3d", "normal", "curvature"],
                       ["keypoints3d", "denoising"],
                       ["keypoints3d", "edge_texture"]]
    enc_repr = list(product([encoder_type], representations)) # baseline + midlevel
    enc_repr.insert(0, ("ResNetEncoder", []))
    # the first empty list means no corruption
    # corruptions = [[], ["brightness"], ["defocus_blur"], ["motion_blur"], ["spatter"], ["speckle_noise"]] 
    corruptions = [[], ["motion_blur"], ["speckle_noise"], ["motion_blur", "speckle_noise"]] 
    severity = 1
    
    combinations = list(product(enc_repr, corruptions, repeats))
    # sort according to the order in corruptions
    combinations = sorted(combinations, key=lambda item: corruptions.index(item[1])) 
    for (encoder, repre), corr, num in combinations:
        complete_command = form_complete_command(basedir, run_type, encoder, repre, corr, severity, num)
        print(complete_command)
        subprocess.run(complete_command, shell=True)