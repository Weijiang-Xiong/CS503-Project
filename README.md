## CS503 Project Repo

Project based on [habitat-lab](HABITAT_README.md)

### download dataset
- download and extract gibson scene dataset, so the files look like `data/scene_datasets/gibson/{scene}.glb`
  
```bash 
aria2c -x 4 https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat.zip
```
    
- download and extract the pointnav dataset so the files look like `data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz`

```bash 
aria2c -x 4 https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip
```


### install mid-level repo and imagenet-c

Mid-level repo

```bash
conda activate habitat
git clone --single-branch --branch visualpriors git@github.com:alexsax/midlevel-reps.git
cd midlevel-reps
python -m pip install -e .
```

Download [this folder](https://github.com/hendrycks/robustness/tree/master/ImageNet-C/imagenet_c) from the imagenet-c repo.
Be sure **NOT** to install the requirement.txt, otherwise the version of numpy will be messed. 

```bash
conda activate habitat
cd imagenet_c
python -m pip install -e .
```

### Experiment configs

File `habitat_baselines/config/pointnav/ppo_pointnav_example.yaml` has a section as shown in the code block below.
To use mid-level representations during RL training, set `encoder` to `"MidLevelEncoder"`, and then set `representations` to a list of required mid-level features. 
Mid-level handles RGB only, but the original ResNetEncoder also encodes depth information, so when we use mid-level encoder, and `depth` is true, we will build a **depth-only** ResNetEncoder in addition to mid-level features. 

```yaml
RL:
  POLICY:
    NET_CONF:
      encoder: "ResNetEncoder" # ResNetEncoder MidLevelEncoder
      representations: ["normal"] # only works when the encoder is MidLevelEncoder
      depth: True # use a separate resnet encoder for depth # only works when the encoder is MidLevelEncoder
      corruption: [] # empty corruption for training, manually assign this during testing 
      severity: 1 # severity of the corruption, must be 1, 2, 3, 4 or 5
```

It's also possible to modify existing settings through command line, this avoids some duplicates in yaml files, but the command looks really ugly.
However, we can not add new settings directly from command line. 

```bash
python -u habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav_example.yaml --run-type train RL.POLICY.NET_CONF.representations "[\"normal\", \"keypoints3d\"]"
```

To run on IZAR, you can use `python_wrapper.sh`, but be sure to change the `--account` and `--reservation` flag

```bash
sbatch python_wrapper.sh habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav_example.yaml --run-type train RL.POLICY.NET_CONF.representations "[\"normal\", \"keypoints3d\"]"
```

### How to form up a complete command 

- common part of the training command, running evaluation, the last `train` should be replaced with `eval`
    
    ```bash
    sbatch python_wrapper.sh habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav_example.yaml --run-type train
    ```
    
- variable part for different settings, put them after the common part, **better be consistent in training and testing command**
    - assign checkpoint save path `EVAL_CKPT_PATH_DIR data/ckpt/my_checkpoint CHECKPOINT_FOLDER data/ckpt/my_checkpoint`
    - assign video path: `VIDEO_DIR video/my_video_dir`
    - use ResNet encoder: `RL.POLICY.NET_CONF.encoder ResNetEncoder`
    - use mid-level encoder: `RL.POLICY.NET_CONF.encoder MidLevelEncoder RL.POLICY.NET_CONF.representations "[\"normal\", \"keypoints3d\"]"`
    - add corruption (**only in testing, don't do this in training**)
     `RL.POLICY.NET_CONF.corruption "[\"defocus_blur\", \"motion_blur\"]" RL.POLICY.NET_CONF.severity 1`

- how to form a complete command:
    - copy the common part, modify the configuration yaml file if needed
    - copy the checkpoint save path and video path, give unambiguous folder names
    - copy an encoder, resnet or midlevel, and modify the representations if using midlevel, remember to use `\"` for an representation instead of `"`
    - when done training, open bash history, copy the training command, and replace the run type `train` with `eval`, and then add the corruption


```bash
# an example with resnet encoder
sbatch python_wrapper.sh habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav_example.yaml --run-type train EVAL_CKPT_PATH_DIR data/ckpt/my_checkpoint CHECKPOINT_FOLDER data/ckpt/my_checkpoint VIDEO_DIR video/my_video_dir RL.POLICY.NET_CONF.encoder ResNetEncoder
```

- what mid level feature to use: `keypoints3d`, `normal`, `curvature`, `denoising`, `edge_texture`
    
    evaluated groups: 
    
    1. `keypoints3d` 
    
    2. `keypoints3d`, `normal` 
    
    3. `keypoints3d`, `normal`, `curvature`  
    
    4. `keypoints3d`, `denoising`  
    
    5. `keypoints3d`, `edge_texture`
    
    - all supported
        
        ```bash
        autoencoding          depth_euclidean          jigsaw                  reshading          
        colorization          edge_occlusion           keypoints2d             room_layout      
        curvature             edge_texture             keypoints3d             segment_unsup2d        
        class_object          egomotion                nonfixated_pose         segment_unsup25d
        class_scene           fixated_pose             normal                  segment_semantic      
        denoising             inpainting               point_matching          vanishing_point
        ```
        

- what corruptions to use `brightness`, `defocus_blur`, `motion_blur`, `spatter`, `speckle_noise`
    - all supported
        
        ```
        gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, 
        zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, 
        jpeg_compression, speckle_noise, gaussian_blur, spatter, saturate
        ```


### Troubleshooting 

If you see an error `Module ‘torch.utils’ has no attribute ‘model_zoo’`, then go to the mid-level repo and find `visualpriors.transforms.py` and add `import torch.utils.model_zoo`.