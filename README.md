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


### install mid-level repo

```bash
conda activate habitat
git clone --single-branch --branch visualpriors git@github.com:alexsax/midlevel-reps.git
cd midlevel-reps
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
      representations: ["normal"]
      depth: True # use a separate resnet encoder for depth
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

### Troubleshooting 
If you see an error Module ‘torch.utils’ has no attribute ‘model_zoo’, then go to visualpriors.transforms.py and add `import torch.utils.model_zoo`.