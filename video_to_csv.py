import re 
import pandas as pd 
import argparse
from pathlib import Path


def get_data(path):
    df=pd.DataFrame(columns=['episode','ckpt','distance_to_goal','SR','SPL','collisions_count'])
    p=Path(path)
    file_list = [item.name for item in list(p.glob('*.mp4'))]
    data=list()
    for i in file_list:
        data.append(re.findall(r"\d+\.?\d*",i)[:-1])
    for i in range(len(data)):
        df.loc[i]=data[i]
    return df.astype(str).astype(float)


def run(args):
    base_dir = Path(args.base_dir)
    all_video_dirs = [x for x in base_dir.iterdir() if x.is_dir()]

    for video_dir in all_video_dirs:
        df = get_data(video_dir)
        save_path = base_dir / "{}.csv".format(video_dir.name)
        df.to_csv(save_path, index=None)
        
        print("File saved to {}".format(save_path))
        

if __name__ == "__main__":
    
    parser  = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="./video")
    args = parser.parse_args()
    
    run(args)
    