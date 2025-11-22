import numpy as np
import os
import argparse
import torch
from exit.utils import visualize_2motions

def get_args():
    parser = argparse.ArgumentParser(description='Visualize AIST++ motion')
    parser.add_argument('--motion-id', type=str, required=True, help='Motion ID to visualize (without extension)')
    return parser.parse_args()

def main():
    args = get_args()
    motion_id = args.motion_id
    
    data_root = './AIST++'
    motion_path = os.path.join(data_root, 'new_joint_vecs', motion_id + '.npy')
    mean_path = os.path.join(data_root, 'Mean.npy')
    std_path = os.path.join(data_root, 'Std.npy')
    
    if not os.path.exists(motion_path):
        print(f"Error: Motion file not found: {motion_path}")
        return
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        print("Error: Mean/Std not found in ./AIST++")
        return
        
    print(f"Loading motion: {motion_id}")
    motion = np.load(motion_path)
    mean = np.load(mean_path)
    std = np.load(std_path)
    
    # Normalize motion because visualize_2motions expects normalized input and denormalizes it
    norm_motion = (motion - mean) / std
    
    out_dir = './output_vis'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'{motion_id}.html')
    
    print(f"Visualizing to {save_path}...")
    visualize_2motions(norm_motion, std, mean, 't2m', len(motion), save_path=save_path)
    print("Done.")

if __name__ == '__main__':
    main()
