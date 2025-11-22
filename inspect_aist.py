import numpy as np
import os

data_root = './AIST++'
motion_dir = os.path.join(data_root, 'new_joint_vecs')
text_dir = os.path.join(data_root, 'texts')
split_file = os.path.join(data_root, 'train.txt')

# Read first ID from train.txt
with open(split_file, 'r') as f:
    first_id = f.readline().strip()

print(f"First ID: {first_id}")

# Load motion
motion_path = os.path.join(motion_dir, first_id + '.npy')
if os.path.exists(motion_path):
    motion = np.load(motion_path)
    print(f"Motion shape: {motion.shape}")
else:
    print(f"Motion file not found: {motion_path}")

# Read text
text_path = os.path.join(text_dir, first_id + '.txt')
if os.path.exists(text_path):
    with open(text_path, 'r') as f:
        print(f"Text content:\n{f.read()}")
else:
    print(f"Text file not found: {text_path}")
