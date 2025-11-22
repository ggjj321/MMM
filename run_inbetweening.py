import torch
import numpy as np
import os
import argparse
from generate import MMM
import options.option_transformer as option_trans
from utils.word_vectorizer import WordVectorizer
import clip
from exit.utils import visualize_2motions

def get_args():
    parser = option_trans.get_args_parser(parse_args=False)
    parser.add_argument('--motion-id', type=str, default=None, help='Motion ID to process (without extension)')
    return parser.parse_args()

def load_aist_data(data_root, motion_id):
    motion_path = os.path.join(data_root, 'new_joint_vecs', motion_id + '.npy')
    text_path = os.path.join(data_root, 'texts', motion_id + '.txt')
    
    if not os.path.exists(motion_path):
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    
    motion = np.load(motion_path)
    
    text_content = ""
    if os.path.exists(text_path):
        with open(text_path, 'r') as f:
            # AIST++ text format: caption#tokens#start#end
            line = f.readline().strip()
            parts = line.split('#')
            if len(parts) >= 1:
                text_content = parts[0]
    
    return motion, text_content

def main():
    args = get_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load AIST++ Mean and Std
    aist_root = './AIST++'
    mean_path = os.path.join(aist_root, 'Mean.npy')
    std_path = os.path.join(aist_root, 'Std.npy')
    
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        print("Warning: AIST++ Mean/Std not found. Using default or T2M mean/std if available.")
        pass
    
    mean = np.load(mean_path)
    std = np.load(std_path)
    
    # Initialize MMM model
    args.dataname = 't2m' # Force t2m for model config compatibility
    
    # Check for checkpoints
    if not args.resume_pth or not args.resume_trans:
        print("Error: Please provide --resume-pth and --resume-trans arguments.")
        return

    print("Loading MMM model...")
    mmm = MMM(args).to(device)
    mmm.eval()
    
    # Load motion
    if args.motion_id:
        motion_id = args.motion_id
    else:
        # Pick random/first
        train_txt = os.path.join(aist_root, 'train.txt')
        with open(train_txt, 'r') as f:
            motion_id = f.readline().strip()
        print(f"No motion ID provided. Using first one: {motion_id}")
    
    print(f"Processing motion: {motion_id}")
    try:
        motion, text = load_aist_data(aist_root, motion_id)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Original Text: {text}")
    
    # Normalize motion
    norm_motion = (motion - mean) / std
    norm_motion = torch.from_numpy(norm_motion).float().to(device).unsqueeze(0) # (1, seq_len, dim)
    
    # Define in-betweening parameters
    seq_len = norm_motion.shape[1]
    if seq_len < 40:
        print(f"Motion too short ({seq_len}), skipping.")
        return
        
    start_f = 20
    end_f = seq_len - 20
    
    print(f"In-betweening from frame {start_f} to {end_f} with text: '{text}'")
    
    m_length = torch.tensor([seq_len]).to(device)
    start_f_tensor = torch.tensor([start_f]).to(device)
    end_f_tensor = torch.tensor([end_f]).to(device)
    
    with torch.no_grad():
        pred_pose = mmm.inbetween_eval(norm_motion, m_length, start_f_tensor, end_f_tensor, [text])
    
    # Prepare for visualization (normalized numpy)
    pred_pose_norm_npy = pred_pose.cpu().numpy()[0]
    
    # Denormalize for saving npy
    pred_pose_denorm = pred_pose_norm_npy * std + mean
    
    # Save result
    out_dir = './output_inbetween'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{motion_id}_inbetween.npy')
    np.save(out_path, pred_pose_denorm)
    print(f"Saved result to {out_path}")
    
    # Save original
    orig_path = os.path.join(out_dir, f'{motion_id}_original.npy')
    np.save(orig_path, motion)
    print(f"Saved original to {orig_path}")

    # Visualize
    html_path = os.path.join(out_dir, f'{motion_id}.html')
    print(f"Visualizing to {html_path}...")
    # visualize_2motions expects normalized motion and does denormalization internally
    visualize_2motions(pred_pose_norm_npy, std, mean, 't2m', seq_len, save_path=html_path)

if __name__ == '__main__':
    main()
