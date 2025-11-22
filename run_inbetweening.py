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
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## dataloader
    parser.add_argument('--dataname', type=str, default='t2m', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--fps', default=[20], nargs="+", type=int, help='frames per second')
    parser.add_argument('--seq-len', type=int, default=64, help='training motion length')
    
    ## optimization
    parser.add_argument('--total-iter', default=300000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[150000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--decay-option',default='all', type=str, choices=['all', 'noVQ'], help='disable weight decay on codebook')
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=32, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=8192, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')

    ## gpt arch
    parser.add_argument("--block-size", type=int, default=51, help="seq len")
    parser.add_argument("--embed-dim-gpt", type=int, default=1024, help="embedding dimension")
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--num-layers", type=int, default=9, help="nb of transformer layers")
    parser.add_argument("--num-local-layer", type=int, default=2, help="nb of transformer local layers")
    parser.add_argument("--n-head-gpt", type=int, default=16, help="nb of heads")
    parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    parser.add_argument("--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding")
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--quantbeta', type=float, default=1.0, help='dataset directory')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume vq pth')
    parser.add_argument("--resume-trans", type=str, default=None, help='resume gpt pth')
    
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    parser.add_argument('--vq-name', type=str, default='VQVAE', help='name of the generated dataset .npy, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=10000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    parser.add_argument("--if-maxtest", action='store_true', help="test in max")
    parser.add_argument('--pkeep', type=float, default=.5, help='keep rate for gpt training')
    
    ## generator
    parser.add_argument('--text', type=str, help='text')
    parser.add_argument('--length', type=int, help='length')

    # Custom arguments
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
