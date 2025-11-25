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
    parser.add_argument("--block-size", type=int, default=256, help="maximum sequence length (increased for dynamic length support)")
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
    parser.add_argument('--base-text', type=str, default=None, help='Text for base motion generation')
    parser.add_argument('--inbetween-text', type=str, default=None, help='Text for in-betweening')
    
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

    humanml3d_root = './content/drive/MyDrive/cvpdl/final_exp/MMM/dataset/HumanML3D'
    humanml3d_mean_path = os.path.join(humanml3d_root, 'Mean.npy')
    humanml3d_std_path = os.path.join(humanml3d_root, 'Std.npy')
    
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        print("Error: AIST++ Mean/Std not found. Please ensure AIST++ dataset is properly set up.")
        return
    
    mean = np.load(humanml3d_mean_path)
    std = np.load(humanml3d_std_path)
    
    # Initialize MMM model
    args.dataname = 't2m' # Force t2m for model config compatibility
    
    # Check for checkpoints
    if not args.resume_pth or not args.resume_trans:
        print("Error: Please provide --resume-pth and --resume-trans arguments.")
        return

    print("Loading MMM model...")
    mmm = MMM(args).to(device)
    mmm.eval()
    
    # Determine mode: AIST++ data or text-based generation
    use_aist_data = args.motion_id is not None
    
    if use_aist_data:
        # Mode 1: Load AIST++ motion data
        print(f"Loading AIST++ motion data for ID: {args.motion_id}")
        try:
            motion_data, motion_text = load_aist_data(aist_root, args.motion_id)
            print(f"Loaded motion shape: {motion_data.shape}")
            print(f"Motion text: {motion_text}")
            
            # Use motion's actual length
            motion_length = len(motion_data)
            m_length = torch.tensor([motion_length]).long().to(device)
            
            # Normalize the motion data
            motion_normalized = (motion_data - mean) / std
            
            # Convert to tensor and add batch dimension
            base_pose = torch.from_numpy(motion_normalized).float().to(device).unsqueeze(0)
            
            # Use motion text or inbetween text
            inbetween_text_str = args.inbetween_text if args.inbetween_text else (motion_text if motion_text else 'A person is dancing')
            
            print(f"Using motion length: {motion_length}")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    else:
        # Mode 2: Text-based generation (original approach)
        print("Using text-based motion generation mode")
        base_text_str = args.base_text if args.base_text else 'a person walks forward in a straight line.'
        length_val = args.length if args.length else 196
        
        print(f"Base Text: {base_text_str}")
        print(f"Length: {length_val}")
        
        text = [base_text_str]
        m_length = torch.tensor([length_val]).long().to(device)
        
        # Generate base pose
        print("Generating base pose from text...")
        with torch.no_grad():
            base_pose = mmm(text, m_length, rand_pos=False)
        
        inbetween_text_str = args.inbetween_text if args.inbetween_text else 'A person jumps forward'
    
    print(f"Inbetween Text: {inbetween_text_str}")
    
    # Calculate start and end frames
    start_f = (m_length * 0.25).int()
    end_f = (m_length * 0.75).int()
    
    inbetween_text = [inbetween_text_str]
    
    print(f"In-betweening from frame {start_f.item()} to {end_f.item()}...")
    
    with torch.no_grad():
        pred_pose_inbetween = mmm.inbetween_eval(base_pose, m_length, start_f, end_f, inbetween_text)
    
    # Visualization
    k = 0
    x = pred_pose_inbetween[k, :m_length[k]].detach().cpu().numpy()
    l = m_length[k].item()
    caption = inbetween_text[k]
    
    # Save result
    out_dir = './output_inbetween'
    os.makedirs(out_dir, exist_ok=True)
    
    # Save npy (denormalized)
    x_denorm = x * std + mean
    
    if use_aist_data:
        out_path = os.path.join(out_dir, f'inbetween_{args.motion_id}.npy')
        html_path = os.path.join(out_dir, f'inbetween_{args.motion_id}.html')
    else:
        out_path = os.path.join(out_dir, 'inbetween_result.npy')
        html_path = os.path.join(out_dir, 'inbetween_result.html')
    
    np.save(out_path, x_denorm)
    print(f"Saved result to {out_path}")

    print(f"Visualizing to {html_path}...")
    visualize_2motions(x, std, mean, 't2m', l, save_path=html_path)
    
    print("Done!")

if __name__ == '__main__':
    main()
