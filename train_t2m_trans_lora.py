import os 
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_train
from dataset import dataset_TM_eval
from dataset import dataset_tokenize
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from exit.utils import get_model, visualize_2motions
from tqdm import tqdm
from exit.utils import get_model, visualize_2motions, generate_src_mask, init_save_folder, uniform, cosine_schedule
from einops import rearrange, repeat
import torch.nn.functional as F
from exit.utils import base_dir

# Import LoRA utilities
from models.lora import add_lora_to_model, get_lora_params, save_lora_weights, merge_lora_weights

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()

# Add LoRA-specific arguments (override defaults if not provided)
if not hasattr(args, 'lora_rank'):
    args.lora_rank = 8
if not hasattr(args, 'lora_alpha'):
    args.lora_alpha = 16
if not hasattr(args, 'lora_dropout'):
    args.lora_dropout = 0.1

torch.manual_seed(args.seed)


init_save_folder(args)

# Only use default VQ paths if resume_pth is not provided
if args.resume_pth is None:
    args.vq_dir = f'./output/vq/{args.vq_name}'
    codebook_dir = f'{args.vq_dir}/codebook/'
    args.resume_pth = f'{args.vq_dir}/net_last.pth'
else:
    # Use user-provided path, infer vq_dir from it
    args.vq_dir = os.path.dirname(args.resume_pth)
    codebook_dir = f'{args.vq_dir}/codebook/'

os.makedirs(args.vq_dir, exist_ok = True)
os.makedirs(codebook_dir, exist_ok = True)
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.out_dir+'/html', exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
logger.info("\n" + "="*60)
logger.info("LoRA Configuration:")
logger.info(f"  - Rank: {args.lora_rank}")
logger.info(f"  - Alpha: {args.lora_alpha}")
logger.info(f"  - Dropout: {args.lora_dropout}")
logger.info("="*60 + "\n")


from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)
clip.model.convert_weights(clip_model)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# https://github.com/openai/CLIP/issues/111
class TextCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        with torch.no_grad():
            word_emb = self.model.token_embedding(text).type(self.model.dtype)
            word_emb = word_emb + self.model.positional_embedding.type(self.model.dtype)
            word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
            word_emb = self.model.transformer(word_emb)
            word_emb = self.model.ln_final(word_emb).permute(1, 0, 2).float()
            enctxt = self.model.encode_text(text).float()
        return enctxt, word_emb
clip_model = TextCLIP(clip_model)

##### ---- VQ-VAE (Frozen) ---- #####
net = vqvae.HumanVQVAE(args,
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)

print ('Loading VQ-VAE checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

# Freeze VQ-VAE
for param in net.parameters():
    param.requires_grad = False

##### ---- Transformer with LoRA ---- #####
trans_encoder = trans.Text2Motion_Transformer(vqvae=net,
                                num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                num_local_layer=args.num_local_layer, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)

# Load pretrained transformer weights
if args.resume_trans is not None:
    print(f'Loading pretrained transformer from {args.resume_trans}')
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    # Use strict=False for dynamic length support
    # Custom loading logic to handle size mismatch (51 -> 42)
    state_dict = ckpt['trans']
    model_dict = trans_encoder.state_dict()
    
    for key, value in state_dict.items():
        if key in model_dict:
            if value.shape != model_dict[key].shape:
                # Handle pos_embedding mismatch
                if 'pos_embedding' in key or 'pos_embed' in key:
                    print(f"Resizing {key}: {value.shape} -> {model_dict[key].shape}")
                    # Truncate to match model size
                    if len(value.shape) == 2:
                        state_dict[key] = value[:model_dict[key].shape[0], :]
                    elif len(value.shape) == 3:
                        state_dict[key] = value[:model_dict[key].shape[0], :, :]
                # Handle attention mask mismatch
                elif 'mask' in key:
                    print(f"Resizing {key}: {value.shape} -> {model_dict[key].shape}")
                    # Truncate mask (usually last two dims are sequence length)
                    if len(value.shape) == 4:
                        state_dict[key] = value[:, :, :model_dict[key].shape[2], :model_dict[key].shape[3]]

    missing_keys, unexpected_keys = trans_encoder.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Note: {len(missing_keys)} parameters not loaded (possibly due to extended position encodings)")
else:
    print("Warning: No pretrained transformer provided. Training from scratch.")

# Apply LoRA to transformer
print(f"\nApplying LoRA to transformer...")
print(f"  Rank={args.lora_rank}, Alpha={args.lora_alpha}, Dropout={args.lora_dropout}")

# Add LoRA to attention layers (query, key, value, projection)
target_modules = ['query', 'key', 'value', 'proj']  # Common choices for attention
trans_encoder = add_lora_to_model(
    trans_encoder,
    target_modules=target_modules,
    rank=args.lora_rank,
    alpha=args.lora_alpha,
    dropout=args.lora_dropout
)

# Count parameters
total_params = sum(p.numel() for p in trans_encoder.parameters())
lora_params = get_lora_params(trans_encoder)
trainable_params = sum(p.numel() for p in lora_params)

print(f"\nParameter Statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable (LoRA) parameters: {trainable_params:,}")
print(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
print(f"  Parameter reduction: {100*(1-trainable_params/total_params):.2f}%\n")

trans_encoder.train()
trans_encoder.cuda()
trans_encoder = torch.nn.DataParallel(trans_encoder)

##### ---- Optimizer & Scheduler (Only LoRA params) ---- #####
# Only optimize LoRA parameters!
optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss(reduction='none')

##### ---- get code ---- #####
##### ---- Dataloader ---- #####
if len(os.listdir(codebook_dir)) == 0:
    print("Generating codebook...")
    train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2**args.down_t)
    for batch in train_loader_token:
        pose, name = batch
        bs, seq = pose.shape[0], pose.shape[1]

        pose = pose.cuda().float()
        target = net(pose, type='encode')
        target = target.cpu().numpy()
        np.save(pjoin(codebook_dir, name[0] +'.npy'), target)


train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, codebook_dir, unit_length=2**args.down_t)
train_loader_iter = dataset_TM_train.cycle(train_loader)

        
##### ---- Training ---- #####
best_fid = 1000 
best_iter = 0 
best_div = 100 
best_top1 = 0 
best_top2 = 0 
best_top3 = 0 
best_matching = 100 

def get_acc(cls_pred, target, mask):
    cls_pred = torch.masked_select(cls_pred, mask.unsqueeze(-1)).view(-1, cls_pred.shape[-1])
    target_all = torch.masked_select(target, mask)
    probs = torch.softmax(cls_pred, dim=-1)
    _, cls_pred_index = torch.max(probs, dim=-1)
    right_num = (cls_pred_index == target_all).sum()
    return right_num*100/mask.sum()

for nb_iter in tqdm(range(1, args.total_iter + 1), position=0, leave=True):
    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len = batch
    m_tokens, m_tokens_len = m_tokens.cuda(), m_tokens_len.cuda()
    bs = m_tokens.shape[0]
    target = m_tokens
    target = target.cuda()
    batch_size, max_len = target.shape[:2]

    text = clip.tokenize(clip_text, truncate=True).cuda()
    
    feat_clip_text, word_emb = clip_model(text)

    # [INFO] Swap input tokens
    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(target.shape,
                                                device=target.device))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(target.shape,
                                                device=target.device))
    seq_mask_no_end = generate_src_mask(max_len, m_tokens_len)
    mask = torch.logical_or(mask, ~seq_mask_no_end).int()
    r_indices = torch.randint_like(target, args.nb_code)
    input_indices = mask*target+(1-mask)*r_indices

    # Time step masking
    mask_id = get_model(net).vqvae.num_code + 2
    rand_mask_probs = torch.zeros(batch_size, device = m_tokens_len.device).float().uniform_(0.5, 1)
    num_token_masked = (m_tokens_len * rand_mask_probs).round().clamp(min = 1)
    seq_mask = generate_src_mask(max_len, m_tokens_len+1)
    batch_randperm = torch.rand((batch_size, max_len), device = target.device) - seq_mask_no_end.int()
    batch_randperm = batch_randperm.argsort(dim = -1)
    mask_token = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

    masked_input_indices = torch.where(mask_token, mask_id, input_indices)

    att_txt = None
    cls_pred = trans_encoder(masked_input_indices, feat_clip_text, src_mask = seq_mask, att_txt=att_txt, word_emb=word_emb)[:, 1:]

    # [INFO] Compute xent loss as a batch
    weights = seq_mask_no_end / (seq_mask_no_end.sum(-1).unsqueeze(-1) * seq_mask_no_end.shape[0])
    cls_pred_seq_masked = cls_pred[seq_mask_no_end, :].view(-1, cls_pred.shape[-1])
    target_seq_masked = target[seq_mask_no_end]
    weight_seq_masked = weights[seq_mask_no_end]
    loss_cls = F.cross_entropy(cls_pred_seq_masked, target_seq_masked, reduction = 'none')
    loss_cls = (loss_cls * weight_seq_masked).sum()

    ## global loss
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    if nb_iter % args.print_iter ==  0 :
        probs_seq_masked = torch.softmax(cls_pred_seq_masked, dim=-1)
        _, cls_pred_seq_masked_index = torch.max(probs_seq_masked, dim=-1)
        target_seq_masked = torch.masked_select(target, seq_mask_no_end)
        right_seq_masked = (cls_pred_seq_masked_index == target_seq_masked).sum()

        writer.add_scalar('./Loss/all', loss_cls, nb_iter)
        writer.add_scalar('./ACC/every_token', right_seq_masked*100/seq_mask_no_end.sum(), nb_iter)
        
        # [INFO] log mask/nomask separately
        no_mask_token = ~mask_token * seq_mask_no_end
        writer.add_scalar('./ACC/masked', get_acc(cls_pred, target, mask_token), nb_iter)
        writer.add_scalar('./ACC/no_masked', get_acc(cls_pred, target, no_mask_token), nb_iter)

    if nb_iter==0 or nb_iter % args.eval_iter ==  0 or nb_iter == args.total_iter:
        num_repeat = 1
        rand_pos = False
        if nb_iter == args.total_iter:
            num_repeat = -30
            rand_pos = True
            val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)
        pred_pose_eval, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = eval_trans.evaluation_transformer(args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper, dataname=args.dataname, num_repeat=num_repeat, rand_pos=rand_pos)
        
        # Save LoRA weights (lightweight)
        lora_save_path = pjoin(args.out_dir, f'lora_{nb_iter:06d}.pth')
        save_lora_weights(trans_encoder.module if hasattr(trans_encoder, 'module') else trans_encoder, lora_save_path)
        logger.info(f"LoRA weights saved to {lora_save_path}")
        
        # Also save merged model for convenience
        merged_model = trans_encoder.module if hasattr(trans_encoder, 'module') else trans_encoder
        merged_model = merge_lora_weights(merged_model)
        merged_save_path = pjoin(args.out_dir, f'net_{nb_iter:06d}.pth')
        torch.save({'trans': merged_model.state_dict()}, merged_save_path)
        logger.info(f"Merged model saved to {merged_save_path}")

    if nb_iter == args.total_iter: 
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        
        # Final save
        final_lora_path = pjoin(args.out_dir, 'lora_last.pth')
        save_lora_weights(trans_encoder.module if hasattr(trans_encoder, 'module') else trans_encoder, final_lora_path)
        logger.info("\n" + "="*60)
        logger.info(f"Final LoRA weights saved to {final_lora_path}")
        logger.info(f"LoRA file size: {os.path.getsize(final_lora_path) / 1024 / 1024:.2f} MB")
        logger.info("="*60 + "\n")
        break
