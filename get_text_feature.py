import argparse
import os

parser = argparse.ArgumentParser(description='Text Condition')
parser.add_argument('--gpu_id', type=str, default='0', help='GPU id')
parser.add_argument('--label_flag', type=str, default='N8', help='label nums: N8, D1,...,D20')
parser.add_argument('--save_path', type=str, default='text_feature.pth', help='save path')
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import clip
import torch
from utils import getClassIndex, get_classes

use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(1111)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

clip_model, _ = clip.load('ViT-B/16')
clip_model = clip_model.to(device)
class_list = getClassIndex()
class_text = torch.cat([clip.tokenize(f"a photo of a {class_list[c][1]}") for c in range(len(class_list))]).to(device)
text_embed = clip_model.encode_text(class_text)
text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
label_set = get_classes(args.label_flag)
text_cond_dict = dict()
for label in label_set:
    text_cond_dict[label] = text_embed[label]
print(text_cond_dict)
torch.save(text_cond_dict, args.save_path)