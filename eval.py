import argparse
from torch.utils.data import DataLoader
from utils import *
from models.generator import CrossAttenGenerator

parser = argparse.ArgumentParser(description='Clip-based Generative Networks')
parser.add_argument('--data_dir', default='./dataset/ImageNet1k', help='ImageNet validation data')
parser.add_argument('--is_nips', action='store_true', default=True,  help='Evaluation on NIPS data')
parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
parser.add_argument('--eps', type=int, default=16, help='Perturbation budget')
parser.add_argument('--model_type', type=str, default='res152',  help='Source model')
parser.add_argument('--load_path', type=str, default='checkpoints/res152/model-9.pth', help='Load path')
parser.add_argument('--label_flag', type=str, default='N8', help='Label nums: N8, C20,...,C200')
parser.add_argument('--nz', type=int, default=16, help='nz')
parser.add_argument('--finetune', action='store_true', help='Finetune for single class attack')
parser.add_argument('--finetune_class', type=int, help='Class id to be finetuned')
parser.add_argument('--save_dir', type=str, default='results', help='Dir to save adv images')
args = parser.parse_args()
print(args)

eps = args.eps/255.
n_class = 1000

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Input dimensions
if args.model_type == 'incv3':
    scale_size = 300
    img_size = 299
else:
    scale_size = 256
    img_size = 224

if args.model_type == 'incv3':
    netG = CrossAttenGenerator(inception=True, nz=args.nz)
else:
    netG = CrossAttenGenerator(nz=args.nz)

# Load Generator
netG.load_state_dict(torch.load(args.load_path, map_location=device))
netG = netG.to(device)
netG.eval()

# Setup-Data
data_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

test_set = datasets.ImageFolder(args.data_dir, data_transform)
test_size = len(test_set)
print('Test data size:', test_size)

# Fix labels if needed
if args.is_nips:
    print('is_nips')
    test_set = fix_labels_nips(args, test_set, pytorch=True)
else:
    test_set = fix_labels(args, test_set)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
if args.finetune:
    class_ids = np.array([args.finetune_class])
else:
    class_ids = get_classes(args.label_flag)

print(class_ids)

text_cond_dict = torch.load('text_feature.pth')

# save
for idx in range(len(class_ids)):
    for i, (img, label) in enumerate(test_loader):
        img = img.to(device)
        cond = torch.tile(text_cond_dict[class_ids[idx]], (args.batch_size, 1)).to(torch.float).to(device)
        noise = netG(img, cond, eps=eps)
        adv = noise + img

        # Projection, tanh() have been operated in models.
        adv = torch.min(torch.max(adv, img - eps), img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)

        save_imgs = adv.detach().cpu()
        for j in range(len(save_imgs)):
            g_img = transforms.ToPILImage('RGB')(save_imgs[j])
            output_dir = '{}/gan_n8/{}_t{}/images'.format(args.save_dir, args.load_path.split('/')[-1].split('.pth')[0], class_ids[idx])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            g_img.save(os.path.join(output_dir, '{}_{}.png'.format(class_ids[idx], i * args.batch_size + j)))
            
        
