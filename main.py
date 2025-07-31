import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from jittor.transform import Compose, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, ToTensor, ImageNormalize,RandomCropAndResize
# from models.vision_transformer import create_model
from jittor.models import Resnet50,Resnet101

from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import argparse
# from models.vit_v1 import ViT
from sklearn.metrics import classification_report
from sklearn import metrics
jt.flags.use_cuda = 1
from utils.focalloss import FocalLoss
from datetime import datetime
from jittor.lr_scheduler import CosineAnnealingLR
from jimm import resnext101_32x8d,efficientnet_b5,efficientnet_b3,resnet50d,efficientnet_b4,swin_base_patch4_window12_384,swin_large_patch4_window12_384_in22k,vit_base_patch16_224_in21k,vit_base_patch32_384
from jimm.loss import LabelSmoothingCrossEntropy,SoftTargetCrossEntropy,HierarchicalLoss
from jimm.data import RandomMixup,RandomCutmix
import shutil
import torch
# ============== Dataset ==============
class ImageFolder(Dataset):
    def __init__(self, root, annotation_path=None, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.transform = transform
        if annotation_path is not None:
            with open(annotation_path, 'r') as f:
                data_dir = [line.strip().split(' ') for line in f]
                
            data_dir = [(x[0], int(x[1])) for x in data_dir]
        else:
            data_dir = sorted(os.listdir(root))
            data_dir = [(x, None) for x in data_dir]
        self.data_dir = data_dir
        self.total_len = len(self.data_dir)

    def __getitem__(self, idx):
        image_path, label = os.path.join(self.root, self.data_dir[idx][0]), self.data_dir[idx][1]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image_name = self.data_dir[idx][0]
        label = image_name if label is None else label
        return jt.array(image), label

# ============== Model ==============
class Net(nn.Module):
    def __init__(self, num_classes, pretrain,model):
        super().__init__()
        if model =='efficientnet_b5':

            self.base_net=efficientnet_b5(pretrained=pretrain,num_classes=num_classes)
            
        elif model=="efficientnet_b3":
            self.base_net=efficientnet_b3(pretrained=pretrain,num_classes=num_classes)
            # load_checkpoint(self.base_net,"checkpoint/efficientnet_b3.pth",True,False)
            # print("loading:")
        elif model=="efficientnet_b4":
            self.base_net=efficientnet_b4(pretrained=pretrain,num_classes=num_classes)
        elif model=='vit_small_patch16_224':
            self.base_net=create_model('vit_small_patch16_224', pretrained=pretrain, num_classes=num_classes)
        elif model =='resnet50':
            self.base_net = Resnet50(num_classes=num_classes, pretrained=pretrain)
        elif model=='resnet101':
            self.base_net = Resnet101(num_classes=num_classes, pretrained=pretrain)
        elif model=='resnet50d':
            self.base_net = resnet50d(num_classes=num_classes, pretrained=pretrain)
        elif model=='swinT-base':
            self.base_net = swin_base_patch4_window12_384(num_classes=num_classes, pretrained=pretrain)
        elif model=='swinT-large':
            self.base_net = swin_large_patch4_window12_384_in22k(num_classes=num_classes, pretrained=pretrain)
        elif model=='vit-base':
            self.base_net = vit_base_patch16_224_in21k(num_classes=num_classes, pretrained=pretrain)
        elif model=='vit-base2':
            self.base_net = vit_base_patch32_384(num_classes=num_classes, pretrained=pretrain) 
        elif model=="dinov2":
            self.base_net= torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    def execute(self, x):
        x = self.base_net(x)
        return x

# ============== Training ==============
def training(model:nn.Module, optimizer:nn.Optimizer, train_loader:Dataset, now_epoch:int, num_epochs:int,scheduler,loss_function,args):
    model.train()

    losses = []
    pbar = tqdm(train_loader, total=len(train_loader), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]" + " " * (80 - 10 - 10 - 10 - 10 - 3))
    step = 0
    for data in pbar:
        step += 1
        image, label = data
        lam,loss=0,0
        if args.mixup:
            alpha=1.0
            
            # mixup=RandomMixup(args.num_classes)
            # image_,label_=mixup(image,label)
            
            lam = np.random.beta(alpha, alpha)
            index = jt.randperm(image.size(0)).cuda()
            image = lam * image + (1 - lam) * image[index, :]
        if args.cutmix:
            cutmix=RandomCutmix()
            image,label=cutmix(image,label)
            
        pred = model(image)
        # print(pred.shape,pred)
        if args.mixup:
            loss = lam * loss_function(pred, label) + (1 - lam) * loss_function(pred, label[index])
        else:
            loss = loss_function(pred, label.squeeze())
        
        loss.sync()
        optimizer.step(loss)
        losses.append(loss.item())
        pbar.set_description(f'Epoch {now_epoch} [TRAIN] loss = {losses[-1]:.2f}')
    scheduler.step()
    log_message(f'Epoch {now_epoch} / {num_epochs} [TRAIN] mean loss = {np.mean(losses):.2f}')

def evaluate(model:nn.Module, val_loader:Dataset):
    model.eval()
    preds, targets = [], []
    log_message("Evaluating...")
    for data in val_loader:
        image, label = data
        pred = model(image)
        pred.sync()
        pred = pred.numpy().argmax(axis=1)
        preds.append(pred)
        targets.append(label.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)


    f1=metrics.f1_score(targets,preds,average="macro")
    target_names=["0","1","2","3","4","5"]
    log_message(classification_report(targets, preds, target_names=target_names))
    acc = np.mean(np.float32(preds == targets))
    return acc,f1
def log_message(message):
        """Helper function to log messages to file and print to console"""
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
def run(model:nn.Module, optimizer:nn.Optimizer, train_loader:Dataset, val_loader:Dataset, num_epochs:int, modelroot:str,scheduler,loss_function,args):
    best_acc = 0
    erly_stop=0
    best_f1=0
    
    for epoch in range(num_epochs):
        training(model, optimizer, train_loader, epoch, num_epochs,scheduler,loss_function,args)
        acc,f1 = evaluate(model, val_loader)
        if f1 > best_f1:
            best_f1 = f1
            erly_stop=0
            model.save(os.path.join(modelroot, 'best_f1.pkl'))
            log_message("saving best f1 model to:"+modelroot+'/best_f1.pkl')
        if acc>best_acc:
            best_acc=acc
            erly_stop=0
            model.save(os.path.join(modelroot, 'best_acc.pkl'))
            log_message("save best acc model to:"+modelroot+"/best_acc.pkl")
        if args.save_everyepoch:
            model.save(os.path.join(modelroot, str(epoch)+'_'+str(f1)+'.pkl'))
            log_message("saving best model to:"+modelroot)
        erly_stop+=1
        if erly_stop>=30:
            log_message("===========early stop=============")
            break
    
        log_message(f'Epoch {epoch} / {num_epochs} [VAL] best_f1 = {best_f1:.4f} and best acc ={best_acc:4f} in epoch:{epoch:int}, acc = {acc:.4f},f1 score:{f1:.4f}')

# ============== Test ==================

def test(model:nn.Module, test_loader:Dataset, result_path:str):
    model.eval()
    preds = []
    names = []
    print("Testing...")
    for data in test_loader:
        image, image_names = data
        pred = model(image)
        pred.sync()
        pred = pred.numpy().argmax(axis=1)
        preds.append(pred)
        names.extend(image_names)
    preds = np.concatenate(preds)
    with open(result_path, 'w') as f:
        for name, pred in zip(names, preds):
            f.write(name + ' ' + str(pred) + '\n')
def fix_random(seed=0):
    import random

    import torch.backends.cudnn
    import torch.cuda

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    jt.set_global_seed(seed)
    # 
    # torch.use_deterministic_algorithms(True)
    # os.environ["PYTHONHASHSEED"] = "0"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
log_file=""
# ============== Main ==============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--dataroot', type=str, default='./data/TrainSet')
    parser.add_argument('--model', type=str, default="resnet50")
    parser.add_argument('--modelroot', type=str, default='./model_save')
    parser.add_argument('--testonly', action='store_true', default=False)
    parser.add_argument('--loadfrom', type=str, default='./model_save/202506101749/epoch_19_best.pkl')
    parser.add_argument('--result_path', type=str, default='./result.txt')
    parser.add_argument('--img_size', type=int, default=512, help='input image size,512')
    parser.add_argument('--batch_size', type=int, default=18, help='batch size')
    parser.add_argument('--learn_rate', type=float, default=1e-5, help='learn_rate ')
    parser.add_argument('--val_batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--num_classes', type=int, default=6, help='number of classes')
    parser.add_argument('--pretrain', action='store_true', default=False, help='use pretrained model')
    parser.add_argument('--pretrain_path', type=str, default='./model_save/best.pkl', help='path to pretrained model')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training from checkpoint')
    parser.add_argument('--resume_path', type=str, default='./model_save/best.pkl', help='path to resume model')
    parser.add_argument('--early_stop',  action='store_true', default=True)
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--eta_min', type=float, default=1e-5)
    parser.add_argument('--T_max', type=int, default=15)
    parser.add_argument('--loss_fuction', type=str, default='SmoothCE')
    parser.add_argument('--label_path', type=str, default='labels/train-ori.txt')
    parser.add_argument('--val_label_path', type=str, default='labels/val.txt')
    parser.add_argument('--mixup',  action='store_true', default=False,  help="use random mixup")
    parser.add_argument('--cutmix',  action='store_true', default=False,  help="use random cutmix")
    parser.add_argument('--diff_lr',  type=bool, default=False,  help="use diffrent lr")
    parser.add_argument('--save_everyepoch',  action='store_true', default=False,  help="save_everyepoch")
    args = parser.parse_args()
    fix_random(args.seed)
    model = Net(pretrain=True, num_classes=args.num_classes,model=args.model)
    transform_train = Compose([
        Resize((args.img_size, args.img_size)),
        RandomCrop(args.img_size),
        RandomHorizontalFlip(),

        ToTensor(),
        ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = Compose([
        Resize((args.img_size, args.img_size)),
        CenterCrop(args.img_size),
        ToTensor(),
        ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    loss_function=None
    if args.loss_fuction=="CE":
        loss_function=nn.CrossEntropyLoss()
        # loss_fuction=HierarchicalLoss(loss_function)
    elif args.loss_fuction=="FocalLoss":
        loss_function=FocalLoss()
    elif args.loss_fuction=="SmoothCE":
        loss_function=LabelSmoothingCrossEntropy()
    elif args.loss_fuction=="SoftCE":
        loss_function=SoftTargetCrossEntropy()
    elif args.loss_fuction=="weightCE":
        weights=jt.array([4.1667,2.5,7.14,100,80,5]).float().cuda()
        loss_function=nn.CrossEntropyLoss(weight=weights)
    if not args.testonly:
        
        if args.diff_lr:
            base_param_tp = [(n, p) for n, p in model.named_parameters() if "head" not in n ]
            head_param_tp = [(n, p) for n, p in model.named_parameters() if "head" in n]
            optimizer_grouped_parameters = [  # 分层设置学习率
            {'params': [p for n, p in base_param_tp], 'weight_decay': 0.00005, 'lr': args.learn_rate},
            {'params': [p for n, p in head_param_tp], 'weight_decay': 0.001, 'lr': 2e-5}
            ]
            
            optimizer = nn.Adam(optimizer_grouped_parameters, lr=args.learn_rate)
        else:
            optimizer = nn.Adam(model.parameters(), lr=args.learn_rate)
        scheduler = CosineAnnealingLR(optimizer, args.T_max, args.eta_min)
        train_loader = ImageFolder(
            root=os.path.join(args.dataroot, 'images/train'),
            annotation_path=os.path.join(args.dataroot, args.label_path),
            transform=transform_train,
            batch_size=args.batch_size,
            num_workers=8,
            shuffle=True,
            drop_last=True,
        )
        val_loader = ImageFolder(
            root=os.path.join(args.dataroot, 'images/train'),
            annotation_path=os.path.join(args.dataroot, args.val_label_path),
            transform=transform_val,
            batch_size=args.val_batch_size,
            num_workers=8,
            shuffle=False
        )
        # 获取当前时间并格式化为精确到分钟的字符串
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        args.modelroot=os.path.join(args.modelroot,current_time)
        # 创建目录（如果不存在）
        os.makedirs(args.modelroot, exist_ok=True)
        log_file = os.path.join(args.modelroot, 'training_log.txt')
        log_message(f"random,torch manual,jt global seed,setting random seed as : {args.seed}")
        log_message(f"Starting training with parameters: {args}")
        shutil.copy("main.py",os.path.join(args.modelroot,"main.py"))
        run(model, optimizer, train_loader, val_loader, 100, args.modelroot,scheduler,loss_function,args)
    else:
        test_loader = ImageFolder(
            root=args.dataroot,
            transform=transform_val,
            batch_size=8,
            num_workers=8,
            shuffle=False
        )
        model.load(args.loadfrom)
        test(model, test_loader, args.result_path)
