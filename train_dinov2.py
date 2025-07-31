import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
 
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
 
import torchvision
from torchvision import datasets, models, transforms
 
# 设置环境变量
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
# 启用分布式调试信息
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
 
def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size
    )
    torch.distributed.barrier()
 
def cleanup():
    dist.destroy_process_group()
 
def main_worker(rank, world_size, args):
    # 设置GPU
    setup(rank, world_size)
    
    # 设置随机种子保证各卡一致性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 数据加载部分 - 仅在主进程打印信息
    if rank == 0:
        print(f"使用 {world_size} 个GPU进行训练")
    
    # 数据处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    ROOT_PATH = args.data_path
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(ROOT_PATH, x), data_transforms[x]) 
        for x in ['train', 'val']
    }
    
    if rank == 0:
        print(f"训练集大小: {len(image_datasets['train'])}")
        print(f"验证集大小: {len(image_datasets['val'])}")
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(
        image_datasets['train'],
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed
    )
    
    val_sampler = DistributedSampler(
        image_datasets['val'],
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # 调整batch_size为每卡的大小
    per_gpu_batch_size = args.batch_size // world_size
    
    data_loaders = {
        'train': DataLoader(
            image_datasets['train'],
            batch_size=per_gpu_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=train_sampler
        ),
        'val': DataLoader(
            image_datasets['val'],
            batch_size=per_gpu_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=val_sampler
        )
    }
    
    class_names = image_datasets['train'].classes
    if rank == 0:
        print(f"类别名称: {class_names}")
    
    # 加载预训练模型
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    
    # 创建模型
    class DinoVisionTransformerClassifier(nn.Module):
        def __init__(self):
            super(DinoVisionTransformerClassifier, self).__init__()
            self.transformer = dinov2_vits14
            self.classifier = nn.Sequential(
                nn.Linear(384, 256),
                nn.ReLU(),
                nn.Linear(256, len(class_names))
            )
        
        def forward(self, x):
            x = self.transformer(x)
            # 确保只使用分类token
            if isinstance(x, dict):
                x = x['x_norm_clstoken']
            x = self.classifier(x)
            return x
    
    model = DinoVisionTransformerClassifier().to(rank)
    
    # 多卡包装，添加find_unused_parameters=True参数
    model = nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[rank],
        find_unused_parameters=True
    )
    
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 训练循环
    best_acc = 0.0
    for epoch in range(args.epochs):
        # 设置采样器epoch，确保每个epoch的shuffle不同
        train_sampler.set_epoch(epoch)
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        if rank == 0:
            loop = tqdm(data_loaders['train'], desc=f"Epoch [{epoch}/{args.epochs}]")
        else:
            loop = data_loaders['train']
        
        for idx, (features, labels) in enumerate(loop):
            features = features.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if rank == 0:
                loop.set_postfix(loss=train_loss/(idx+1), acc=100.*train_correct/train_total)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predicted = []
        val_labels_list = []
        
        with torch.no_grad():
            for features, labels in data_loaders['val']:
                features = features.to(rank, non_blocking=True)
                labels = labels.to(rank, non_blocking=True)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # 收集预测结果用于后续分析
                val_predicted.append(predicted.cpu())
                val_labels_list.append(labels.cpu())
        
        # 收集所有GPU的结果
        val_loss = torch.tensor(val_loss).to(rank)
        val_correct = torch.tensor(val_correct).to(rank)
        val_total = torch.tensor(val_total).to(rank)
        
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_total, op=dist.ReduceOp.SUM)
        
        val_loss = val_loss.item() / len(data_loaders['val'])
        val_acc = 100. * val_correct.item() / val_total.item()
        
        # 仅在主进程打印结果
        if rank == 0:
            print(f"Epoch [{epoch}/{args.epochs}] 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.module.state_dict(), 'best_model.pth')
                print(f"模型已保存，准确率: {best_acc:.2f}%")
        all_predicted = torch.cat(val_predicted).numpy()
        all_labels = torch.cat(val_labels_list).numpy()
        
        # 打印分类报告
        print("***************打印分类报告***************")
        print(classification_report(all_labels, all_predicted, target_names=class_names))
        # 更新学习率
        scheduler.step()
    
    # 评估模型
    if rank == 0:
        print(f"最佳验证准确率: {best_acc:.2f}%")
        
        # 加载最佳模型
        model.module.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        
        # 收集所有预测结果
        all_predicted = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in data_loaders['val']:
                features = features.to(rank)
                labels = labels.to(rank)
                
                outputs = model(features)
                _, predicted = outputs.max(1)
                
                all_predicted.append(predicted.cpu())
                all_labels.append(labels.cpu())
        
        all_predicted = torch.cat(all_predicted).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        # 打印分类报告
        print(classification_report(all_labels, all_predicted, target_names=class_names))
        
        # 绘制混淆矩阵
        cm = confusion_matrix(all_labels, all_predicted)
        df_cm = pd.DataFrame(
            cm, 
            index=class_names,
            columns=class_names
        )
        
        def show_confusion_matrix(confusion_matrix):
            hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
            plt.ylabel("真实标签")
            plt.xlabel("预测标签")
            plt.savefig('confusion_matrix.png')
        
        show_confusion_matrix(df_cm)
    
    # 清理进程组
    cleanup()
 
def main():
    import argparse
    parser = argparse.ArgumentParser(description='DINOv2 多卡训练')
    parser.add_argument('--data-path', default='./data/trainData', help='数据集路径')
    parser.add_argument('--batch-size', default=20, type=int, help='总批量大小')
    parser.add_argument('--epochs', default=100, type=int, help='训练轮数')
    parser.add_argument('--lr', default=0.00001, type=float, help='学习率')
    parser.add_argument('--num-workers', default=4, type=int, help='数据加载进程数')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    args = parser.parse_args()
    
    # 设置GPU数量
    world_size = torch.cuda.device_count()
    print(f"发现 {world_size} 个GPU")
    
    # 启动多进程训练
    mp.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
def infer():
    model = DinoVisionTransformerClassifier().to(0)
    model.module.load_state_dict(torch.load('best_model.pth'))
        model.eval()
    
if __name__ == "__main__":
    main()