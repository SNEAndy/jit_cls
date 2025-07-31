# python main.py --dataroot data/TrainSet --learn_rate 1e-5 --batch_size 24 --model resnet50d
# python main.py --dataroot data/TrainSet --learn_rate 1e-4 --batch_size 24 --model efficientnet_b3 --loss_fuction CE #f1:0.80 A:0.573
# python main.py --dataroot data/TrainSet --learn_rate 1e-4 --batch_size 20 --model efficientnet_b5 failed
# python main.py --dataroot data/TrainSet --learn_rate 1e-4 --batch_size 20 --model efficientnet_b4 --loss_fuction SmoothCE #f1:0.69
python main.py --dataroot data/TrainSet --learn_rate 1e-4 --batch_size 20 --model efficientnet_b4 --loss_fuction CE --label_path labels/train.txt --img_size 512#f1:0.96
python main.py --dataroot data/TrainSet --learn_rate 1e-5 --batch_size 20 --model swinT-base --loss_fuction CE --img_size 384 #0.8141 A:0.824 baseline best
python main.py --dataroot data/TrainSet --learn_rate 1e-5 --batch_size 20 --model swinT-base --loss_fuction weightCE --img_size 384 #0.8141 A:0.824 baseline
python main.py --dataroot data/TrainSet --learn_rate 2e-6 --batch_size 12 --model swinT-large --loss_fuction CE --img_size 384
python main.py --dataroot data/TrainSet --learn_rate 1e-5 --batch_size 20 --model swinT-base --loss_fuction CE --img_size 384 --diff_lr True
python main.py --dataroot data/TrainSet --learn_rate 1e-5 --batch_size 20 --model swinT-base --loss_fuction CE --label_path labels/train.txt --img_size 384#只针对3，4类做数据增强:放射变换、旋转、平移
python main.py --dataroot data/TrainSet --learn_rate 1e-5 --batch_size 20 --model swinT-base --loss_fuction SmoothCE --label_path labels/train.txt --img_size 384 #使用labelsmooth损失 A:0.784
python main.py --dataroot data/TrainSet --learn_rate 1e-5 --batch_size 20 --model swinT-base --loss_fuction SoftCE --mixup --img_size 384 #使用mixup数据增强方法 0.726
python main.py --dataroot data/TrainSet --learn_rate 1e-5 --batch_size 20 --model swinT-base --loss_fuction CE --cutmix --img_size 384 #使用mixup数据增强方法
python main.py --dataroot data/TrainSet --learn_rate 1e-5 --batch_size 20 --model vit-base --loss_fuction CE --img_size 224 #vit模型 #0.748
python main.py --dataroot data/TrainSet --learn_rate 1e-5 --batch_size 20 --model vit-base2 --loss_fuction CE --img_size 384 #vit模型2

python main.py --dataroot data/supTrainSet --learn_rate 1e-5 --batch_size 20 --model swinT-base --loss_fuction CE --img_size 384 --label_path labels/train_only_3_4.txt #增加数据集：BrEaST-Lesions_USG 3、4类
python main.py --dataroot data/supTrainSet --learn_rate 1e-5 --batch_size 20 --model dinov2 --loss_fuction CE --img_size 384 --label_path labels/train_only_3_4.txt #增加数据集：BrEaST-Lesions_USG 3、4类
python train_dino.py --dataroot data/supTrainSet --learn_rate 1e-5 --batch_size 20 --model dinov2 --loss_fuction CE --img_size 224 --label_path labels/trainval.txt#0.8141 A:0.824 baseline best