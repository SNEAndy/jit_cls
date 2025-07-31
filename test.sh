python main.py --dataroot data/TestSetA --testonly --loadfrom model_save/202506111142/epoch_62_best.pkl --model efficientnet_b3 --img_size 512 # A:0.573
python main.py --dataroot data/TestSetA --testonly --loadfrom model_save/202506121225/best.pkl --model swinT-base --img_size 384 #0.784
python main.py --dataroot data/TestSetA --testonly --loadfrom model_save/202506131033/best.pkl --model vit-base  --img_size 224 #0.748
python main.py --dataroot data/TestSetA --testonly --loadfrom model_save/202506121225/best.pkl --model swinT-base --img_size 384 #0.784
python main.py --dataroot data/TestSetA --testonly --loadfrom model_save/202506201124/best.pkl --model swinT-base --img_size 384 --loss_fuction weightCE#0.8125
python main.py --dataroot data/TestSetA --testonly --loadfrom model_save/202506201339/best.pkl --model swinT-base --img_size 384 --loss_fuction weightCE#0.8125

python main.py --dataroot data/TestSetA --testonly --loadfrom model_save/202506270947/best.pkl --model swinT-base --img_size 384

python main.py --dataroot data/TestSetA --testonly --loadfrom model_save/202507281657/best.pkl --model swinT-base --img_size 384 #0.819
python main.py --dataroot data/TestSetA --testonly --loadfrom model_save/202507290916/best.pkl --model swinT-base --img_size 384
python main.py --dataroot data/TestSetA --testonly --loadfrom model_save/202507291024/best_f1.pkl --model swinT-base --img_size 384 #0.79

python main.py --dataroot data/TestSetA --testonly --loadfrom model_save/202507291024/best_acc.pkl --model swinT-base --img_size 384 #0.79