import os
import torch
import shutil
label0,label1,label2,label3,label4,label5=0,0,0,0,0,0
with open('data/TrainSet/labels/trainval.txt',"a") as file:
    src_="data/augData/medaugment/training"
    for img in os.listdir(src_+'/3/'):
        if img.strip().split(".")[1]=="jpg":
            # shutil.copy(src_+'/3/'+img,"data/TrainSet/images/train/"+img)
            # print(img)
            file.write(img+" "+'3')
            file.write("\n")
    for img in os.listdir(src_+'/4'):
        if img.strip().split(".")[1]=="jpg":
            # shutil.copy(src_+'/4/'+img,"data/TrainSet/images/train/"+img)
            # print(img)
            file.write(img+" "+'4')
            file.write("\n")     
#     lines=file.readlines()
#     for line in lines:
#         label=int(line.split(" ")[-1])
#         if label == 0:
#             label0+=1
#         elif label == 1:
#             label1+=1
#         elif label == 2:
#             label2+=1
#         elif label == 3:
#             label3+=1
#         elif label == 4:
#             label4+=1
#         elif label == 5:
#             label5+=1
# print(label0,label1,label2,label3,label4,label5)
# a=torch.as_tensor([label0/1500,label1/1500,label2/1500,label3/1500,label4/1500,label5/1500]).cuda()
# print(a)