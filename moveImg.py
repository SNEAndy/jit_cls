import os
import shutil
# file_=open("TrainSet/labels/trainval-Copy1.txt","r")
# lines=file_.readlines()
# src="augData/baseline/training/"
# for i in range(6):
#     os.makedirs("augData/baseline/training/"+str(i),exist_ok=True)
# for line in lines:
#     label=line.strip().split(" ")[1]
#     name=line.strip().split(" ")[0]
#     shutil.copy(os.path.join("TrainSet/images/train",name),os.path.join(src,label+"/"+name))
# print("finish")
# file_=open("./data/TrainSet/labels/trainval-ori.txt","r")
# with open("./data/supTrainSet/labels/trainval.txt","a+") as file2:
#     lines=file_.readlines()
#     dst="./data/supTrainSet/images/train"
#     for line in lines:
#         label=line.strip().split(" ")[1]
#         name=line.strip().split(" ")[0]
#         shutil.copy(os.path.join("./data/TrainSet/images/train",name),os.path.join(dst,name))
#         print(name,'and ',label)
#         # file2.write(name+" "+label)
#         # file2.write("\n")
# print("finish")
# file2=open("./data/supTrainSet/labels/train_only_3_4.txt",'w')
# file1=open("./data/supTrainSet/labels/train2.txt",'r')
# with open("./data/supTrainSet/labels/trainval.txt",'r') as file_:
#     lines=file_.readlines()
#     for line in lines:
        
#         file2.write(line)
#         # file2.write("\n")
# lines=file1.readlines()
# for line in lines:
#     label=line.strip().split(" ")[1]
#     if label=="3" or label=="4":
#         file2.write(line)
# file2.close()
# file1.close()
root="./data/trainData/val"
data_path="./data/supTrainSet/images/train"
with open("./data/supTrainSet/labels/val.txt",'r') as file_:
    lines=file_.readlines()
    for line in lines:
        label=line.strip().split(" ")[1]
        name=line.strip().split(" ")[0]
        os.makedirs(os.path.join(root,label),exist_ok=True)
        shutil.copy(os.path.join(data_path,name),os.path.join(root,label+"/"+name))
# print(finish)