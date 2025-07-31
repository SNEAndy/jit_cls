import os
import shutil
# src="Dataset_BUSI_with_GT"
# labels=os.listdir(src)
# tgt="data/TrainSet/images/train"
# file_=open("./train.txt","w")
# for label in labels:
#     path_=os.path.join(src,label)
    
#     for img_name in os.listdir(path_):
#         if(label=="benign"):
#             label_=0
#         elif(label=="malignant"):
#             label_=4
#         elif(label=="normal"):
#             label_=5   
        
#         if "mask" not in img_name and "png" in img_name:
#             # os.remove(os.path.join(tgt,img_name))
#             new_name=img_name.strip().split(" ")
#             new_name=new_name[0]+new_name[1]
#             # shutil.copy(os.path.join(path_,img_name),os.path.join(tgt,new_name))
#             if label_==4:
#                 file_.write(new_name+" "+str(label_))
#                 file_.write("\n")
# file_.close() 
file_=open("./train.txt","r")
with open("data/TrainSet/labels/train.txt","a")as file1:
    lines=file_.readlines()
    for line in lines:
        file1.write(line)
        # file1.write('\n')
    file1.close()
file_.close()
            
        