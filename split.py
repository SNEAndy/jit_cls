import pandas as pd
import shutil
import os 
excel_file_path="BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
sheet_name='BrEaST-Lesions-USG clinical dat'
data_path='BrEaST-Lesions_USG-images_and_masks'
labels="./data/supTrainSet/labels/trainval.txt"
df=pd.read_excel(excel_file_path, sheet_name)
# 提取 'age' 和 'name' 两列
print("提取 'age' 和 'name' 两列")
columns_data = df[['BIRADS']]
with open(labels,'w')as file:
    for excel_row_number, row in df.iterrows():
        id=excel_row_number+1
        try:
            label=int(row['BIRADS'])
        except:
            label=row['BIRADS']
        if(label==2):
            if id<10:
                img_name='case00'+str(id)+'.png'
            elif id>=10 and id <100:
                img_name='case0'+str(id)+'.png'
            elif id>=100:
                img_name='case'+str(id)+'.png'
            shutil.copy(os.path.join(data_path,img_name),os.path.join("./data/supTrainSet/images",img_name))
            file.write(img_name+" "+str(0))
            file.write("\n")
        elif(label=='4b'):
            id=excel_row_number+1
            if id<10:
                img_name='case00'+str(id)+'.png'
            elif id>=10 and id <100:
                img_name='case0'+str(id)+'.png'
            elif id>=100:
                img_name='case'+str(id)+'.png'
            shutil.copy(os.path.join(data_path,img_name),os.path.join("./data/supTrainSet/images",img_name))
            file.write(img_name+" "+str(3))
            file.write("\n")
        elif(label=='4a'):
            id=excel_row_number+1
            if id<10:
                img_name='case00'+str(id)+'.png'
            elif id>=10 and id <100:
                img_name='case0'+str(id)+'.png'
            elif id>=100:
                img_name='case'+str(id)+'.png'
            shutil.copy(os.path.join(data_path,img_name),os.path.join("./data/supTrainSet/images",img_name))
            file.write(img_name+" "+str(2))
            file.write("\n")
        elif(label==3):
            id=excel_row_number+1
            if id<10:
                img_name='case00'+str(id)+'.png'
            elif id>=10 and id <100:
                img_name='case0'+str(id)+'.png'
            elif id>=100:
                img_name='case'+str(id)+'.png'
            shutil.copy(os.path.join(data_path,img_name),os.path.join("./data/supTrainSet/images",img_name))
            file.write(img_name+" "+str(1))
            file.write("\n")
        elif(label=='4c' or label==5):
            id=excel_row_number+1
            if id<10:
                img_name='case00'+str(id)+'.png'
            elif id>=10 and id <100:
                img_name='case0'+str(id)+'.png'
            elif id>=100:
                img_name='case'+str(id)+'.png'
            shutil.copy(os.path.join(data_path,img_name),os.path.join("./data/supTrainSet/images",img_name))
            file.write(img_name+" "+str(4))
            file.write("\n")
        elif(label==1):
            id=excel_row_number+1
            if id<10:
                img_name='case00'+str(id)+'.png'
            elif id>=10 and id <100:
                img_name='case0'+str(id)+'.png'
            elif id>=100:
                img_name='case'+str(id)+'.png'
            shutil.copy(os.path.join(data_path,img_name),os.path.join("./data/supTrainSet/images",img_name))
            file.write(img_name+" "+str(5))
            file.write("\n")
        else:
            print(excel_row_number+1,"and", row['BIRADS'])
        
        
        
    

