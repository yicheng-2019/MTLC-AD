import os
import csv

root_path = '/export/space0/qiu-y/dataset/UCFCrime/UCFClipFeatures/'
# txt = 'list/Anomaly_Train.txt'
# txt = "Anomaly_Train.txt" # 训练集文件名列表
txt = "Anomaly_Test.txt"
files = list(open(txt))
normal = []
count = 0

# with open('list/ucf_CLIP_rgb.csv', 'w+') as f:  ## the name of feature list
# with open('ucf_CLIP_rgb_yanai.csv', 'w+') as f:  ## the name of feature list
with open('ucf_CLIP_rgbtest_yanai.csv', 'w+') as f:
    writer = csv.writer(f)
    writer.writerow(['path', 'label'])
    for file in files:
        # print(file) 
        file = file.strip()  # 去除换行符，否则最后一行有问题
        filename = root_path + file[:-4] + '__0.npy'  # 去除换行符后，每行少一个字符，所以从-5变成-4
        # print(filename)
        label = file.split('/')[0]
        if os.path.exists(filename):
            # print(f'{filename} 存在')
            if 'Normal' in label:
                #continue
                filename = filename[:-5]
                # for i in range(0, 10, 1):  # train list时使用
                    # normal.append(filename + str(i) + '.npy') # train list时使用
                normal.append(filename + "0" + '.npy') # test list时使用
            else:
                filename = filename[:-5]
                # for i in range(0, 10, 1):  # train list时使用
                    # writer.writerow([filename + str(i) + '.npy', label])  # train list时使用
                writer.writerow([filename + "0" + '.npy', label])  # test list时使用
        else:
            count += 1
            # print(file)
            print(filename)
            
    for file in normal:
        writer.writerow([file, 'Normal'])

print(count)