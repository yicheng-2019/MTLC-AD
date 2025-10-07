import os
import csv

root_path = '/aidata/qiuyc/datasets/UCF_Crime/UCFClipFeatures/'
# txt = 'list/Anomaly_Train.txt'
# txt = "Anomaly_Train.txt"
txt = "Anomaly_Test.txt"
files = list(open(txt))
normal = []
count = 0

# with open('list/ucf_CLIP_rgb.csv', 'w+') as f:  ## the name of feature list
# with open('ucf_CLIP_rgb.csv', 'w+') as f:  ## the name of feature list
with open('ucf_CLIP_rgbtest_L40S.csv', 'w+') as f:
    writer = csv.writer(f)
    writer.writerow(['path', 'label'])
    for file in files:
        filename = root_path + file[:-5] + '__0.npy'
        label = file.split('/')[0]
        if os.path.exists(filename):
            if 'Normal' in label:
                #continue
                filename = filename[:-5]
                # for i in range(0, 10, 1):  # train list时使用
                    # normal.append(filename + str(i) + '.npy') # train list时使用
                normal.append(filename + "5" + '.npy') # test list时使用
            else:
                filename = filename[:-5]
                # for i in range(0, 10, 1):  # train list时使用
                    # writer.writerow([filename + str(i) + '.npy', label])  # train list时使用
                writer.writerow([filename + "5" + '.npy', label])  # test list时使用
        else:
            count += 1
            print(filename)
            
    for file in normal:
        writer.writerow([file, 'Normal'])

print(count)