import os

DATA_DIR = '/m2-data/rushuai.liu/faceQuality'
train_data_dirs=[
        'ms1mv2',
]

out_file = open('face_train_ms1mv2.txt', 'w')

person_count = 0

for root_dir in train_data_dirs:
    root_dir = os.path.join(DATA_DIR, root_dir)
    if not os.path.isdir(root_dir):
        continue
    for person_dir in os.listdir(root_dir):
        person_dir = os.path.join(root_dir, person_dir)
        count = 0
        for filename in os.listdir(person_dir):
            filename = os.path.join(person_dir, filename)
            if filename.endswith(('.png','jpg','.bmp')) and os.path.isfile(filename):
                count+=1
                print(os.path.abspath(filename)+';'+str(person_count), file=out_file)

        if count > 0:
            person_count+=1


out_file.close()
