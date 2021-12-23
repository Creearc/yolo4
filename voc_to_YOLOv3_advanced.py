import xml.etree.ElementTree as ET
from os import getcwd
import os
import argparse
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("-T", "--train_path", type=str, default='dataset/train/')
ap.add_argument("-V", "--test_path", type=str)
ap.add_argument("-P", "--test_part", type=float, default=0.2)
ap.add_argument("-N", "--name", type=str, default='dataset')
ap.add_argument("-O", "--output_path", type=str)
args = vars(ap.parse_args())

print(args['test_path'])

check_path = lambda x : x if x[-1] == '/' else '{}/'.format(x)

def clear_file(file):
    f = open(file, 'w')
    f.close()

train_path = check_path(args['train_path'])
test_path = None if args['test_path'] is None else check_path(args['test_path'])
output_path = '' if args['output_path'] is None else check_path(args['output_path'])


train_file = '{}{}_train.txt'.format(output_path, args['name'])
test_file = '{}{}_test.txt'.format(output_path, args['name'])
classes_file = '{}{}_classes.txt'.format(output_path, args['name'])

clear_file(train_file)
if not(test_path is None):
    clear_file(test_file)
clear_file(classes_file)

CLS = os.listdir(train_path)
classes = [train_path + CLASS for CLASS in CLS]
wd = getcwd()


def test(fullname, output_file):
    bb = ""
    in_file = open(fullname)
    tree = ET.parse(in_file)
    root = tree.getroot()
    for i, obj in enumerate(root.iter('object')):
        cls = fullname.split('/')[-2]
        cls_id = CLS.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        if b != (0, 0, 0, 0):
            bb = '{} {},{}'.format(bb, ",".join([str(a) for a in b]), cls_id)

    if bb != "":
        list_file = open(output_file, 'a')
        file_string = str(fullname)[:-4]+'.jpg'+bb+'\n'
        list_file.write(file_string)
        list_file.close()



for CLASS in classes:
    if test_path is None:
        for filename in os.listdir(CLASS):
            if not filename.endswith('.xml'):
                continue
            fullname = '{}{}/{}'.format(train_path, CLASS.split('/')[-1], filename)
            test(fullname, train_file)
    else:
        files = os.listdir(CLASS)
        np.random.shuffle(files)
        edge_file = int(len(files) * args['test_part'])
        train_files = files[edge_file:]
        test_files =  files[:edge_file]

        for filename in train_files:
            if not filename.endswith('.xml'):
                continue
            fullname = '{}{}/{}'.format(train_path, CLASS.split('/')[-1], filename)
            test(fullname, train_file)

        for filename in test_files:
            if not filename.endswith('.xml'):
                continue
            fullname = '{}{}/{}'.format(test_path, CLASS.split('/')[-1], filename)
            test(fullname, test_file)

        
for CLASS in CLS:
    list_file = open(classes_file, 'a')
    file_string = str(CLASS)+"\n"
    print(file_string)
    list_file.write(file_string)
    list_file.close()
