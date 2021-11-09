import json
import os

'''
Make JSON to be Prettier and Readable
'''

data_root = "./"
train_json = "densepose_coco_2014_train.json"
valid_json = "densepose_coco_2014_minival.json"

def make_path(json_file):
    file_path = os.path.join(data_root, json_file)
    
    return file_path

def pretty_printing(json_file):
    json_data = {}
    file_path = make_path(json_file)

    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)

    with open(file_path, 'w') as outfile:
        json.dump(json_data, outfile, indent = 4, sort_keys = True)

def main():
    pretty_printing(train_json)
    pretty_printing(valid_json)

if __name__ == "__main__":
    main()