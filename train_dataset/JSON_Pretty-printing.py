import json

'''
Make JSON to be Prettier and Readable
'''
# Train.json
file_path = "./densepose_coco_2014_train.json"
json_data = {}
with open(file_path, "r") as json_file:
    json_data = json.load(json_file)

with open(file_path, 'w') as outfile:
    json.dump(json_data, outfile, indent = 4, sort_keys = True)


# Val.json
file_path = "./densepose_coco_2014_minival.json"
json_data = {}
with open(file_path, "r") as json_file:
    json_data = json.load(json_file)

with open(file_path, 'w') as outfile:
    json.dump(json_data, outfile, indent = 4, sort_keys = True)