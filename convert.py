import json
import os
import time


coco_path = '/SSDe/sangbeom_lee/mask-eee-rcnn/datasets/armbench/mix-object-tote/test_perturbed.json'

# get_categories
with open(coco_path, 'r') as f:
    coco = json.load(f)
# remove 'id' 1 category
print(coco['categories'])
coco['categories'] = [{'id': 2, 'name': 'object'}]

# remove 'id' 1 category from 'annotations'
for i in range(len(coco['annotations'])):
    if coco['annotations'][i]['category_id'] == 1:
        coco['annotations'].pop(i)
        break
    
# save new coco json
new_coco_path = '/SSDe/sangbeom_lee/mask-eee-rcnn/datasets/armbench/mix-object-tote/test_object_only.json'
with open(new_coco_path, 'w') as f:
    json.dump(coco, f)