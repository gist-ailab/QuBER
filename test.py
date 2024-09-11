import json

gt_json_path = '/SSDe/sangbeom_lee/mask-eee-rcnn/datasets/armbench/mix-object-tote/test_panoptic_perturbed.json'

with open(gt_json_path, 'r') as f:
    gt_json = json.load(f)
    
# leave only sinle  image
images = gt_json['images']
for gt_ann in gt_json['annotations']:
    if gt_ann['image_id'] == images[0]['id']:
        gt_json['annotations'] = [gt_ann]
        break
gt_json['images'] = [images[0]]
# wirte to new json file
with open('/SSDe/sangbeom_lee/mask-eee-rcnn/datasets/armbench/mix-object-tote/test_panoptic_perturbed_single.json', 'w') as f:
    json.dump(gt_json, f)
print(gt_json.keys())   