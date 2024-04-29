import sys
import csv
import torch
# import pickle

# OBJECTS365_ANN_PATH = 'datasets/objects365/annotations/objects365v2_val_0422.json'


def csvread(file): 
    with open(file, 'r', encoding='utf-8') as f:
        csv_f = csv.reader(f)
        data = []
        for row in csv_f:
            data.append(row)
    return data


# def get_unified_label_map(unified_label, cats):
#     '''
#     Inputs:

#     Return:
#         unified_label_map: dict of dict
#             (dataset (string), cat_id (int)) --> unified_id (int)
#     '''
#     unified_label_map = {}
#     for dataset in cats:
#         unified_label_map[dataset] = {}
#         col = COL[dataset]
#         table_names = [x[col].lower().strip() for x in unified_label[1:]]
#         cat_ids = sorted([x['id'] for x in cats[dataset]])
#         id2contid = {x: i for i, x in enumerate(cat_ids)}
#         for cat_info in cats[dataset]:
#             if dataset != 'oid':
#                 cat_name = cat_info['name']
#             else:
#                 cat_name = cat_info['freebase_id']
#             cat_id = id2contid[cat_info['id']]
#             if cat_name.lower().strip() in table_names:
#                 unified_id = table_names.index(cat_name.lower().strip())
#                 unified_label_map[dataset][cat_id] = unified_id
#             else:
#                 print('ERROR!', cat_name, 'not find!')
#         print(dataset, 'OK')
#     return unified_label_map

if __name__ == '__main__':
    unified_label_path = sys.argv[1]
    unified_label = csvread(unified_label_path)
    cats = {}
    print('Loading')
    unified_label_num = len(unified_label)
    print(f'unified_label_num:{unified_label_num}')
    total_cats = 0
    n_datasets = 7
    n_cats = [19, 64, 37, 19, 26, 150, 133]
    # n_cats = [19, 64, 19, 26, 26]
    sta_cats = []
    for i in range(n_datasets):
        sta_cats.append(total_cats)
        total_cats += n_cats[i]
        
    # COL = {'city': 1, 'ade': 2, 'coco': 3}
    target_bi = torch.zeros(total_cats, unified_label_num)
    for i, lb in enumerate(unified_label):
        for j, id in enumerate(lb[1:]):
            if id != ' ':
                target_bi[sta_cats[j]+int(id)][i] = 1
            # else:
            #     print("!")

    torch.save(target_bi, f'init_adj_7_datasets_{unified_label_num}.pt')
