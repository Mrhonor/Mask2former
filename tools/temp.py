import os

with open('temp.log', 'r') as f:
    lines = f.readlines()
    names = []
    for l in lines:
        lists = l.strip().split(':')
        if len(lists)==1:
            names.append(lists[0])
        else:
            names.append(lists[1])
    print(names)
        