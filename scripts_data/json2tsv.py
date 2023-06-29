
import json 
import pandas as pd

json_file = "dataset/0629.json"
f = open(json_file, 'r')
data = json.load(f)
# print(data)
labels = []
txts = []

for line in data:
    print('--', line)
    output = line['output']
    txt = line['input']
    if output[0] == '是':
        labels.append(1)
    else:
        labels.append(0)
    txts.append(txt)

result = pd.DataFrame({"label": labels, "text_a": txts})
result.to_csv(json_file.replace('json', 'tsv'), sep='\t', index=False, header=True)