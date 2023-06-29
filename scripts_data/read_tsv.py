import csv

path = '0629.tsv'

dataset, columns = [], {}
with open(path, 'r') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for i, line in enumerate(reader):
        # print(line)
        if i == 0:
            for i, column_name in enumerate(line):
                columns[column_name] = i 
            continue
      
        print(line)

print(columns)



         