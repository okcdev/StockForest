import csv
csv_file = csv.reader(open('../Data/demo.csv','r'))
for stu in csv_file:
    print(stu)