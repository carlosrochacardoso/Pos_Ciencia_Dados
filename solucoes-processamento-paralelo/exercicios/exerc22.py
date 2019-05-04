import mincemeat
import glob
import csv

text_files = glob.glob('c:\\TEMP\\exerc\\Join\\*')

def file_contents(file_name):
    f = open(file_name)
    try:
        return f.read()
    finally:
        f.close()

source = dict((file_name, file_contents(file_name))for file_name in text_files)

def mapfn(k,v):
    print 'map ' + k
    for line in v.splitlines():
        if k == 'c:\\TEMP\\exerc\\Join\\2.2-vendas.csv':
            yield line.split(';')[0], 'vendas' + ':' + line.split(';')[5]
        if k == 'c:\\TEMP\\exerc\\Join\\2.2-filiais.csv':
            yield line.split(';')[0], 'filial' + ':' + line.split(';')[1]
 
def reducefn(k, v):
    print 'reduce ' + k
    return v

def reducefn2(k, v):
    print 'reduce ' + k
    
    total = 0 
    for index, item in enumerate(v):
        if item.split(":")[0] == 'vendas':
            total = int(item.split(':')[1]) + total
        if item.split(":")[0] == 'filial':
            NomeFilial = item.split(':')[1]
    
    l = list()
    l.append(NomeFilial + ' , ' + str(total))
    
    return l

s = mincemeat.Server()

s.datasource = source
s.mapfn = mapfn
s.reducefn = reducefn2

results = s.run_server(password='oioioi')

w = csv.writer(open('C:\\TEMP\\exerc\\result_join_2.csv','w'))
for k,v in results.items():
    w.writerow([k,str(v).replace("[","").replace("]","").replace("'","").replace(' ','')])