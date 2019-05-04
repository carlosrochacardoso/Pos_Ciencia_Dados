import mincemeat
import glob
import csv

text_files = glob.glob('arquivos/*')

def file_contents(file_name):
    f = open(file_name)
    try:
        return f.read()
    finally:
        f.close()

source = dict((file_name, file_contents(file_name))for file_name in text_files)


def mapfn(k,v):
 
    from string import punctuation
    from stopwords import allStopWords

    print 'map ' + k

    for line in v.splitlines():
                
        autores = line.split(":::")[1]
        palavras = line.split(":::")[2]

        palavras = palavras.translate(None, punctuation)
        palavras = palavras.lower()

        for palavra in palavras.split():
            if(palavra not in allStopWords):
                for autor in autores.split("::"):
                    yield autor + '::' + palavra, 1


def reducefn(k, v):
    print 'reduce ' + k
    return sum(v)

s = mincemeat.Server()

s.datasource = source
s.mapfn = mapfn
s.reducefn = reducefn

results = s.run_server(password='trabalho')

w = csv.writer(open('result.csv','w'))

w.writerow(['autor','palavra','total'])

for k,v in results.items():
    autor = k.split('::')[0]
    palavra = k.split('::')[1]
    w.writerow([autor,palavra,v])
