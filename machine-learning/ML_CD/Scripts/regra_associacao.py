# -*- coding: utf-8 -*-
"""

@author: Hugo
"""

from scipy.sparse import lil_matrix  # other types would convert to LIL anyway
import pandas as pd
from orangecontrib.associate.fpgrowth import *
import Orange


data = Orange.data.Table('zoo')
X, mapping = OneHot.encode(data, include_class=True)

# Calcula os itemsets frequentes
itemsets = dict(frequent_itemsets(X, .4))
print(list(itemsets))
class_items = {item
               for item, var, _ in OneHot.decode(mapping, data, mapping)
               if var is data.domain.class_var}
print(sorted(class_items))
print(data.domain.class_var.values)




rules = [(P, Q, supp, conf)
         for P, Q, supp, conf in association_rules(itemsets, .8)
         if len(Q) == 1 and Q & class_items]
    
print(len(rules))
print(rules)

names = {item: '{}={}'.format(var.name, val)
         for item, var, val in OneHot.decode(mapping, data, mapping)}
for ante, cons, supp, conf in rules[:5]:
    print(', '.join(names[i] for i in ante), '-->',
          names[next(iter(cons))],
          '(supp: {}, conf: {})'.format(supp, conf))
    