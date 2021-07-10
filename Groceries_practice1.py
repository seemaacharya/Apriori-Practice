# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 13:39:33 2021

@author: DELL
"""

Groceries=[["egg","milk","kidney beans","potato"],
           ["bread","butter","egg","kidney beans"],
            ["egg","potato","butter"]]

import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
Groceries

te=TransactionEncoder()
te1=te.fit(Groceries).transform(Groceries)

df=pd.DataFrame(te1,columns=te.columns_)

from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets=apriori(df,min_support=0.6,use_colnames=True)
frequent_itemsets

from mlxtend.frequent_patterns import association_rules
rules=association_rules(frequent_itemsets,metric="confidence",min_threshold=0.7)
rules

rules1=rules[["antecedents","consequents","support","confidence","lift"]]
rules2=rules1[rules1["confidence"]>=1]
