#start, stop and unk in vocab; top 30k

passages = []
questions = []
answers = []
vocab_dict = {}

for tuple in jsonFile():
    #Extarct ans, quest, passages

    #O/p; lsit of 3 things

