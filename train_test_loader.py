import json

def load_train_test():
    d = json.load(open("train_test.json", "r"))
    return d['train'], d['test']
