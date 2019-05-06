import pandas as pd

def get_UAS(data):
    gt_labels = data[6]
    predicted_labels = data[6]
    assert(len(predicted_labels) == len(gt_labels))
    all_labels = len(gt_labels)
    correct_labels = sum(predicted_labels == gt_labels)
    UAS = correct_labels / float(all_labels)
    return UAS

if __name__ == '__main__':
    data = pd.read_csv("test.conllu", sep='\t', header=None)
    print(get_UAS(data))