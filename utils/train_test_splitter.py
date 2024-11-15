import pandas as pd
import numpy as np
from variables import ROOT_DIR

np.random.seed(42)

if __name__ == '__main__':

    train = pd.read_csv(ROOT_DIR + '/data/train.tsv', sep='\t', header=None)

    new_train = train.iloc[np.random.permutation(len(train))[:-3301]]
    new_test = train.iloc[np.random.permutation(len(train))[-3301:]]

    new_train.to_csv(ROOT_DIR + '/data/new_train.tsv', index=False, header=False, sep='\t')
    new_test.to_csv(ROOT_DIR + '/data/new_test.tsv', index=False, header=False, sep='\t')
    print('Done')