import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ ==" __main__":
    df = pd.read_csv("../input/train.csv")
    print(df.head())
    df.loc[:'kfold'] = -1                           # create a column called 'kfold' and fill it with -1

    df = df.sample(frac=1).reset_index(drop=True)
    """good idea to do shuffling the dataset.
    sample(): We use sample function to shuffle the dataset with frac = 1, meaning we will use the whole 
    fraction of rows.
    reset_index(drop=True): When we randomly shuffle the indexes also gets shuffled hence we need to drop the current
    indexes and reset it with new
    """
    X = df.image_id.values                          # image id column
    y = df[["grapheme_root" ,"vowel_diacritic" ,"consonant_diacritic"]].values

    mskf = MultilabelStratifiedKFold(n_splits=5)    # initializing a kfold class with 5 folds

    for fold, (trn_, val_) in enumerate(mskf.split(X, y)):  # iterating to get kfolds training and validation indexes
        print("TRAIN: ", trn_, "VAL: ", val_)       # to check
        df.loc[val, "kfold"] = fold                 # only for validation indexes we fill the kfold column with fold no.

    print(df.kfold.value_counts())                  # to check
    df.to_csv("..input/train_folds.csv", index = False) # putting back in input folder, no need of index column
    """we now have created folds"""





