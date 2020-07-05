import albumentations           # augmentation library
import joblib
import pandas as pd
import numpy as np
from PIL import Image

class BengaliDatasetTrain:
    def __init__(self, folds, img_height, img_width, mean, std):
        df = pd.read_csv("../input/train_folds.csv")
        df = df[["image_ids", "grapheme_root", "vowel_diacritic", "consonant_diacritic", "kfold"]]

        df = df[df.kfold.isin(folds)].reset_index(drop=True)
        #  we subset the dataset according to the required folds. folds variable can be a list of folds for training

        self.image_ids = df.image_ids.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values
        # converted all columns into numpy array

        if len(folds) == 1:                                     # Augmentations
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.Normalize(mean, stf, always_apply=True)
            ])
        else:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.Normalize(mean, stf, always_apply=True),
                albumentations.ShiftScaleRotate(shift_limit=0.0625, 
                                                scale_limit=0.1,
                                                rotate_limit=5,
                                                p=0.9)
            ])

        

    def __len__(self):                                              # returns the length of the dataset
        return len(self.image_ids)

    def __getitem__(Self):                                          # gets item indexes
        image = joblib.load(f"../input/image_pickles/{self.image_ids[item]}.pkl")   # load image pickles
        image = image.reshape(137, 236).astype(float)               # reshape image from 1D vector to 137x236 (2D)
        image = Image.fromarray(image).convert("RGB")               # converts numpy array to PIL image of RGB format
        """ RGB because all pretrained models are trained on RGB formats"""

        image = self.aug(image=np.array(image))["image"]            # image output taken from the augmentation output
        image = np.trasnspose(image, (2, 0, 1)).astype(np.float32)  # transpose image to fit torchvision models, exchange RGB channels (2,0,1).check how torchvision models expects the images
        return {
            'image':torch.trensor(image, dtype=torch.float),
            'grapheme_root': torch.tensor(self.grapheme_root[item], dtype=torch.long),    # extract items from grapheme root
            'vowel_diacritic': torch.tensor(self.vowel_diacritic[item], dtype=torch.long),
            'consonant_diacritic': torch.tensor(self.consonant_diacritic[item], dtype=torch.long),
                                            
        }



