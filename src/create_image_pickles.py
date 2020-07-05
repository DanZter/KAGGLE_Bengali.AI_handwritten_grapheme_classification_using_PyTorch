import pandas as pd
import joblib
import glob
from tqdm import tqdm

"""if we load images using dataframes it will be slow,
hence we use image pickles to speed up the process.
You can also upload on GCP bucket and carry out TPU trainings.
We have 200,000 images and we are going to create 200,000 image pickles"""

if __name__ == "__main__":
    files = glob.glob("../input/train_*.parquet")   # list all the files using glob
    for f in files:
        df = pd.read_parquet(f)
        image_ids = df.image_id.values              # separated image ids from df
        df = df.drop("image_id", axis=1)
        image_array = df.values                     # separated pixel values from df

        for j, img_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            joblib.dump(image_array[j, :], f"../input/image_pickles/{img_id}.pkl")
            """this is going to save all images vectors in image pickle directory"""
