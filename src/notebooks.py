import pandas as pd

if __name__ == "__main__":
    df = pd.read_parquet("../input/train_image_data_0.parquet")

    print(df.head())