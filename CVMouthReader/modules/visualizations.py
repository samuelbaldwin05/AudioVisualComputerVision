import pandas as pd


def csv_cleaner(input_csv, columns_to_drop):
    df = pd.read_csv(input_csv)
    df = df.drop(columns=columns_to_drop, errors='ignore')
    df.to_csv(input_csv, index=False)

def main():
    cols = ["word_type","error_type","song","annotations"]
    csv = "CVMouthReader/data/input/script.csv"
    csv_cleaner(csv, cols)

if __name__ == "__main__":
    main()