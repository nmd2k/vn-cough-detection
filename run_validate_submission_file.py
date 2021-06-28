import argparse
import pandas as pd
from pandas import DataFrame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_file", type=str, help="Path to submission file")
    parser.add_argument("--sample_file", type=str, help="Path to sample submission file")

    args = parser.parse_args()

    df_sample = pd.read_csv(args.sample_file)
    df_submission = pd.read_csv(args.submission_file)

    if len(df_sample["uuid"]) != len(df_submission["uuid"]):
        print("Invalid size")
        return

    submission_ids = {}

    for idx, uuid in enumerate(df_submission["uuid"]):
        submission_ids[uuid] = df_submission["assessment_result"][idx]
    
    for idx, uuid in enumerate(df_sample["uuid"]):
        if not uuid in submission_ids:
            print("Invalid uuid", uuid)
            return

    print("All ids are valid. Aligning to sample file")

    data = {
        "uuid": [],
        "assessment_result": []
    }

    for idx, uuid in enumerate(df_sample["uuid"]):
        data["uuid"].append(uuid)
        data["assessment_result"].append(submission_ids[uuid])

    df = DataFrame.from_dict(data)
    df.to_csv('results.csv', index=False)

    print("Done. Result saved")

if __name__ == "__main__":
    main()