import pandas as pd 

sample_submissions = ".\private_test_sample_submissions.csv"
output = ".\results.csv"

sample_df = pd.read_csv(sample_submissions)

df = pd.read_csv(output)
df['uuid'] = df['uuid'].str[:36]


aggregation_functions = {'assessment_result': 'mean'}
df_new = df.groupby(df['uuid']).aggregate(aggregation_functions)

# df_new.info()

final_df = pd.merge(sample_df, df_new, on='uuid', how='left')
del final_df["assessment_result_x"]
final_df.columns = ['uuid', 'assessment_result']

final_df['assessment_result'] = final_df['assessment_result'].fillna(0)

print(final_df.head())

final_df.to_csv('results.csv')