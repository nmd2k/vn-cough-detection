import pandas as pd 

sample_submissions =  "./private_test_sample_submissions.csv"
output = "./results.csv"

sample_df = pd.read_csv(sample_submissions)

df = pd.read_csv(output)
df['uuid'] = df['uuid'].str[:36]

# merge timestamped audio via "max" aggregration method

aggregation_functions = {'assessment_result': 'max'}
df_new = df.groupby(df['uuid']).aggregate(aggregation_functions)


# merge predicted submission with sample submission
# all non-predicted values are casted 0

result_df = pd.merge(sample_df, df_new, on='uuid', how='left')
del result_df["assessment_result_x"]
result_df.columns = ['uuid', 'assessment_result']

result_df['assessment_result'] = result_df['assessment_result'].fillna(0)

print(result_df.head())

result_df.to_csv('results.csv')