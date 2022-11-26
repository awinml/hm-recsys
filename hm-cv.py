import pandas as pd

#### Cross Validation Scheme
# The last day in the transaction dataframe is 2020-09-22. The public LB
# contains 1 week of transactions after this date. Therefore to create a local
# validation set, we can train on all transactions before 2020-9-15 and
# validate on the last week in train data.

# Creating a local cross validation scheme
train = pd.read_parquet("transactions_train.parquet")
train.t_dat = pd.to_datetime(train.t_dat)
train = train.loc[train.t_dat <= pd.to_datetime("2020-09-15")]

valid = pd.read_parquet("transactions_train.parquet")
valid.t_dat = pd.to_datetime(valid.t_dat)
test = valid.loc[valid.t_dat >= pd.to_datetime("2020-09-16")]
test = test.groupby("customer_id").article_id.apply(list).reset_index()
test = test.rename({"article_id": "prediction"}, axis=1)
test["prediction"] = test.prediction.apply(
    lambda x: " ".join(["0" + str(k) for k in x])
)
