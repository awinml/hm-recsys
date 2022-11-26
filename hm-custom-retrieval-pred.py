import os
import gc
import shelve
import cudf

from utils import io as h_io, sub as h_sub, cv as h_cv, fe as h_fe
from utils import modeling as h_modeling, candidates as h_can, pairs as h_pairs


# Load and convert data

customers, transactions, articles = h_io.load_data(
    files=["customers.csv", "transactions_train.csv", "articles.csv"]
)

index_to_id_dict_path = h_fe.reduce_customer_id_memory(customers, [transactions])
transactions["week_number"] = h_fe.day_week_numbers(transactions["t_dat"])
transactions["t_dat"] = h_fe.day_numbers(transactions["t_dat"])

pairs_per_item = 5

week_number_pairs = {}
for week_number in [96, 97, 98, 99, 100, 101, 102, 103, 104]:
    print(f"Creating pairs for week number {week_number}")
    week_number_pairs[week_number] = h_pairs.create_pairs(
        transactions, week_number, pairs_per_item, verbose=False
    )


def create_candidates_with_features_df(t, c, a, customer_batch=None, **kwargs):
    # Splitting cv
    features_df, label_df = h_cv.feature_label_split(
        t, kwargs["label_week"], kwargs["feature_periods"]
    )

    # Converting relative day_number
    features_df["t_dat"] = h_fe.how_many_ago(features_df["t_dat"])
    features_df["week_number"] = h_fe.how_many_ago(features_df["week_number"])

    # Pull out the cv week
    article_pairs_df = week_number_pairs[kwargs["label_week"] - 1]

    # Check if we can limit customers
    if len(label_df) > 0:
        customers = label_df["customer_id"].unique()
    elif customer_batch is not None:
        customers = customer_batch
    else:
        customers = None

    # Creating candidates (and adding features)

    features_db = shelve.open("features_db")

    # Creating candidate (and saving features created)
    (
        recent_customer_cand,
        features_db["customer_article"],
    ) = h_can.create_recent_customer_candidates(
        features_df,
        kwargs["ca_num_weeks"],
        customers=customers,
    )

    (
        cust_last_week_cand,
        cust_last_week_pair_cand,
        features_db["clw"],
        features_db["clw_pairs"],
    ) = h_can.create_last_customer_weeks_and_pairs(
        features_df,
        article_pairs_df,
        kwargs["clw_num_weeks"],
        kwargs["clw_num_pair_weeks"],
        customers=customers,
    )

    _, features_db["popular_articles"] = h_can.create_popular_article_cand(
        features_df,
        c,
        a,
        kwargs["pa_num_weeks"],
        kwargs["hier_col"],
        num_candidates=kwargs["num_recent_candidates"],
        num_articles=kwargs["num_recent_articles"],
        customers=customers,
    )
    age_bucket_can, _, _ = h_can.create_age_bucket_candidates(
        features_df,
        c,
        kwargs["num_age_buckets"],
        articles=kwargs["num_recent_articles"],
        customers=customers,
    )

    cand = [
        recent_customer_cand,
        cust_last_week_cand,
        cust_last_week_pair_cand,
        age_bucket_can,
    ]
    cand = cudf.concat(cand).drop_duplicates()
    cand = cand.sort_values(["customer_id", "article_id"]).reset_index(drop=True)

    del (
        recent_customer_cand,
        cust_last_week_cand,
        cust_last_week_pair_cand,
        age_bucket_can,
    )

    cand = h_can.filter_candidates(cand, t, **kwargs)

    # Creating other features
    h_fe.create_cust_hier_features(features_df, a, kwargs["hier_cols"], features_db)
    h_fe.create_price_features(features_df, features_db)
    h_fe.create_cust_features(c, features_db)
    h_fe.create_article_cust_features(features_df, c, features_db)
    h_fe.create_lag_features(features_df, a, kwargs["lag_days"], features_db)
    h_fe.create_rebuy_features(features_df, features_db)
    h_fe.create_cust_t_features(features_df, a, features_db)
    h_fe.create_art_t_features(features_df, features_db)

    del features_df

    # Another filter at the end, for the ones that didn't get filtered earlier
    if customers is not None:
        cand = cand[cand["customer_id"].isin(customers)]

    # Report recall/precision of candidates
    if kwargs["cv"]:
        ground_truth_candidates = label_df[
            ["customer_id", "article_id"]
        ].drop_duplicates()
        h_cv.report_candidates(cand, ground_truth_candidates)
        del ground_truth_candidates

    # Adding features to candidates
    cand_with_f_df = h_can.add_features_to_candidates(cand, features_db, c, a)

    # Manually adding article features (
    for article_col in kwargs["article_columns"]:
        art_col_map = a.set_index("article_id")[article_col]
        cand_with_f_df[article_col] = cand_with_f_df["article_id"].map(art_col_map)

    # Limiting features
    if kwargs["selected_features"] is not None:
        cand_with_f_df = cand_with_f_df[
            ["customer_id", "article_id"] + kwargs["selected_features"]
        ]

    features_db.close()
    os.remove("features_db.bak"), os.remove("features_db.dir"), os.remove(
        "features_db.dat"
    )

    assert len(cand) == len(
        cand_with_f_df
    ), "seem to have duplicates in the feature dfs"
    del cand

    return cand_with_f_df, label_df


def calculate_model_score(ids_df, preds, truth_df):
    predictions = h_modeling.create_predictions(ids_df, preds)
    true_labels = h_cv.ground_truth(truth_df).set_index("customer_id")["prediction"]
    score = round(h_cv.comp_average_precision(true_labels, predictions), 5)

    return score


cv_params = {
    "cv": True,
    "feature_periods": 105,
    "label_week": 104,
    "index_to_id_dict_path": index_to_id_dict_path,
    "pairs_file_version": "_v3_5_ex",
    "num_recent_candidates": 36,
    "num_recent_articles": 12,
    "hier_col": "department_no",
    "ca_num_weeks": 3,
    "clw_num_weeks": 12,
    "clw_num_pair_weeks": 2,
    "pa_num_weeks": 1,
    "num_age_buckets": 4,
    "filter_recent_art_weeks": 1,
    "filter_num_articles": None,
    "lag_days": [1, 3, 14, 30],
    "article_columns": ["index_code"],
    "hier_cols": [
        "department_no",
        "section_no",
        "index_group_no",
        "index_code",
        "product_type_no",
        "product_group_name",
    ],
    "selected_features": None,
    "lgbm_params": {"n_estimators": 200, "num_leaves": 20},
    "log_evaluation": 10,
    "early_stopping": 20,
    "eval_at": 12,
    "save_model": True,
    "num_concats": 5,
}
sub_params = {
    "cv": False,
    "feature_periods": 105,
    "label_week": 105,
    "index_to_id_dict_path": index_to_id_dict_path,
    "pairs_file_version": "_v3_5_ex",
    "num_recent_candidates": 60,
    "num_recent_articles": 12,
    "hier_col": "department_no",
    "ca_num_weeks": 3,
    "clw_num_weeks": 12,
    "clw_num_pair_weeks": 2,
    "pa_num_weeks": 1,
    "num_age_buckets": 4,
    "filter_recent_art_weeks": 1,
    "filter_num_articles": None,
    "lag_days": [1, 3, 14, 30],
    "article_columns": ["index_code"],
    "hier_cols": [
        "department_no",
        "section_no",
        "index_group_no",
        "index_code",
        "product_type_no",
        "product_group_name",
    ],
    "selected_features": None,
    "lgbm_params": {
        "n_estimators": 150,
        "num_leaves": 20,
    },
    "log_evaluation": 10,
    "eval_at": 12,
    "prediction_models": ["model_104", "model_105"],
    "save_model": True,
    "num_concats": 5,
}


cand_features_func = create_candidates_with_features_df
scoring_func = calculate_model_score

cv_weeks = [104]
results = h_modeling.run_all_cvs(
    transactions,
    customers,
    articles,
    cand_features_func,
    scoring_func,
    cv_weeks=cv_weeks,
    **cv_params,
)

gc.collect()
h_modeling.full_sub_train_run(
    transactions, customers, articles, cand_features_func, scoring_func, **sub_params
)
predictions = h_modeling.full_sub_predict_run(
    transactions, customers, articles, cand_features_func, **sub_params
)


sub = h_sub.create_sub(customers["customer_id"], predictions, index_to_id_dict_path)
sub.to_csv("dev_submission.csv", index=False)
