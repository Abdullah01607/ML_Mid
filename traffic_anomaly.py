# =====================================================================
# Website Traffic Forecasting & Anomaly Detection   (version 2)
# Machine Learning Fundamentals - Lab Mid Project
#
# What this program does (in simple words):
#   1. Loads the website traffic data (number of requests every 5 minutes)
#   2. Cleans it, saves a cleaned copy of the dataset, builds simple features
#   3. Trains a Decision Tree to predict the next traffic value (forecasting)
#      and compares it with TWO simple baselines
#   4. If real traffic is VERY different from the prediction -> anomaly
#      (4 detectors are compared, tested on the real label AND on fake
#       anomalies that we add ourselves, repeated for 30 random seeds)
#   5. Uses Naive Bayes and Decision Tree to classify traffic as Low/Medium/High
#
# Labs used: Lab 1 (Python basics), Lab 2 (data & memory),
#            Lab 3 (functions), Lab 4 (Decision Tree), Lab 5 (Naive Bayes)
#
# Dataset: NAB (Numenta Anomaly Benchmark), file elb_request_count_8c0756.csv
#          It is the request count of a real website load balancer (AWS),
#          saved here as website_traffic.csv (the original, never changed).
#          The cleaned copy made by this program is website_traffic_clean.csv
#
# What is new compared with version 1 (and why):
#   - The tree depth is now chosen with 5-fold TIME-SERIES cross-validation
#     instead of one small validation window (that window was only ~2 days,
#     so the choice of depth 7 was not reliable).
#   - A second, stronger baseline (average of the last hour) is added.
#     Version 1 only compared with "same as last value", which is weak.
#   - A 4th detector: Tree residual with a separate error limit for every
#     leaf of the tree (noisy situations get a bigger limit).
#   - The fake-anomaly test is repeated for 30 seeds, not only seed 42,
#     because one seed with 13 events is too small to trust.
#   - The cleaned dataset is saved with an "is_interpolated" flag and the
#     NAB label.
# =====================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # draw graphs into files, no window needed
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix

# Find the folder of this file, so the program works from anywhere
folder = os.path.dirname(os.path.abspath(__file__))

# Folder where the graphs will be saved
plot_folder = os.path.join(folder, "plots")
os.makedirs(plot_folder, exist_ok=True)


# =====================================================================
# STEP 1: LOAD THE DATA  (Lab 1 + Lab 2)
# =====================================================================
print("STEP 1: Loading data")

df = pd.read_csv(os.path.join(folder, "website_traffic.csv"))
df["timestamp"] = pd.to_datetime(df["timestamp"])   # text -> real date/time
df = df.set_index("timestamp")                      # use time as the row label

print(df.head())
print("Rows:", len(df))
print("Time range:", df.index.min(), "to", df.index.max())
print("Missing values:", df["value"].isna().sum())
print("Duplicate timestamps:", df.index.duplicated().sum())
print("Smallest value:", df["value"].min(), "| Largest value:", df["value"].max(),
      "| Zeros:", int((df["value"] == 0).sum()))

# Lab 2: memory usage. float64 uses 8 bytes per number, float32 uses 4 bytes.
# We change the type to save memory and check how much we saved.
memory_before = df.memory_usage(deep=True).sum()
df["value"] = df["value"].astype("float32")
memory_after = df.memory_usage(deep=True).sum()
print("Memory before:", memory_before, "bytes")
print("Memory after :", memory_after, "bytes")
print("Memory saved :", round((1 - memory_after / memory_before) * 100, 1), "%")


# =====================================================================
# STEP 2: CLEAN THE DATA
# =====================================================================
print("\nSTEP 2: Cleaning data")

# The data should have one reading every 5 minutes, but a few readings
# are missing. asfreq() makes the time steps perfectly regular (it adds an
# empty row wherever a reading is missing and keeps the original time
# stamps, e.g. 00:04, 00:09 ...) and interpolate() fills the empty places
# with a value between its neighbours.
# (Version 1 used resample().mean(), which also moved every time stamp down
#  to the nearest 5 minutes. The values were the same, but the clock
#  times were shifted by 4 minutes, so we now use asfreq().)
df = df.asfreq("5min")
missing_places = df["value"].isna()                  # remember which rows were missing
print("Missing after resample:", int(missing_places.sum()))
df = df.interpolate()
print("Missing after fixing  :", df["value"].isna().sum())

# These are the REAL anomalies, written by the NAB authors (their label file).
# We will use them only to check our results, never to train.
real_anomalies = [pd.Timestamp("2014-04-12 17:24:00"),
                  pd.Timestamp("2014-04-22 19:34:00")]

# Save the cleaned dataset. Extra columns:
#   is_interpolated = 1 if we had to fill this reading (it was missing)
#   nab_anomaly     = 1 at the exact time NAB labelled as an anomaly
clean = df.copy()
clean["value"] = clean["value"].round(1)
clean["is_interpolated"] = missing_places.astype(int)
clean["nab_anomaly"] = 0
for t in real_anomalies:
    assert t in clean.index, "label time must be on the 5-minute grid"
    clean.loc[t, "nab_anomaly"] = 1
clean.to_csv(os.path.join(folder, "website_traffic_clean.csv"))
print("Saved website_traffic_clean.csv with", len(clean), "rows")


# =====================================================================
# STEP 3: LOOK AT THE DATA (EDA)
# =====================================================================
print("\nSTEP 3: Making graphs of the data")

# Graph 1: the whole traffic with the real anomalies marked in red
plt.figure(figsize=(12, 4))
plt.plot(df.index, df["value"], linewidth=0.6)
for t in real_anomalies:
    plt.axvline(t, color="red", linestyle="--")
plt.title("Website traffic (requests per 5 minutes), red lines = real anomalies")
plt.xlabel("Time")
plt.ylabel("Requests")
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "1_traffic.png"), dpi=150)
plt.close()

# Graph 2: average traffic for every hour of the day
average_by_hour = df.groupby(df.index.hour)["value"].mean()
plt.figure(figsize=(8, 4))
plt.bar(average_by_hour.index, average_by_hour.values)
plt.title("Average traffic by hour of the day")
plt.xlabel("Hour of the day")
plt.ylabel("Average requests")
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "2_hourly_pattern.png"), dpi=150)
plt.close()

# How noisy is the data? Compare the spread at quiet and busy hours.
# (If the spread is bigger when traffic is high, one single error limit
#  for the whole day is not fair. This matters for Step 8.)
hourly_mean = df.groupby(df.index.hour)["value"].mean()
hourly_std = df.groupby(df.index.hour)["value"].std()
print("Hour | mean | std")
for h in range(24):
    print(f"{h:4d} | {hourly_mean[h]:5.1f} | {hourly_std[h]:5.1f}")
print("Average at quiet hours 2-5 :", round(hourly_mean.loc[2:5].mean(), 1),
      "| spread (std):", round(hourly_std.loc[2:5].mean(), 1))
print("Average at busy hours 12-17:", round(hourly_mean.loc[12:17].mean(), 1),
      "| spread (std):", round(hourly_std.loc[12:17].mean(), 1))


# =====================================================================
# STEP 4: MAKE FEATURES  (Lab 3 - a simple function)
# =====================================================================
# A "feature" is a clue the model uses to guess the next value.
# Our clues are: what the traffic was a few minutes ago, the average of
# the last hour, the hour of the day and the day of the week.
def make_features(series):
    data = pd.DataFrame({"value": series})
    data["lag_1"] = data["value"].shift(1)      # traffic 5 minutes ago
    data["lag_2"] = data["value"].shift(2)      # traffic 10 minutes ago
    data["lag_3"] = data["value"].shift(3)      # traffic 15 minutes ago
    data["lag_12"] = data["value"].shift(12)    # traffic 1 hour ago
    data["avg_last_hour"] = data["value"].shift(1).rolling(12).mean()
    data["hour"] = data.index.hour              # 0 to 23
    data["day_of_week"] = data.index.dayofweek  # 0 = Monday
    data = data.dropna()                        # first rows have no history, drop them
    return data


features = ["lag_1", "lag_2", "lag_3", "lag_12",
            "avg_last_hour", "hour", "day_of_week"]

all_data = make_features(df["value"])
print("\nSTEP 4: Features created. Rows:", len(all_data))


# =====================================================================
# STEP 5: SPLIT INTO TRAIN AND TEST
# =====================================================================
# In time data we must NOT shuffle. The past is used for training and the
# future is used for testing. First 70% = train, last 30% = test.
split_point = int(len(all_data) * 0.7)
train = all_data.iloc[:split_point]
test = all_data.iloc[split_point:]

print("\nSTEP 5: Train/Test split")
print("Train:", train.index.min(), "to", train.index.max(), "(", len(train), "rows )")
print("Test :", test.index.min(), "to", test.index.max(), "(", len(test), "rows )")


# =====================================================================
# STEP 6: CHOOSE THE TREE SETTINGS WITH TIME-SERIES CROSS-VALIDATION (Lab 4)
# =====================================================================
# If a tree is too deep it memorises the training data (overfitting).
# If it is too shallow it learns nothing (underfitting).
#
# Version 1 used ONE validation window (the last 20% of train, about 2 days).
# One short window can be lucky or unlucky. Now we use TimeSeriesSplit:
# the train data is cut into 5 folds, and each time the model is trained on
# the PAST folds and checked on the NEXT fold (the future is never used to
# train). The 5 errors are averaged. The test data is not touched.
#
# We also compare two ways for the tree to measure its own mistakes:
#   squared_error  = the usual one (punishes big mistakes a lot)
#   absolute_error = the same kind of error as our MAE score
print("\nSTEP 6: Choosing the tree settings with 5-fold time-series cross-validation")

splitter = TimeSeriesSplit(n_splits=5)
depths = list(range(1, 11))
cv_results = {}                                  # criterion -> list of (mean, std)

for criterion in ["squared_error", "absolute_error"]:
    cv_results[criterion] = []
    for depth in depths:
        fold_errors = []
        for fit_rows, check_rows in splitter.split(train):
            tree = DecisionTreeRegressor(criterion=criterion, max_depth=depth,
                                         min_samples_leaf=10, random_state=0)
            tree.fit(train.iloc[fit_rows][features], train.iloc[fit_rows]["value"])
            fold_errors.append(mean_absolute_error(
                train.iloc[check_rows]["value"],
                tree.predict(train.iloc[check_rows][features])))
        cv_results[criterion].append((np.mean(fold_errors), np.std(fold_errors)))
        print(criterion, "depth", depth, "-> CV error", round(np.mean(fold_errors), 2),
              "(+/-", round(np.std(fold_errors), 2), ")")

# pick the combination with the smallest average CV error
best_error = 10 ** 9
for criterion in cv_results:
    for i, depth in enumerate(depths):
        if cv_results[criterion][i][0] < best_error:
            best_error = cv_results[criterion][i][0]
            best_criterion = criterion
            best_depth = depth
print("Chosen:", best_criterion, ", max_depth =", best_depth,
      "(CV error", round(best_error, 2), ")")

# What did version 1 use? depth 7 with squared_error, shown here for comparison
v1_depth = 7
v1_cv = cv_results["squared_error"][v1_depth - 1][0]
print("Version 1 choice (squared_error, depth 7) has CV error", round(v1_cv, 2))

plt.figure(figsize=(8, 4))
for criterion in cv_results:
    means = [m for m, s in cv_results[criterion]]
    stds = [s for m, s in cv_results[criterion]]
    plt.errorbar(depths, means, yerr=stds, marker="o", capsize=3, label=criterion)
plt.title("Cross-validation error vs tree depth (bars = spread over 5 folds)")
plt.xlabel("max_depth")
plt.ylabel("Mean absolute error")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "3_depth_vs_error.png"), dpi=150)
plt.close()


# =====================================================================
# STEP 7: TRAIN THE FORECASTING MODEL AND COMPARE WITH BASELINES
# =====================================================================
print("\nSTEP 7: Forecasting")

# Final model, trained on the full train part
model = DecisionTreeRegressor(criterion=best_criterion, max_depth=best_depth,
                              min_samples_leaf=10, random_state=0)
model.fit(train[features], train["value"])
predicted = model.predict(test[features])

# The version 1 model (depth 7), kept only to compare
model_v1 = DecisionTreeRegressor(max_depth=v1_depth, min_samples_leaf=10, random_state=0)
model_v1.fit(train[features], train["value"])
predicted_v1 = model_v1.predict(test[features])

# Baseline 1 = "next value will be same as the last value".
# Baseline 2 = "next value will be the average of the last hour".
# A real model must beat BOTH, otherwise it is not useful.
baseline_last = test["lag_1"]
baseline_avg = test["avg_last_hour"]


def score(actual, guess):
    mae = mean_absolute_error(actual, guess)
    rmse = np.sqrt(mean_squared_error(actual, guess))
    return mae, rmse


last_mae, last_rmse = score(test["value"], baseline_last)
avg_mae, avg_rmse = score(test["value"], baseline_avg)
tree_mae, tree_rmse = score(test["value"], predicted)
v1_mae, v1_rmse = score(test["value"], predicted_v1)

print("Baseline 1 (same as last value)   -> MAE:", round(last_mae, 2), " RMSE:", round(last_rmse, 2))
print("Baseline 2 (average of last hour) -> MAE:", round(avg_mae, 2), " RMSE:", round(avg_rmse, 2))
print("Decision Tree (chosen by CV)      -> MAE:", round(tree_mae, 2), " RMSE:", round(tree_rmse, 2))
print("Decision Tree (version 1, depth 7)-> MAE:", round(v1_mae, 2), " RMSE:", round(v1_rmse, 2))
print("Tree vs baseline 1: MAE", round((1 - tree_mae / last_mae) * 100, 1), "% lower")
print("Tree vs baseline 2: MAE", round((1 - tree_mae / avg_mae) * 100, 1), "% lower")

# Which clues did the tree find most useful?
print("\nFeature importance (chosen tree):")
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(importance.round(3))
print("Number of leaves in the chosen tree:", model.get_n_leaves())

# Print the rules of the tree in plain text
print("\nRules of the chosen tree:")
print(export_text(model, feature_names=features, max_depth=3))

print("Feature importance of the version 1 tree (depth 7), for comparison:")
print(pd.Series(model_v1.feature_importances_, index=features).sort_values(ascending=False).round(3))

# Graph: actual vs predicted for the first 2 days of the test period
two_days = 288 * 2
plt.figure(figsize=(12, 4))
plt.plot(test.index[:two_days], test["value"].iloc[:two_days], label="Actual", linewidth=0.8)
plt.plot(test.index[:two_days], predicted[:two_days], label="Decision Tree (chosen by CV)", linewidth=1.2)
plt.plot(test.index[:two_days], baseline_avg.iloc[:two_days], label="Average of last hour", linewidth=0.8, alpha=0.8)
plt.title("Actual vs predicted traffic (first 2 days of test data)")
plt.xlabel("Time")
plt.ylabel("Requests")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "4_actual_vs_predicted.png"), dpi=150)
plt.close()


# =====================================================================
# STEP 8: ANOMALY DETECTION  (Lab 3 function + Lab 4 model)
# =====================================================================
# We compare 4 simple detectors:
#   A) Tree residual         : error = actual - predicted. If the error is
#      bigger than 3 x (normal error size), the point is an anomaly.
#   A2) Tree residual per leaf: same idea, but every leaf of the tree has its
#      OWN normal error size (busy situations are noisier, so they get a
#      bigger limit, quiet situations get a smaller limit).
#   B) Rolling z-score       : compare the value with the average and spread
#      of the last 1 day (288 readings). Far away (more than 3) -> anomaly.
#   C) Fixed threshold       : anything above (train average + 3 x train
#      spread) is an anomaly. Very simple, used as a base to compare.
# The "3" is the same rule of thumb as in version 1. It was NOT tuned.
print("\nSTEP 8: Anomaly detection")

# Normal error size of the tree and the fixed threshold, taken from TRAIN data
train_residual = train["value"] - model.predict(train[features])
tree_limit = 3 * train_residual.std()
fixed_limit = train["value"].mean() + 3 * train["value"].std()

# One limit for every leaf: group the train errors by the leaf they fall in
train_leaf = model.apply(train[features])
leaf_limit = 3 * train_residual.groupby(train_leaf).std()
print("Tree error limit, one for all (3 x std):", round(tree_limit, 1))
print("Tree error limit per leaf              :", leaf_limit.round(1).to_dict())
print("Fixed threshold                        :", round(fixed_limit, 1))


# This function runs all 4 detectors on a traffic series and returns
# True/False flags (True = anomaly) for every row of the TEST period.
def run_detectors(series):
    data = make_features(series)
    data = data.loc[test.index[0]:]            # only the test period

    # A) Decision Tree residual, one limit for all
    error = data["value"] - model.predict(data[features])
    flag_tree = error.abs() > tree_limit

    # A2) Decision Tree residual, one limit for every leaf
    this_leaf = pd.Series(model.apply(data[features]), index=data.index)
    flag_leaf = error.abs() > this_leaf.map(leaf_limit)

    # B) Rolling z-score (uses only the PAST 288 values)
    past_mean = series.shift(1).rolling(288).mean()
    past_std = series.shift(1).rolling(288).std()
    z_score = (series - past_mean) / past_std
    flag_zscore = z_score.loc[data.index].abs() > 3

    # C) Fixed threshold
    flag_fixed = data["value"] > fixed_limit

    return pd.DataFrame({"Tree residual": flag_tree,
                         "Tree residual per leaf": flag_leaf,
                         "Rolling z-score": flag_zscore,
                         "Fixed threshold": flag_fixed})


# This function checks the flags against a list of true anomaly periods.
# events = list of (start_time, end_time)
def check_detector(flags, events):
    found = 0
    for start, end in events:
        if flags.loc[start:end].any():     # at least one alarm inside the event
            found = found + 1

    inside_event = pd.Series(False, index=flags.index)
    for start, end in events:
        inside_event.loc[start:end] = True

    total_alarms = int(flags.sum())
    good_alarms = int((flags & inside_event).sum())
    false_alarms = total_alarms - good_alarms

    recall = found / len(events)                       # how many events found
    if total_alarms > 0:
        precision = good_alarms / total_alarms         # how many alarms were correct
    else:
        precision = 0
    return found, len(events), total_alarms, false_alarms, precision, recall


# ---------- 8a) Test on the REAL labelled anomaly ----------
# Only the 22 April anomaly is in the test period.
# We accept alarms that come within 2 hours of the labelled time.
# (NAB itself uses a much wider window of about 16 hours, so ours is stricter.)
real_event = real_anomalies[1]
real_events = [(real_event - pd.Timedelta(hours=2), real_event + pd.Timedelta(hours=2))]

real_flags = run_detectors(df["value"])

print("\n8a) Real anomaly (22 April 2014, 19:34)")
for name in real_flags.columns:
    found, n_events, alarms, false_alarms, precision, recall = check_detector(real_flags[name], real_events)
    print(name, "-> found:", found, "/", n_events, "| total alarms:", alarms,
          "| false alarms:", false_alarms)

# Graph: where did the tree detector raise alarms?
plt.figure(figsize=(12, 4))
plt.plot(test.index, test["value"], linewidth=0.6, label="Traffic")
alarm_times = real_flags.index[real_flags["Tree residual per leaf"]]
plt.scatter(alarm_times, test.loc[alarm_times, "value"], color="orange", s=15,
            label="Alarms (tree residual per leaf)")
plt.axvline(real_event, color="red", linestyle="--", label="Real anomaly")
plt.title("Anomalies found by the Decision Tree residual method (test period)")
plt.xlabel("Time")
plt.ylabel("Requests")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "5_tree_anomalies.png"), dpi=150)
plt.close()


# ---------- 8b) Test with FAKE anomalies we add ourselves ----------
# The real data has only one labelled anomaly in the test period, so one
# result is not enough to judge. We add 8 fake spikes and 4 fake drops
# (like a server outage) into a COPY of the data, so we know exactly where
# the anomalies are. The random seed makes the result the same every run.
# Lab 3: all of this is inside ONE function so we can repeat it for many seeds.
def make_fake_series(seed):
    np.random.seed(seed)
    fake_series = df["value"].copy()
    fake_events = []
    fake_kinds = []

    # We keep a safe distance from the real anomaly and from each other
    busy_places = []          # positions where traffic is busy (good for outage drops)
    all_places = []           # any position
    for i in range(30, len(test) - 30, 60):
        time_here = test.index[i]
        if abs((time_here - real_event).total_seconds()) < 4 * 3600:
            continue                                  # too close to the real anomaly
        all_places.append(i)
        if test["avg_last_hour"].iloc[i] > 60:
            busy_places.append(i)

    # 4 drops (traffic suddenly becomes 0 for 30 minutes) at busy times
    drop_places = np.random.choice(busy_places, 4, replace=False)
    for i in drop_places:
        start = test.index[i]
        end = test.index[i + 5]
        fake_series.loc[start:end] = 0
        fake_events.append((start, end))
        fake_kinds.append("drop")

    # 8 spikes (traffic suddenly becomes 4 times bigger, plus 50, for 15 minutes)
    left_places = [p for p in all_places if p not in drop_places]
    spike_places = np.random.choice(left_places, 8, replace=False)
    for i in spike_places:
        start = test.index[i]
        end = test.index[i + 2]
        fake_series.loc[start:end] = fake_series.loc[start:end] * 4 + 50
        fake_events.append((start, end))
        fake_kinds.append("spike")

    return fake_series, fake_events, fake_kinds


# Main run: seed 42 (the same fake anomalies as in version 1)
print("\n8b) Fake anomalies added to a copy of the data (seed 42)")
fake_series, fake_events, fake_kinds = make_fake_series(42)
print("Fake anomalies added:", len(fake_events), "(4 drops + 8 spikes)")

# Now the list of events = 12 fake ones + the 1 real one
all_events = fake_events + real_events
fake_flags = run_detectors(fake_series)

print("Total events to find:", len(all_events), "(12 fake + 1 real)")
for name in fake_flags.columns:
    found, n_events, alarms, false_alarms, precision, recall = check_detector(fake_flags[name], all_events)
    spikes_found = sum(fake_flags[name].loc[s:e].any()
                       for (s, e), k in zip(fake_events, fake_kinds) if k == "spike")
    drops_found = sum(fake_flags[name].loc[s:e].any()
                      for (s, e), k in zip(fake_events, fake_kinds) if k == "drop")
    print(name, "-> found:", found, "/", n_events, "| total alarms:", alarms,
          "| false alarms:", false_alarms,
          "| precision:", round(precision, 2), "| recall:", round(recall, 2),
          "| spikes found:", spikes_found, "/ 8 | drops found:", drops_found, "/ 4")

# Event by event table for seed 42: which detector found which fake anomaly?
print("\nEvent by event (seed 42):  A = tree, A2 = tree per leaf, B = z-score, C = fixed")
for (s, e), k in zip(fake_events, fake_kinds):
    peak = fake_series.loc[s:e].max()
    normal = df["value"].loc[s:e].max()
    marks = ["A" if fake_flags["Tree residual"].loc[s:e].any() else "-",
             "A2" if fake_flags["Tree residual per leaf"].loc[s:e].any() else "--",
             "B" if fake_flags["Rolling z-score"].loc[s:e].any() else "-",
             "C" if fake_flags["Fixed threshold"].loc[s:e].any() else "-"]
    print(s, "|", k.ljust(5), "| highest value inside:", int(peak),
          "| value without the fake change:", int(normal), "|", " ".join(marks))

# Repeat for 30 different seeds. One seed can be lucky or unlucky.
print("\n8c) The same test repeated for 30 seeds (each seed puts the fake anomalies in other places)")
seeds = list(range(30))
summary = {name: {"recall": [], "precision": [], "false": [], "spike": [], "drop": []}
           for name in fake_flags.columns}
for seed in seeds:
    series_s, events_s, kinds_s = make_fake_series(seed)
    flags_s = run_detectors(series_s)
    for name in flags_s.columns:
        found, n_events, alarms, false_alarms, precision, recall = check_detector(
            flags_s[name], events_s + real_events)
        summary[name]["recall"].append(recall)
        summary[name]["precision"].append(precision)
        summary[name]["false"].append(false_alarms)
        summary[name]["spike"].append(np.mean([flags_s[name].loc[s:e].any()
                                               for (s, e), k in zip(events_s, kinds_s) if k == "spike"]))
        summary[name]["drop"].append(np.mean([flags_s[name].loc[s:e].any()
                                              for (s, e), k in zip(events_s, kinds_s) if k == "drop"]))

for name in summary:
    r = summary[name]
    print(name, "-> recall:", round(np.mean(r["recall"]), 2), "+/-", round(np.std(r["recall"]), 2),
          "| precision:", round(np.mean(r["precision"]), 2), "+/-", round(np.std(r["precision"]), 2),
          "| false alarms:", round(np.mean(r["false"]), 1),
          "| spikes found:", round(np.mean(r["spike"]), 2),
          "| drops found:", round(np.mean(r["drop"]), 2))

# Graph: recall and precision, averaged over the 30 seeds
names = list(summary.keys())
positions = np.arange(len(names))
plt.figure(figsize=(9, 4))
plt.bar(positions - 0.2, [np.mean(summary[n]["recall"]) for n in names], 0.4,
        yerr=[np.std(summary[n]["recall"]) for n in names], capsize=3, label="Recall")
plt.bar(positions + 0.2, [np.mean(summary[n]["precision"]) for n in names], 0.4,
        yerr=[np.std(summary[n]["precision"]) for n in names], capsize=3, label="Precision")
plt.xticks(positions, names)
plt.ylim(0, 1)
plt.title("Detectors on fake + real anomalies (mean over 30 seeds, bars = std)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "6_detector_comparison.png"), dpi=150)
plt.close()


# =====================================================================
# STEP 9: CLASSIFY TRAFFIC LEVEL WITH NAIVE BAYES AND DECISION TREE (Lab 5)
# =====================================================================
# Operators often only want to know "will traffic be Low, Medium or High?"
# instead of an exact number. We make 3 classes from the TRAIN data:
#   Low    = lowest third of values
#   Medium = middle third
#   High   = top third
print("\nSTEP 9: Classification of traffic level")

low_limit = train["value"].quantile(0.33)
high_limit = train["value"].quantile(0.66)
print("Low is below", round(low_limit, 1), "| High is above", round(high_limit, 1))


# Turns a number into a class: 0 = Low, 1 = Medium, 2 = High
def make_level(values):
    levels = pd.Series(1, index=values.index)       # start with Medium
    levels[values <= low_limit] = 0                 # Low
    levels[values > high_limit] = 2                 # High
    return levels


y_train = make_level(train["value"])
y_test = make_level(test["value"])

# Naive Bayes (Lab 5)
nb_model = GaussianNB()
nb_model.fit(train[features], y_train)
nb_predicted = nb_model.predict(test[features])

# Decision Tree classifier (Lab 4), to compare
dt_class = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=0)
dt_class.fit(train[features], y_train)
dt_predicted = dt_class.predict(test[features])

# Baseline: "same class as the last value"
last_level = make_level(test["lag_1"])

nb_accuracy = accuracy_score(y_test, nb_predicted)
dt_accuracy = accuracy_score(y_test, dt_predicted)
base_accuracy = accuracy_score(y_test, last_level)

print("Class share in test data (Low/Medium/High):",
      (y_test.value_counts(normalize=True).sort_index().round(3)).tolist())
print("Baseline accuracy   :", round(base_accuracy, 3))
print("Naive Bayes accuracy:", round(nb_accuracy, 3))
print("Decision Tree accuracy:", round(dt_accuracy, 3))

class_names = ["Low", "Medium", "High"]
print("\nNaive Bayes confusion matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(confusion_matrix(y_test, nb_predicted), index=class_names, columns=class_names))
print("\nDecision Tree confusion matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(confusion_matrix(y_test, dt_predicted), index=class_names, columns=class_names))


# =====================================================================
# STEP 9b: SAVE THE COMPLETE DATASET (all columns in one file)
# =====================================================================
# One row per 5-minute reading, with the raw value, the time clues, the
# clues used by the models, the labels, and the model results.
print("\nSTEP 9b: Saving the complete dataset")

full = pd.DataFrame({"value": df["value"].round(1)})
full["is_interpolated"] = missing_places.astype(int)
full["date"] = full.index.date
full["hour"] = full.index.hour
full["day_of_week"] = full.index.dayofweek                    # 0 = Monday
full["day_name"] = full.index.day_name()
full["is_weekend"] = (full.index.dayofweek >= 5).astype(int)
full["lag_1"] = df["value"].shift(1).round(1)
full["lag_2"] = df["value"].shift(2).round(1)
full["lag_3"] = df["value"].shift(3).round(1)
full["lag_12"] = df["value"].shift(12).round(1)
full["avg_last_hour"] = df["value"].shift(1).rolling(12).mean().round(2)
full["std_last_hour"] = df["value"].shift(1).rolling(12).std().round(2)
full["avg_last_day"] = df["value"].shift(1).rolling(288).mean().round(2)
full["std_last_day"] = df["value"].shift(1).rolling(288).std().round(2)

# Traffic level (Low/Medium/High) with the limits learned from TRAIN only
full["traffic_level"] = make_level(df["value"]).map({0: "Low", 1: "Medium", 2: "High"})

# train / test marker (the first 12 rows have no history = warmup)
full["split"] = "warmup"
full.loc[train.index, "split"] = "train"
full.loc[test.index, "split"] = "test"

# NAB labels: exact time and the wide official window (used only for checking)
full["nab_anomaly"] = 0
for t in real_anomalies:
    full.loc[t, "nab_anomaly"] = 1
nab_windows = [("2014-04-12 09:04", "2014-04-13 01:44"),
               ("2014-04-22 11:14", "2014-04-23 03:54")]
full["nab_window"] = 0
for start, end in nab_windows:
    full.loc[start:end, "nab_window"] = 1

# Model results: prediction and error of the chosen tree, and the alarms
# of the 4 detectors (alarms are only made for the TEST period)
full["predicted"] = pd.Series(model.predict(all_data[features]), index=all_data.index).round(1)
full["error"] = (df["value"] - full["predicted"]).round(1)
alarm_columns = {"Tree residual": "alarm_tree", "Tree residual per leaf": "alarm_tree_leaf",
                 "Rolling z-score": "alarm_zscore", "Fixed threshold": "alarm_fixed"}
for name, column in alarm_columns.items():
    full[column] = np.nan
    full.loc[real_flags.index, column] = real_flags[name].astype(int)

full.index.name = "timestamp"
full.to_csv(os.path.join(folder, "website_traffic_complete.csv"))
print("Saved website_traffic_complete.csv:", len(full), "rows,", len(full.columns) + 1, "columns")


# =====================================================================
# STEP 10: FINAL SUMMARY
# =====================================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print("Forecasting MAE   : last value", round(last_mae, 2), "| last-hour average", round(avg_mae, 2),
      "| Decision Tree", round(tree_mae, 2), "| v1 tree (depth 7)", round(v1_mae, 2))
print("Forecasting RMSE  : last value", round(last_rmse, 2), "| last-hour average", round(avg_rmse, 2),
      "| Decision Tree", round(tree_rmse, 2), "| v1 tree (depth 7)", round(v1_rmse, 2))
print("Classification acc: baseline", round(base_accuracy, 3),
      "| Naive Bayes", round(nb_accuracy, 3),
      "| Decision Tree", round(dt_accuracy, 3))
print("Graphs are saved in the folder:", plot_folder)
