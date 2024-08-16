# %%
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd


# %%
def create_weekday_dict(mean_val, range_h):
    # 各日の状態を作成
    weekday_dict = {}
    for do in range(1, 8):
        val = np.random.randint(mean_val - range_h, mean_val + range_h)
        weekday_dict[do] = val
    return weekday_dict

def add_date_int(date_df, col):
    year = []
    month = []
    day = []
    for val in date_df[col]:
        year.append(val.year)
        month.append(val.month)
        day.append(val.day)
    date_df["year"] = year
    date_df["month"] = month
    date_df["day"] = day
    return date_df


def create_day_df(weekday_dict):
    # 日付を作成
    date_df = []
    for d in range(365 * 2):
        if len(date_df) == 0:
            date = datetime.datetime(2020, 4, 1)
        else:
            date =  date_df[-1]["date"] + datetime.timedelta(days=1)
        week_day = date.isoweekday()
        weekday_val = weekday_dict[week_day]
        x = np.random.poisson(lam=weekday_val, size=1)[0]
        date_df.append({
            "date": date,
            "weekday": week_day,
            "weekday_val": weekday_val,
            "x": x
        })
    date_df = pd.DataFrame(date_df)
    date_df["is_change_point"] = 0

    date_df = add_date_int(date_df, "date")
    return date_df


def create_year_dates(weekday_mean, weekday_range_h, summer_rate, winter_rate):
    weekday_dict = create_weekday_dict(weekday_mean, weekday_range_h)
    date_df = create_day_df(weekday_dict=weekday_dict)

    days = date_df["day"].values
    for year in date_df["year"].unique():
        for months, rate in zip([[7, 8], [1]], [summer_rate, winter_rate]):
            ta_df = date_df.copy()
            ta_df = ta_df[ta_df["month"].isin(months)]
            ta_df = ta_df[ta_df["year"] == year]
            idxs = ta_df.index
            if len(idxs) == 0:
                continue
            date_df["x"].values[idxs] = (date_df["x"].values[idxs] * rate).astype(np.int64)
            ch1 = idxs[0]
            ch2 = idxs[-1]
            date_df["is_change_point"].values[ch1] = 1 
            date_df["is_change_point"].values[ch2] = 1 

    return date_df


def create_x_dates(
    weekday_means,
    weekday_range_hs,
    summer_rates,
    winter_rates,
    shift_days,
    start_dates
):
    date_dfs = []
    for i in range(len(weekday_means)):
        weekday_mean = weekday_means[i]
        weekday_range_h = weekday_range_hs[i]
        summer_rate = summer_rates[i]
        winter_rate = winter_rates[i]
        date_df = create_year_dates(weekday_mean, weekday_range_h, summer_rate, winter_rate)
        date_dfs.append(date_df)

    xs = date_dfs[0]["x"].values

    is_change_points = date_dfs[0]["is_change_point"].values.copy()
    for i in range(3):
        start_date = start_dates[i]
        end_date = start_date + datetime.timedelta(days=shift_days[i])

        xs2 = date_dfs[i+1]["x"].values

        ta_idxs = np.where((start_date <= date_df["date"]) & (date_df["date"]<= end_date))[0]
        add_ps = np.linspace(0, 1, len(ta_idxs))
        p_vals = np.zeros(len(date_df))
        p_vals[:(ta_idxs[0])] = 0
        p_vals[(ta_idxs[-1]):] = 1
        p_vals[ta_idxs] = add_ps
        xs = xs * (1 - p_vals) + xs2 * p_vals

        is_change_points[ta_idxs[0]] = 1

    dst_df = pd.DataFrame()
    dst_df["date"] = date_dfs[0]["date"]
    dst_df["x"] = xs
    dst_df["is_change_point"] = is_change_points
    return dst_df, date_dfs


# %%
if __name__ == "__main__":
    # weekday_means = [10, 20, 30, 20]
    # weekday_range_hs = [5, 5, 10, 10]
    # summer_rates = [1.1, 1.2, 1.3, 1.4]
    # winter_rates = [0.9, 0.8, 0.7, 0.6]
    # shift_days = [2, 10, 20]
    # start_dates = [
    #     datetime.datetime(2020, 10, 1),
    #     datetime.datetime(2021, 4, 1),
    #     datetime.datetime(2021, 10, 1)
    # ]
    weekday_means = [20, 20, 20, 20]
    weekday_range_hs = [5, 5, 10, 10]
    summer_rates = [1.5, 1.5, 1.5, 1.5]
    winter_rates = [0.5, 0.5, 0.5, 0.5]
    shift_days = [2, 2, 2]
    start_dates = [
        datetime.datetime(2020, 10, 1),
        datetime.datetime(2021, 4, 1),
        datetime.datetime(2021, 10, 1)
    ]

    date_df, date_dfs = create_x_dates(
        weekday_means,
        weekday_range_hs,
        summer_rates,
        winter_rates,
        shift_days,
        start_dates)


    plt.figure(figsize=(15, 5))
    for i in range(4):
        plt.plot(date_dfs[i]["date"], date_dfs[i]["x"], label=str(i))
    plt.plot(date_df["date"], date_df["x"], label="dst")

    print("change days")
    for day in date_df[date_df["is_change_point"] == 1]["date"].values:
        print(day)
        plt.axvline(day, color="r")
    plt.legend()

# # %%

# for i in range(4):
#     plt.figure(figsize=(15, 5))
#     plt.plot(date_dfs[i]["day"], date_dfs[i]["x"], label=str(i))
#     date_df = date_dfs[i]
#     for day in date_df[date_df["is_change_point"] == 1]["day"].values:
#         print(day)
#         plt.axvline(day, color="r")
#         print("change days")
#     plt.legend()

# # %%
# vals = date_df["day"]
# # %%
# for i, di in enumerate(vals):
#     month_int = int(str(di)[5:7])


# # %%
# str(di)
# # %%
# type(di)
# # %%
# print(di.year, di.month, di.day)

# # %%

# # %%
