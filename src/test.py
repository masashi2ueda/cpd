# %%
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random

from ocpdet import CUSUM, EWMA, TwoSample, NeuralNetwork
import changefinder

import create_sim_data

random.seed(0)
np.random.seed(0)
# %%
####################
# データ生成
####################
# パラメータ
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
weekday_means = [20, 30, 40, 20]
weekday_range_hs = [5, 5, 10, 10]
summer_rates = [1.5, 1.5, 1.5, 1.5]
winter_rates = [0.5, 0.5, 0.5, 0.5]
shift_days = [2, 2, 2]
start_dates = [
    datetime.datetime(2020, 10, 1),
    datetime.datetime(2021, 4, 1),
    datetime.datetime(2021, 10, 1)
]

# データ生成
date_df, date_dfs = create_sim_data.create_x_dates(
    weekday_means,
    weekday_range_hs,
    summer_rates,
    winter_rates,
    shift_days,
    start_dates)

# for i in range(4):
#     plt.figure(figsize=(15, 5))
#     plt.plot(date_dfs[i]["date"], date_dfs[i]["x"], label=str(i))
#     plt.plot(date_df["date"], date_df["x"], label="dst")
#     plt.legend()

# 真の変化点
true_change_idxs = np.where(date_df["is_change_point"].values==1)[0]
# データ
xs = date_df["x"].values


# %%
def draw_trues():
    for ii, idx in enumerate(true_change_idxs):
        date = dates[idx]
        label = "true_cp" if ii == 0 else None
        plt.axvline(date, color="r", linestyle="solid", label=label)

# %%
####################
# CUMSUM
####################
model = CUSUM(k=1., h=2., burnin=50, mu=0., sigma=1.)
model.process(xs)
my_changepoints = model.changepoints


plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
dates = date_df["date"].values
plt.plot(dates, xs) 
for pr in my_changepoints:
    if len(dates) <= pr:
        continue
    plt.axvline(dates[pr], color="b", linestyle="dashed")
draw_trues()

plt.subplot(2, 1, 2)
plt.plot(dates, model.S, c="m", label="$S$")
plt.plot(dates, model.T, c="y", label="$T$")
plt.axhline(model.h, color="r", linestyle="-", zorder=-10)
plt.scatter(dates[model.changepoints], len(model.changepoints) * [model.h], marker="v",
            label="Alarm", color="green")
plt.xlabel("Time")
plt.ylabel("$S$ and $T$ statistics")
draw_trues()
plt.legend()
# %%
####################
# EWMA(Exponentially Weighted Moving Average algorithm)
####################
model = EWMA(r=0.15, L=2.4, burnin=50, mu=0., sigma=1.)
model.process(xs)
my_changepoints = model.changepoints

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
dates = date_df["date"].values
plt.plot(dates, xs) 
for pr in my_changepoints:
    if len(dates) <= pr:
        continue
    plt.axvline(dates[pr], color="b", linestyle="dashed")
draw_trues()

plt.subplot(2, 1, 2)
plt.scatter(dates, xs, c="b", s=20, alpha=0.2)
plt.plot(dates, model.Z, c="m", label="$Z$")
plt.plot(dates, np.asarray(model._mu) + np.asarray(model.sigma_Z) * model.L,
         c="y", label="$\mu \pm L \sigma_Z$")
plt.plot(dates, np.asarray(model._mu) - np.asarray(model.sigma_Z) * model.L,
         c="y")
plt.scatter(dates[model.changepoints], np.asarray(model.Z)[model.changepoints], marker="v",
            label="Alarm", color="green", zorder=10)
plt.xlabel("Time")
plt.ylabel("Observations")
draw_trues()
plt.legend()

# %%
####################
# Two-sample test algorithm
####################
model = TwoSample(statistic="Lepage", threshold=3.1)
model.process(xs)
my_changepoints = model.changepoints

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
dates = date_df["date"].values
plt.plot(dates, xs) 
for pr in my_changepoints:
    if len(dates) <= pr:
        continue
    plt.axvline(dates[pr], color="b", linestyle="dashed")
draw_trues()

plt.subplot(2, 1, 2)
plt.plot(dates, model.D, c="m", label="$D$")
plt.axhline(model.threshold, color="r", linestyle="-", zorder=-10)
plt.scatter(dates[model.changepoints], len(model.changepoints) * [model.threshold], marker="v",
            label="Alarm", color="green")
plt.xlabel("Time")
plt.ylabel("$D$ statistic")
plt.legend()
draw_trues()
plt.legend()
# %%

# %%
xs = np.arange(100)
model = NeuralNetwork(k=5, n=10, lag=20, f=None, r=.1, L=2, burnin=10, method="increase", timeout=100)
model.process(xs)
# %%
print(model.X.shape)
for i in range(3):
    print(i, model.X[i])

# %%
print("model.Xi.shape:", model.Xi.shape)
for i in range(3):
    print(f"model.Xi[{i}].shape:", model.Xi[i].shape)
    for ii in range(3):
        print(f"model.Xi[{i}][{ii}].shape:", model.Xi[i][ii].shape)
        print(model.Xi[i][[ii]])

# %%
from tqdm import tqdm
import tensorflow as tf
self = model
print("len(self.Xi):", len(self.Xi))
print("self.l :", self.l )
print("len(self.Xi) - self.l - 1 :", len(self.Xi) - self.l - 1)
for i in tqdm(range(len(self.Xi) - self.l - 1)):
    X_lagged = self.Xi[i][tf.newaxis, ...]
    X_recent = self.Xi[i+self.l][tf.newaxis, ...]
    print("X_lagged.shape:", X_lagged.shape)
    print("X_recent.shape:", X_recent.shape)
    1/0
    d = tf.math.log((1 - self.f(X_lagged)) / self.f(X_lagged)) + tf.math.log(self.f(X_recent) / (1 - self.f(X_recent)))
    if i <= self.l + 1:
        self.dissimilarity.append(0.)
    else:
        d_bar = self.dissimilarity[-1] + (d - self.divergence[i-1-self.l]) / self.l
        self.dissimilarity.append(d_bar.numpy()[0][0])
    self.divergence.append(d.numpy()[0][0])
    with tf.GradientTape() as tape:
        loss_value = - tf.math.log(1 - self.f(X_lagged)) - tf.math.log(self.f(X_recent))
    grads = tape.gradient(loss_value, self.f.trainable_weights)
    self.optimiser.apply_gradients(zip(grads, self.f.trainable_weights))
self.divergence = np.asarray(self.divergence)
self.dissimilarity = np.asarray(self.dissimilarity)

# %%
####################
# Neural network for changepoint detection algorithm
####################
model = NeuralNetwork(k=5, n=10, lag=60, f=None, r=.1, L=2, burnin=100, method="increase", timeout=100)
model.process(xs)
my_changepoints = model.changepoints

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
dates = date_df["date"].values
plt.plot(dates, xs) 
for pr in my_changepoints:
    if len(dates) <= pr:
        continue
    plt.axvline(dates[pr], color="b", linestyle="dashed")
draw_trues()
# %%
####################
# change finder
####################
cf = changefinder.ChangeFinder(r=0.01, order=1, smooth=7)
ret = []
for x in xs:
    score = cf.update(x)
    ret.append(score)

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
dates = date_df["date"].values
plt.plot(dates, xs)
plt.subplot(2, 1, 2)
plt.plot(dates, ret)
# %%
