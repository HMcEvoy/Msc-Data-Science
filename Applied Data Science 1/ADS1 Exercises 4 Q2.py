# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:59:12 2023

@author: harle
"""

#importing pandas and pyplot modules
import pandas as pd
import matplotlib.pyplot as plt

#importing data from csv files to pandas dataframe 
df_BCS = pd.read_csv("BCS_ann.csv")
df_BP = pd.read_csv("BP_ann.csv")
df_TSCO = pd.read_csv("TSCO_ann.csv")
df_VOD = pd.read_csv("VOD_ann.csv")

#creating figure 1, 2x2 subplots of data
plt.figure(1).figsize=(15, 15)

#subplot histogram showing annual returns of Barclays
plt.subplot(2, 2, 1)
plt.hist(df_BCS["ann_return"], bins=15)
plt.ylabel("Barclays")

#subplot histogram showing annual returns of BP
plt.subplot(2, 2, 2)
plt.hist(df_BP["ann_return"], bins=15)
plt.ylabel("BP")

##subplot histogram showing annual returns of Tesco
plt.subplot(2, 2, 3)
plt.hist(df_TSCO["ann_return"], bins=15)
plt.ylabel("Tesco")


#subplot histogram showing annual returns of Vodaphone. Amended to also
#include the histogram of Tesco for comparison 
plt.subplot(2, 2, 4)
plt.hist(df_VOD["ann_return"], bins=15)
plt.hist(df_TSCO["ann_return"], bins=15, alpha=0.7)
plt.ylabel("Vodaphone")
plt.legend(["Vodaphone", "Tesco"], prop={"size": 6})

plt.savefig("annual revenue histograms.png")
#creating figure 2, including boxplot of all four companies annual returns
plt.figure(2)
plt.boxplot([df_BCS["ann_return"], df_BP["ann_return"],
             df_TSCO["ann_return"], df_VOD["ann_return"]],
             labels=["Barclays", "BP", "Tesco", "Vodaphone"])
plt.ylabel("Annual Returns (%)")
plt.savefig("annual revenue boxplot.png")
plt.show()
