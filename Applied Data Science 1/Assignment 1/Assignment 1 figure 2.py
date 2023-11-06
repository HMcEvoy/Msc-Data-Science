# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:35:57 2023

@author: harle
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir("C:\\Users\\harle\\Data Science\\Applied Data Science\\Data")

world_pop = pd.read_excel("Assignment_2_figure_2_"
                          "data_population.xlsx")

#removes columns that are not necessary
world_pop = world_pop.drop(["Country Code", "Indicator Name",
                            "Indicator Code"], axis=1)

#transposes and removes excess index row
world_pop = world_pop.transpose()
world_pop.columns = world_pop.iloc[0]
world_pop = world_pop[1:]

#this first selects a single row from the dataframe (the year 1960), then sorts
#the row by giving the 5 highest populations as a list, then indexes the
#selected row by the new list and finally converts it into a 1D np array so
#that it can be plotted into a pie chart
pop_1960 = (world_pop.iloc[[0], :])
order_1960 = np.argsort(-pop_1960.values, axis=1)[:, :5]
top_pop_1960 = pop_1960.columns[order_1960].tolist()[0]
top_1960 = pop_1960[top_pop_1960]
top_1960 = top_1960.values.flatten()


#this first selects a single row from the dataframe (the year 2023), then sorts
#the row by giving the 5 highest populations as a list, then indexes the
#selected row by the new list and finally converts it into a 1D np array so
#that it can be plotted into a pie chart
pop_2023 = (world_pop.iloc[[-1], :])
order_2023 = np.argsort(-pop_2023.values, axis=1)[:, :5]
top_pop_2023 = pop_2023.columns[order_2023].tolist()[0]
top_2023 = pop_2023[top_pop_2023]
top_2023 = top_2023.values.flatten()

#providing labels for the pie charts
labels_1960 = ["China", "India", "United States", "Russian Federation",
               "Japan"]

labels_2023 = ["India", "China", "United States", "Indonesia", "Pakistan"]

#assigning colours to avoid confusion between pie charts
country_colors_dict = {"China" : "C0", "India" : "C1", "United States" : "C3", 
           "Russian Federation" : "C4", "Japan" : "C5", "Indonesia" : "C6",
           "Pakistan" : "C7"}

kvp_cc = country_colors_dict.items()

cc_data = list(kvp_cc)

country_colors = np.array(cc_data)

#plots a pie chart of the top 5 largest population countries in the year 1960
plt.figure(1)
plt.pie(top_1960, labels=labels_1960, colors=country_colors)
plt.title("The Top 5 Countries by Population in 1960")
#plots a pie chart of the top 5 largest population countries in the year 2023
plt.figure(2)
plt.pie(top_2023, labels=labels_2023)
plt.title("The Top 5 Countries by Population in 2023")

plt.show()
