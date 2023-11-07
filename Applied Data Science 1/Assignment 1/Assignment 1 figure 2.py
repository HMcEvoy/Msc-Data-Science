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

#converts the data into a pandas dataframe
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
top_pop_1960 = pop_1960.columns[order_1960[0]].tolist()
top_1960 = pop_1960[top_pop_1960]
top_1960 = top_1960.values.flatten()


#this first selects a single row from the dataframe (the year 2023), then sorts
#the row by giving the 5 highest populations as a list, then indexes the
#selected row by the new list and finally converts it into a 1D np array so
#that it can be plotted into a pie chart
pop_2022 = (world_pop.iloc[[-1], :])
order_2022 = np.argsort(-pop_2022.values, axis=1)[:, :5]
top_pop_2022 = pop_2022.columns[order_2022[0]].tolist()
top_2022 = pop_2022[top_pop_2022]
top_2022 = top_2022.values.flatten()

#providing labels for the pie charts
labels_1960 = ["China", "India", "United States", "Russian Federation",
               "Japan"]

labels_2022 = ["India", "China", "United States", "Indonesia", "Pakistan"]


#plots a pie chart of the top 5 largest population countries in the year 1960
def pieplot_1960(world_pop):
    plt.figure(1)
    plt.pie(top_1960, labels=labels_1960,
            colors=["C0", "C1", "C2", "C3", "C4"], autopct="%.2f%%")
    plt.title("The Top 5 Countries by Population in 1960")
    plt.savefig("Assignment_1_fig_2.1.png", bbox_inches="tight")
    plt.show()
    
    return



#plots a pie chart of the top 5 largest population countries in the year 2023
def pieplot_2023(world_pop):
    plt.figure(2)
    plt.pie(top_2022, labels=labels_2022,
            colors=["C1", "C0", "C2", "C5", "C6"], autopct="%.2f%%")
    plt.title("The Top 5 Countries by Population in 2022")
    plt.savefig("Assignment_1_fig_2.2.png", bbox_inches="tight")
    plt.show()
    
    return

pieplot_1960(world_pop)

pieplot_2023(world_pop)

