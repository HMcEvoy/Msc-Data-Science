# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:30:52 2023

@author: harle
"""
#importing modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir("C:\\Users\\harle\\Data Science\\Applied Data Science\\Data")

#importing the data from excel to pandas dataframe
renewable_data = pd.read_excel("Assignment_1_figure_3_data_renewable_energy"
                               ".xlsx")

#creating variables for x and y axes from dataframe
year = np.arange(1990, 2015)
hydro = renewable_data["Hydro"]
wind = renewable_data["Wind"]
solar_pv = renewable_data["Solar PV"]
bioenergy = renewable_data["Bioenergy"]


def barplot(data):
    """This function creates bar plots of the electrical energy generation
    in gigawatts (GWe) of the UK for four type of renewable energy; hydro,
    wind, solar photovoltaic (pv) and bioenergy. The function then saves the
    plot as a png
    """
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].bar(year, hydro, color="b", alpha=0.4)
    axs[0, 0].set_title("Hydro")
    axs[0, 1].bar(year, wind, color="k", alpha=0.3)
    axs[0, 1].set_title("Wind")
    axs[1, 0].bar(year, solar_pv, color="orange", alpha=0.7)
    axs[1, 0].set_title("Solar PV")
    axs[1, 1].bar(year, bioenergy, color="g", alpha=0.5)
    axs[1, 1].set_title("Bioenergy")
    fig.suptitle("UK Renewable Energy Capacity 1990-2014")
    fig.supxlabel("Year")
    fig.supylabel("Electrical Energy Capacity (GWe)")
    fig.tight_layout()
    plt.savefig("Assignment_1_fig_3.png", bbox_inches="tight")
    plt.show()

#calling the function to create the bar plots
barplot(renewable_data)

