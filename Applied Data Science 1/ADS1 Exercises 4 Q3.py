# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:13:45 2023

@author: harle
"""
#importing pyplot and numpy modules
import matplotlib.pyplot as plt
import numpy as np

#data for plots - market capitalisation for these four companies 
MC = np.array([33367, 68785, 20979, 29741])
Comp = ["Barclays", "BP", "Tesco", "Vodaphone"]

#pie chart showing market capitalisation of the four companies relative
#to one another
plt.figure(1)
plt.pie(MC, labels=Comp)
plt.title("Market Capitalisation")

#defining market capitalisation of all four companies relative to the 
#total market capitalisation
TMC = 1814000
MC = MC/TMC

#pie chart of the market capitalisation of all four companies relative
#to the total market capitalisation
plt.figure(2)
plt.pie(MC, labels=Comp, normalize=False)
plt.title("Market Capitalisation as a Fraction of Total Market Capitlisation")

#redefining MC for the following bar chart
MC = np.array([33367, 68785, 20979, 29741])

#figure containing bar chart of all four companies market capitalisation
#relative to each other
plt.figure(3)
plt.bar(Comp, MC, width=0.8)
plt.title("Market Capitalisation")
plt.xlabel("Companies")
plt.ylabel("Market Capitalisation Mill. £")

plt.show
