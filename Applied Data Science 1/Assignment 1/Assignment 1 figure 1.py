
#importing modules and fixing working directory
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir("C:\\Users\\harle\\Data Science\\Applied Data Science\\Data")

#importing the data from an excel file into a pandas dataframe
uk_max_temp = pd.read_excel("Assignment_1_figure_1_data_max_temp_UK.xlsx")

#saving the specific columns that we want to work with from the dataframe as
#specific variables
winter = uk_max_temp["win"]
summer = uk_max_temp["sum"]
year = uk_max_temp["year"]

plt.figure()

#using pyplot to plot a line plot with two sets of data, winter and summer,
#against the year
plt.plot(year, winter)
plt.plot(year, summer)

#labelling the line plot
plt.xlabel("Year")
plt.ylabel("Mean Max Temp °C")

#making the x axis neater by stepping consistently and angling for clarity
plt.xticks(year[::10], rotation=45)

#removing excess white space at the edges of the data inside the plot
plt.margins(x=0)

#adding a relevant legend
plt.legend(["Winter", "Summer"])

#saving the figure as a PNG
plt.savefig("Assignment_1_fig_1.png", bbox_inches="tight")
plt.show()

