
#importing modules and fixing working directory
import pandas as pd
import matplotlib.pyplot as plt
import os

#changing to the correct directory
os.chdir("C:\\Users\\harle\\Data Science\\Applied Data Science\\Data")

#importing the data from an excel file into a pandas dataframe
uk_max_temp = pd.read_excel("Assignment_1_figure_1_data_max_temp_UK.xlsx")

def lineplot(uk_max_temp):
    """Creates a line plot of the mean maximum temperature against year for
    the seasons winter and summer, including labels and legend. The function
    then saves the plot as a png file
    """
    plt.figure()
    plt.plot(year, winter)
    plt.plot(year, summer)
    plt.xlabel("Year")
    plt.ylabel("Mean Max Temp °C")
    plt.xticks(year[::10], rotation=45)
    plt.margins(x=0)
    plt.legend(["Winter", "Summer"])
    plt.savefig("Assignment_1_fig_1.png", bbox_inches="tight")
    plt.show()
    
    return

#calling the function to create the plot and save the png
lineplot(uk_max_temp)

#saving the columns that we want to work with from the dataframe as
#specific variables
winter = uk_max_temp["win"]
summer = uk_max_temp["sum"]
year = uk_max_temp["year"]
