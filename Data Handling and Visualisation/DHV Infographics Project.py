import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap

#Creating a function to return normal and transposed dataframes for later use
#in plots
def create_dfs(df):
    """
    This function takes a pandas dataframe of World Bank data and creates two
    new dataframes. The first is a copy of the original dataframe, and the
    second is a copy of the original dataframe transposed and tidied for use.

    Parameters
    ----------
    df : pandas dataframe
        An input dataframe created by reading a csv file from World Bank.

    Returns
    -------
    df1 : pandas dataframe
        A copy of the original dataframe.
    df2 : pandas dataframe
        A transposed copy of the original dataframe, with new column values
        and removed unneccesary rows resulting from transposing.
    """
    df1 = df.copy()
    df2 = pd.DataFrame.transpose(df)
    header = df2.iloc[0].values.tolist()
    df2.columns = header
    df2 = df2.iloc[2:]
    return df1, df2


#Selecting potential useful columns from the datasets
useful_land_cols = ["Country Name", "1990", "1991", "1992", "1993", "1994", "1995",
              "1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003",
              "2004","2005", "2006", "2007", "2008", "2009", "2010", "2011", 
              "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019",
              "2020", "2021"]

useful_5_cols = ["Country Name", "1990", "2000", "2015"]

#Reading the datasets using pandas, removing the unnecessary rows and columns,
#filling empty values with a 0 and creating usable dataframes with the
#create_dfs function
df_agr = pd.read_csv("API_AG.LND.AGRI.ZS_DS2_en_csv_v2_6299921.csv", header=1,
                     usecols=useful_land_cols, skiprows=[0, 3])
df_agr = df_agr.fillna(0)
df_agr, df_agr_t = create_dfs(df_agr)

df_for = pd.read_csv("API_AG.LND.FRST.ZS_DS2_en_csv_v2_6299844.csv", header=1,
                     usecols=useful_land_cols, skiprows=[0, 3])
df_for = df_for.fillna(0)
df_for, df_for_t = create_dfs(df_for)

df_pop5 = pd.read_csv("API_EN.POP.EL5M.ZS_DS2_en_csv_v2_6304536.csv",
                       header=1, usecols=useful_5_cols, skiprows=[0, 3])
df_pop5 = df_pop5.fillna(0)
df_pop5, df_pop5_t = create_dfs(df_pop5)

df_land5 = pd.read_csv("API_AG.LND.EL5M.ZS_DS2_en_csv_v2_6304336.csv",
                       header=1, usecols=useful_5_cols, skiprows=[0, 3])
df_land5 = df_land5.fillna(0)
df_land5, df_land5_t = create_dfs(df_land5)



#Slicing out the rows and columns needed for figures 1 and 2 and containing
#them within a numpy array
agr1992 = [df_agr_t.loc["1992"]["World"]]
for1992 = [df_for_t.loc["1992"]["World"]]
pie1992_val = np.array([agr1992[0], for1992[0],
                        100.0 - (agr1992[0] + for1992[0])], dtype=float)

agr2021 = [df_agr_t.loc["2021"]["World"]]
for2021 = [df_for_t.loc["2021"]["World"]]
pie2021_val = np.array([agr2021[0], for2021[0],
                        100.0 - (agr2021[0] + for2021[0])], dtype=float)

#Creating labels for figures 1 and 2, and defining which slices to explode for
#emphasis
labels = ["Agricultural Land", "Forest Land", "Other Land"]
explode = (0.1, 0.1, 0)

#Defining the BRICs countries as a variable for use in figures XXX
brics = ["Brazil", "Russian Federation", "India", "China", "South Africa"]

brics_pop5 = df_pop5_t[brics]
brics_land5 = df_land5_t[brics]

#plotting the figures
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.pie(pie1992_val, explode=explode, labels=labels,
       colors=["wheat", "forestgreen", "slategrey"], autopct="%1.1f%%",
        shadow=True)
ax1.set_title("World Land Area 1992")

ax2.pie(pie2021_val, explode=explode, labels=labels,
       colors=["wheat", "forestgreen", "slategrey"], autopct="%1.1f%%",
        shadow=True)
ax2.set_title("World Land Area 2021")

ax3.scatter(df_pop5_t.loc["2000"], df_land5_t.loc["2000"])
ax3.set_ylim(0)
ax3.set_xlim(0)
ax3.set_xlabel("\n".join(wrap("Population living in areas where elevation is "
                              "below 5 meters (% of total population)",
                              width=50)))
ax3.set_ylabel("\n".join(wrap("Land area where elevation is below 5 meters "
               "(% of total land area)", width=40)))
ax3.set_title("\n".join(wrap("Comparison of Land Area and Population in Areas "
                             "where Elevation is below 5 meters", width=50)))


plt.tight_layout()
plt.show()
