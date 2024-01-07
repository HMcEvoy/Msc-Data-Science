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
useful_land_cols = ["Country Name", "1990", "1991", "1992", "1993", "1994",
                    "1995", "1996", "1997", "1998", "1999", "2000", "2001",
                    "2002", "2003", "2004","2005", "2006", "2007", "2008",
                    "2009", "2010", "2011", "2012", "2013", "2014", "2015",
                    "2016", "2017", "2018", "2019", "2020", "2021"]

useful_pop_cols = ["Country Name", "1960", "1961", "1962", "1963", "1964",
                   "1965", "1966", "1967", "1968", "1969", "1970", "1971",
                   "1972", "1973", "1974", "1975", "1976", "1977", "1978",
                   "1979","1980", "1981", "1982", "1983", "1984", "1985",
                   "1986", "1987", "1988", "1989", "1990", "1991", "1992",
                   "1993", "1994", "1995", "1996", "1997", "1998", "1999",
                   "2000", "2001", "2002", "2003", "2004","2005", "2006",
                   "2007", "2008", "2009", "2010", "2011", "2012", "2013",
                   "2014", "2015", "2016", "2017", "2018", "2019", "2020",
                   "2021", "2022"]

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

df_poptot = pd.read_csv("API_SP.POP.TOTL_DS2_en_csv_v2_6298256.csv",
                        header=1, usecols=useful_pop_cols, skiprows=[0,3])
df_poptot = df_poptot.fillna(0)
df_poptot, df_poptot_t = create_dfs(df_poptot)

df_popgro = pd.read_csv("API_SP.POP.GROW_DS2_en_csv_v2_6298705.csv",
                        header=1, usecols=useful_pop_cols, skiprows=[0,3])
df_popgro = df_popgro.fillna(0)
df_popgro, df_popgro_t = create_dfs(df_popgro)

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

#Defining the BRICs countries as a variable, then using that along with 
#slicing to create smaller dataframes for visualisation in figure 4
brics = ["Brazil", "Russian Federation", "India", "China", "South Africa"]

brics_poptot = df_poptot_t[brics]
brics_popgro = df_popgro_t[brics]

#Creating a list of years for the x-axis of figures 4 and 5
years = np.arange(1960, 2022)

#Creating subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

#Plotting figure 1
ax1.pie(pie1992_val, explode=explode, labels=labels,
       colors=["wheat", "forestgreen", "slategrey"], autopct="%1.1f%%")
ax1.set_title("Fig 1: World Land Area 1992")

#Plotting figure 2
ax2.pie(pie2021_val, explode=explode, labels=labels,
       colors=["wheat", "forestgreen", "slategrey"], autopct="%1.1f%%")
ax2.set_title("Fig 2: World Land Area 2021")

#Plotting figure 3
ax3.scatter(df_pop5_t.loc["2000"], df_land5_t.loc["2000"], c="lightseagreen",
            label="2000", alpha=0.6)
ax3.scatter(df_pop5_t.loc["2015"], df_land5_t.loc["2015"], c="seagreen",
            label="2015", alpha=0.6)
ax3.set_ylim(0)
ax3.set_xlim(0)
ax3.set_xlabel("\n".join(wrap("Population living in areas where elevation is "
                              "below 5 meters (% of total population)",
                              width=50)))
ax3.set_ylabel("\n".join(wrap("Land area where elevation is below 5 meters "
               "(% of total land area)", width=40)))
ax3.set_title("\n".join(wrap("Fig 3: Comparison of Land Area and Population "
                             "in Areas where Elevation is below 5 meters",
                             width=50)))
ax3.legend(loc="center right")

#Plotting combined figures 4 and 5
ax4.plot(years, brics_poptot, label=brics)
ax4.set_xlim(1960, 2020)
ax4.set_xlabel("Year")
ax4.set_ylim(0, 1800000000)
ax4.set_yticks(ax4.get_yticks())
ax4.set_yticklabels(["{:.0f}".format(brics_poptot/100000000)
                     for brics_poptot in ax4.get_yticks()])
ax4.set_ylabel("Population —— (hundred millions)")
ax4.set_title("\n".join(wrap("Fig 4: Total Population and Population Growth "
                             "for the BRICS countries 1960 to 2020",
                             width=50)))
ax4.legend(loc='upper left')
ax5 = ax4.twinx()
ax5.plot(years, brics_popgro, "--")
ax5.set_ylabel("Population Growth - - - (%)")
ax5.set_ylim(-1, 6)

#Fixing overall format for subplots and saving as a png file
plt.tight_layout()
plt.show()