
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import stats as stats

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

#Creating list variables containing the various useful columns
#of each dataframe (ignoring columns of string data or empty values)
useful_rec = ["Country Name", "1990", "1991", "1992", "1993", "1994", "1995",
              "1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003",
              "2004","2005", "2006", "2007", "2008", "2009", "2010", "2011", 
              "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019",
              "2020"]

useful_reo = ["Country Name", "1990", "1991", "1992", "1993", "1994", "1995",
              "1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003",
              "2004","2005", "2006", "2007", "2008", "2009", "2010", "2011", 
              "2012", "2013", "2014", "2015"]

useful_eu = ["Country Name", "1971", "1972", "1973", "1974", "1975", "1976",
             "1977", "1978", "1979","1980", "1981", "1982", "1983", "1984",
             "1985", "1986", "1987", "1988", "1989", "1990", "1991", "1992",
             "1993", "1994", "1995", "1996", "1997", "1998", "1999", "2000",
             "2001", "2002", "2003", "2004","2005", "2006", "2007", "2008",
             "2009", "2010", "2011", "2012", "2013", "2014"]

useful_ate = ["Country Name", "1990", "1991", "1992", "1993", "1994", "1995",
              "1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003",
              "2004","2005", "2006", "2007", "2008", "2009", "2010", "2011", 
              "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019",
              "2020", "2021"]

#Creating a list of years for use in any time series figures
years = np.arange(1971, 2021)


#Importing the data into pandas dataframes, skipping unnecessary columns and
#rows, then running the dataframe through the create_dfs function to get
#easiliy plottable dataframes. Note that df_eu and df_epc both use useful_eu
#because they have the same range of useful columns, visible in the csv data
df_rec = pd.read_csv("renewable_energy_consumption.csv", header=1,
                     usecols=useful_rec, skiprows=[0, 3])
df_rec = df_rec.fillna(0)
df_rec, df_rec_t = create_dfs(df_rec)

df_reo = pd.read_csv("renewable_electricity_output.csv", header=1,
                     usecols=useful_reo, skiprows=[0, 3])
df_reo = df_reo.fillna(0)
df_reo, df_reo_t = create_dfs(df_reo)

df_eu = pd.read_csv("energy_use.csv", header=1, usecols=useful_eu, 
                    skiprows=[0, 3])
df_eu = df_eu.fillna(0)
df_eu, df_eu_t = create_dfs(df_eu)

df_epc = pd.read_csv("electrical_power_consumption.csv", header=1,
                     usecols=useful_eu, skiprows=[0, 3])
df_epc = df_epc.fillna(0)
df_epc, df_epc_t = create_dfs(df_epc)

df_ate = pd.read_csv("access_to_electricity.csv", header=1, usecols=useful_ate,
                     skiprows=[0, 3])
df_ate = df_ate.fillna(0)
df_ate, df_ate_t = create_dfs(df_ate)


plt.figure(1)
plt.bar(years[0:-7], df_epc_t["High income"], label="High income",
        color="purple")
plt.bar(years[0:-7], df_epc_t["Low & middle income"],
        label="Low & middle income", color="lime")
plt.xlabel("Year")
plt.ylabel("Electric power consumption (kWh per capita)")
plt.legend()
plt.title("Comparison of Electrical Consumption for High and Low & Middle"
          " Incomes")

reo2003 = [df_reo_t.loc["2003"]["World"]]
pie2003_val = np.array([reo2003[0], 100 - reo2003[0]])

reo2015 = [df_reo_t.loc["2015"]["World"]]
pie2015_val = np.array([reo2015[0], 100 - reo2015[0]])
labels = ["Renewable Electricity Output", "Non-renewable Electricity Output"]
explode = (0, 0.1)

plt.figure(2)
plt.pie(pie2003_val, explode=explode, labels=labels, colors=["forestgreen",
                                                             "darkred"],
        autopct="%1.1f%%", shadow=True)
plt.title("World Electrical Output 2003")

plt.figure(3)
plt.pie(pie2015_val, explode=explode, labels=labels, colors=["forestgreen",
                                                             "darkred"],
        autopct="%1.1f%%", shadow=True)
plt.title("World Electrical Output 2015")

brics = ["Brazil", "Russian Federation", "India", "China", "South Africa"]
brics1990 = [df_eu_t.loc["1990", brics]]
print(brics1990)

plt.figure(3)

plt.show





