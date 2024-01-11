#importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import errors as err
import cluster_tools_copy as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt

#defining function to read, tidy and make two useable pandas dataframes from
#worldbank csv data files
def create_dfs(df, cols):
    """
    This function takes a csv file of World Bank data and creates two
    new dataframes. The first is a pandas copy of the original dataframe, and
    the second is a copy of the original dataframe transposed and tidied for
    use.

    Parameters
    ----------
    df : A csv file containing data
        An input dataframe created by reading a csv file from World Bank.
    cols : A list of strings
        This list corresponds to the useful columnns within the data set - 
        those columns that actually contain data
    
    Returns
    -------
    df1 : pandas dataframe
        A copy of the original dataframe.
    df2 : pandas dataframe
        A transposed copy of the original dataframe, with new column values
        and removed unneccesary rows resulting from transposing.
    """
    dfr = pd.read_csv(df, header=1, usecols=cols, skiprows=[0, 3])
    dfr = dfr.fillna(0)
    df1 = dfr.copy()
    df2 = pd.DataFrame.transpose(dfr)
    header = df2.iloc[0].values.tolist()
    df2.columns = header
    df2 = df2.iloc[1:]
    return df1, df2

def cluster_plot(nc, df_fit, df_min, df_max, x, y, xlab, ylab, title):
    """
    This function creates a cluster plot, highlighting clusters and marking 
    their centre points, shows the plot and saves it as a png

    Parameters
    ----------
    nc : Integer
        This is the number of clusters, which should be determined by finding
        the silhouette score beforehand.
    df_fit : A pandas dataframe
        A dataframe of fitted data produced by the ct.scaler function.
    df_min : a pandas series
        A series of minimum values produced by the ct.scaler function.
    df_max : A pandas series
        A series of maximum values produced by the ct.scaler function.
    x : A pandas series
        A series of one column from the original dataframe, produced by
        slicing.
    y : A pandas series
        A series of another column from the original dataframe, produced by
        slicing.
    xlab : String
        This is a string for the x label of the cluster plot.
    ylab : String
        This is a string for the y label of the cluster plot.
    title : String
        This is a string for the title of the cluster plot.

    Returns
    -------
    fig : A matplotlib figure
        This figure is a cluster plot produced using the input dataframe, 
        sklearns.cluster module method and the cluster tools provided by
        cluster_tools module

    """
    #clustering the fitted data
    kmeans = cluster.KMeans(n_clusters=nc, n_init=10)
    kmeans.fit(df_fit)
    
    #defining labels, cluster centers and plotting the scatter data
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(x, y, c=labels, cmap="cool")
    
    #backscaling the data to apply the clustered data to the true scale axes
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:,0]
    yc = scen[:,1]
    
    #plotting the cluster centers, assigning values to the various sections of
    #the plot such as labels and titles, tighting plot layout and showing the
    #final plot
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    plt.xlim(0, )
    plt.ylim(0, )
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    return fig

def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    f = scale * np.exp(growth * (t - 1970))
    return np.array(f, dtype=float)

def logistics(t, a, k, t0):
    """ Computes logistics function with scale and incr as free parameters
    """
    f = a / (1.0 + np.exp(-k * (t - t0)))
    return np.array(f, dtype=float)

#defining the potentially useful columns for the data for use in the cluster
# and fitting figures
useful_clu = ["Country Name", "1971", "1972", "1973", "1974", "1975", "1976",
              "1977", "1978", "1979","1980", "1981", "1982", "1983", "1984",
              "1985", "1986", "1987", "1988", "1989", "1990", "1991", "1992",
              "1993", "1994", "1995", "1996", "1997", "1998", "1999", "2000",
              "2001", "2002", "2003", "2004","2005", "2006", "2007", "2008",
              "2009", "2010", "2011", "2012", "2013", "2014"]

#reading the data into a pair of pandas dataframes for the cluster figures
df_el, df_el_t = create_dfs("API_EG.USE.ELEC.KH.PC_DS2_en_csv_v2_6300057.csv",
                            useful_clu)
df_gdp, df_gdp_t = create_dfs("API_NY.GDP.PCAP.CD_DS2_en_csv_v2_6298251.csv",
                              useful_clu)
"""
#defining some specific years from the data set for closer analysis for
#clustering
years_clu = ["1990", "1995", "2000", "2005", "2010"]

#preparing the dataframe for clustering by slicing by years and ensuring the
#correct usable data type, as well as removing any rows with 0 values
years_el = df_el[years_clu]
years_el = years_el.apply(pd.to_numeric)
years_el = years_el.loc[(years_el!=0).all(axis=1)]

#creating a sliced copy of the original dataframe for cluster fitting and
#manipulation
years_el_fit = years_el[["1990", "2005"]].copy()
years_el_fit, years_el_min, years_el_max = ct.scaler(years_el_fit)

#finding the silhouette values of the data to determine how many clusters are
#present
print("n   score")

for ic in range(2, 11):
    kmeans = cluster.KMeans(n_clusters=ic, n_init=10)
    kmeans.fit(years_el_fit)
    
    labels = kmeans.labels_
    print(ic, skmet.silhouette_score(years_el_fit, labels))
"""

#china, switzerland, canada
#selecting specific years as Switzerland does not have any data before 1980
years_fit = ["1980", "1981", "1982", "1983", "1984", "1985", "1986", "1987",
             "1988", "1989", "1990", "1991", "1992", "1993", "1994", "1995",
             "1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003",
             "2004","2005", "2006", "2007", "2008", "2009", "2010", "2011",
             "2012", "2013", "2014"]
"""
years_fit = [1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987,
             1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995,
             1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
             2004,2005, 2006, 2007, 2008, 2009, 2010, 2011,
             2012, 2013, 2014]
"""
years_index = np.arange(1980, 2015)
#slicing 3 countries from the gdp dataframe, one for each cluster identified
#from the cluster figure from the electricity consumption dataframe for further
#analysis via fitting. The data is also converted from an object to numeric
#for any relevant processes

df_gdp_t = df_gdp_t.loc[years_fit]
df_gdp_ch = df_gdp_t["China"]
df_ch = pd.DataFrame(df_gdp_ch)

year_int = [int(x) for x in df_gdp_t.index]
year_df = pd.DataFrame(year_int)


"""
#initial guess values
initial_guess_scale = 0.5
initial_guess_growth = 0.01

#attempting curve fitting
gdpopt, gdpcorr = opt.curve_fit(exp_growth, years_index, df_gdp_ch,
                                p0=(initial_guess_scale, initial_guess_growth))

print("Fit parameter", gdpopt)
df_gdp_ch["gdp_exp"] = exp_growth(years_index, *gdpopt)
df_gdp_exp = pd.DataFrame(df_gdp_ch["gdp_exp"])


plt.figure()
plt.plot(year_int, df_ch, label="data")
plt.plot(year_int, df_gdp_ch["gdp_exp"], label="fit")
plt.xlabel("Year")
plt.ylabel("GDP per capita (current US$)")
plt.title("GDP per capita for China")
plt.legend()
plt.show()
"""
#gdpopt = [10000, 0.3, 2010]
gdpopt, gdpcovar = opt.curve_fit(logistics, years_index, df_gdp_ch,
                                p0=(10000, 0.3, 2010))

print("Fit parameter", gdpopt)
years = np.linspace(1980, 2030)
gdp_logistics = logistics(years, *gdpopt)
#df_gdp_log = pd.DataFrame(df_gdp_ch["gdp_logistics"])

sigma = err.error_prop(years, logistics, gdpopt, gdpcovar)
low = gdp_logistics - sigma
up = gdp_logistics + sigma


plt.figure()
plt.plot(year_df, df_ch, label="data")
plt.plot(years, gdp_logistics, label="fit")
plt.fill_between(years, low, up, alpha=0.5, color="y")
plt.legend()
plt.title("logistics function")
plt.show()


"""
#creating a heatmap of correlation and saving it as a png
ct.map_corr(years_el, 10)
plt.savefig("Electrical Consumption Heatmap.png", dpi=300)
plt.show()

#creating a scatter matrix to more clearly show correlation and saving it as a
#png
pd.plotting.scatter_matrix(years_el, figsize=(10, 10))
plt.tight_layout()
plt.savefig("Electrical Consumption Scatter Matrix.png", dpi=300)
plt.show()

#plotting the cluster plot figure and saving it as a png
fig3 = cluster_plot(3, years_el_fit, years_el_min, years_el_max, 
                   years_el["1990"], years_el["2005"], "1990 (kWh per capita)",
                   "2005 (kWh per capita)",
                   "Clustering of Electrical Power Consumption")

fig3.savefig("Electrical Power Consumption Cluster Plot.png", dpi=300)
"""

