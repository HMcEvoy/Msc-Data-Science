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

def fitting_plot(df_year, df, range_years, data_logistic, data_low, data_up,
                 xlim, ylim, title):
    """
    This function creates a fitted line plot, showing the data, a predictive 
    line of best fit and a highlighted area showing how the data may diverge
    from the line of best fit. This function is specifically for producing
    lines of best fit for worldbank GPP per capita data

    Parameters
    ----------
    df_year : A pandas dataframe
        This is a dataframe of years formed from the index of a transposed
        dataframe created by the create_dfs function.
    df : A pandas dataframe
        This is a dataframe formed from the single column of a transposed
        dataframe created by the create_dfs function.
    range_years : A NumPy ndarray
        This is a 1D array of years, spanning from the start of the dataset to
        a predicted end point.
    data_logistic : A NumPy ndarray
        This is a 1D array produced by the logistics function from range_years
        and the optimised curve produced by scipy.optimize curve_fit function.
    data_low : A NumPy ndarray
        The lower bound of the data_logistic, reliant on the sigma derived 
        from the error_prop function.
    data_up : A NumPy ndarray
        The upper bound of the data_logistic, reliant on the sigma derived
        from the error_prop function.
    xlim : A list
        This is a list of two integers denoting the starting and ending values
        for the x-axis
    ylim : A list
        This is a list of two integers denoting the starting and ending values
        for the y-axis
    title : String
    

    Returns
    -------
    fig : a matplotlib figure
        This figure shows the data, the predicted line of best fit and provides
        a highlighted area showing how the data may diverge from the line of
        best fit.

    """
    fig = plt.figure(figsize=(10, 10))
    plt.plot(df_year, df, label="data")
    plt.plot(range_years, data_logistic, label="fit")
    plt.fill_between(range_years, data_low, data_up, alpha=0.5, color="y")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("Year")
    plt.ylabel("GDP per capita (current US$)")
    plt.title(title)
    plt.show()
    
    return fig

#defining the potentially useful columns for the data for use in the cluster
# and fitting figures
useful_clu = ["Country Name", "1971", "1972", "1973", "1974", "1975", "1976",
              "1977", "1978", "1979","1980", "1981", "1982", "1983", "1984",
              "1985", "1986", "1987", "1988", "1989", "1990", "1991", "1992",
              "1993", "1994", "1995", "1996", "1997", "1998", "1999", "2000",
              "2001", "2002", "2003", "2004","2005", "2006", "2007", "2008",
              "2009", "2010", "2011", "2012", "2013", "2014"]

useful_gdp = ["Country Name", "1971", "1972", "1973", "1974", "1975", "1976",
              "1977", "1978", "1979","1980", "1981", "1982", "1983", "1984",
              "1985", "1986", "1987", "1988", "1989", "1990", "1991", "1992",
              "1993", "1994", "1995", "1996", "1997", "1998", "1999", "2000",
              "2001", "2002", "2003", "2004","2005", "2006", "2007", "2008",
              "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016",
              "2017", "2018", "2019", "2020", "2021", "2022"]

#reading the data into a pair of pandas dataframes for the cluster figures
df_el, df_el_t = create_dfs("API_EG.USE.ELEC.KH.PC_DS2_en_csv_v2_6300057.csv",
                            useful_clu)
df_gdp, df_gdp_t = create_dfs("API_NY.GDP.PCAP.CD_DS2_en_csv_v2_6298251.csv",
                              useful_gdp)

#defining some specific years from the data set for closer analysis for
#clustering
years_cluster = ["1990", "1995", "2000", "2005", "2010"]

#preparing the dataframe for clustering by slicing by years and ensuring the
#correct usable data type, as well as removing any rows with 0 values
years_el = df_el[years_cluster]
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


#china, switzerland, canada
#selecting specific years as Switzerland does not have any data before 1980
years_clu = ["1980", "1981", "1982", "1983", "1984", "1985", "1986", "1987",
             "1988", "1989", "1990", "1991", "1992", "1993", "1994", "1995",
             "1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003",
             "2004","2005", "2006", "2007", "2008", "2009", "2010", "2011",
             "2012", "2013", "2014"]

years_fit = ["1980", "1981", "1982", "1983", "1984", "1985", "1986", "1987",
             "1988", "1989", "1990", "1991", "1992", "1993", "1994", "1995",
             "1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003",
             "2004","2005", "2006", "2007", "2008", "2009", "2010", "2011",
             "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019",
             "2020", "2021", "2022"]


#slicing 3 countries from the gdp dataframe - China, Switzerland and Canada -
#one for each cluster identified from the cluster figure from the electricity
#consumption dataframe for further analysis via fitting.
df_gdp_t = df_gdp_t.loc[years_fit]
df_gdp_ch = df_gdp_t["China"]
df_gdp_sw = df_gdp_t["Switzerland"]
df_gdp_ca = df_gdp_t["Canada"]
df_ch = pd.DataFrame(df_gdp_ch)
df_sw = pd.DataFrame(df_gdp_sw)
df_ca = pd.DataFrame(df_gdp_ca)

#converting the years from the index of the dataframe into integers and storing
#them as a dataframe
year_int = [int(x) for x in df_gdp_t.index]
year_df = pd.DataFrame(year_int)


#creating an integer range of years equal to the dataframe
years_index = np.arange(1980, 2023)

#using curve_fit to find the key values opt and covar for each country
chopt, chcovar = opt.curve_fit(logistics, years_index, df_gdp_ch, 
                               p0=(10000, 0.3, 2010))
swopt, swcovar = opt.curve_fit(logistics, years_index, df_gdp_sw,
                               p0=(10000, 0.3, 2010))
caopt, cacovar = opt.curve_fit(logistics, years_index, df_gdp_ca,
                               p0=(10000, 0.3, 2010))

#creating an arry of years and then creating the logistic for each dataset
#using the logistics function
years = np.linspace(1980, 2030)
ch_logistics = logistics(years, *chopt)
sw_logistics = logistics(years, *swopt)
ca_logistics = logistics(years, *caopt)

#calculating the sigma, lower and upper values for each country
ch_sigma = err.error_prop(years, logistics, chopt, chcovar)
ch_low = ch_logistics - ch_sigma
ch_up = ch_logistics + ch_sigma

sw_sigma = err.error_prop(years, logistics, swopt, swcovar)
sw_low = sw_logistics - sw_sigma
sw_up = sw_logistics + sw_sigma

ca_sigma = err.error_prop(years, logistics, caopt, cacovar)
ca_low = ca_logistics - ca_sigma
ca_up = ca_logistics + ca_sigma

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


#plotting and saving figures of all three countries using the fitting_plot
#function
fig4 = fitting_plot(year_df, df_ch, years, ch_logistics, ch_low, ch_up,
                    [1980, 2030], [0, 16000],
                    "GDP per Capita forecast for China")

fig4.savefig("GDP per Capita forecast for China Plot.png", dpi=300)

fig5 = fitting_plot(year_df, df_sw, years, sw_logistics, sw_low, sw_up,
                    [1980, 2030], [0, 100000],
                    "GDP per Capita forecast for Switzerland")
fig5.savefig("GDP per Capita forecast for Switzerland Plot.png", dpi=300)

fig6 = fitting_plot(year_df, df_ca, years, ca_logistics, ca_low, ca_up,
                    [1980, 2030], [0, 75000],
                    "GDP per Capita forecast for Canada")
fig6.savefig("GDP per Capita forecast for Canada Plot.png", dpi=300)
