import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
import collections
import sklearn.preprocessing
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.animation as animation
import tempfile
from PIL import Image

first_date = dt.date(2020, 3, 1)

def main():
    df = download_data()
    plot_by_country(df=df, ctype='deaths')

def download_data():
    covid_raw_pd = pd.read_csv('https://opendata.ecdc.europa.eu/covid19/casedistribution/csv')
    # covid_raw_pd = pd.read_csv('/Users/daniel/Downloads/cv.csv')
    cols_to_drop = ['day', 'month', 'year', 'geoId', 'countryterritoryCode', 'continentExp']
    covid_raw_pd = covid_raw_pd[covid_raw_pd.columns.drop(cols_to_drop)]
    covid_raw_pd['dateRep'] = pd.to_datetime(covid_raw_pd['dateRep'], format=r'%d/%m/%Y')
    return covid_raw_pd

def get_all_countries(df):
    return df.loc[:, 'countriesAndTerritories'].drop_duplicates().to_list()

def get_eu_countries():
    return pd.Series(['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden'])

def country_series(df, country, ctype, cumsum=False, log=False):
    country_df = df.loc[df['countriesAndTerritories']  == country]
    cases = pd.Series(data=country_df.loc[:, ctype].values, index=country_df.loc[:, 'dateRep'], dtype=np.int32)
    cases = cases.iloc[::-1]
    
    if cumsum: 
        cases = pd.Series.cumsum(cases)
    if log: 
        cases = np.log(cases)
    
    return cases

def normalised_progression_by_country(df, countries, ctype, log=False):
    results = pd.DataFrame(index=pd.date_range(start=first_date, end='today'), columns=countries)
    for country in countries:
        sr = country_series(df, country, ctype, cumsum=True, log=log)
        sr /= df[df.countriesAndTerritories == country].iloc[0].popData2018
        results[country] = sr
    results = results.fillna(0)
    normalised = sklearn.preprocessing.normalize(results.to_numpy())
    results = pd.DataFrame(data=normalised, index=results.index, columns=results.columns)
    return results

def plot_by_country(df, ctype):
    df = normalised_progression_by_country(df, get_all_countries(df), ctype)
    countries_shp = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
    cmap = matplotlib.cm.get_cmap('Spectral')

    saved_figs = []
    limit=5
    for i in range(df.shape[0]):
        tfile = tempfile.TemporaryFile()
        ax = plt.axes(projection=ccrs.PlateCarree(), label=str(i))
        for country in shpreader.Reader(countries_shp).records():
            c = clean_country(country.attributes['NAME_LONG'])
            if c == None: 
                rgba = (0.5, 0.5, 0.5, 1.0)
            else:
                rgba = cmap(df[c][i])
            ax.add_geometries([country.geometry], ccrs.PlateCarree(), facecolor=rgba, label=country.attributes['NAME_LONG'])
        plt.title(str(df.index[i]).split(' ')[0])
        plt.savefig(tfile, dpi=400)
        saved_figs.append(tfile)

    plt.close()
    fig = plt.figure()
    ims = []
    for temp_img in saved_figs:
        X = Image.open(temp_img)
        ims.append([plt.imshow(X, animated=True)])
    ani = animation.ArtistAnimation(fig, ims, interval=800, blit=True, repeat_delay=1000)
    plt.axis('off')
    plt.show()
    ani.save('/Users/daniel/Desktop/animation.gif', writer='imagemagick', fps=2, dpi=400)
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=100000)
    # ani.save('/Users/daniel/Desktop/animation.mp4', writer=writer, dpi=400)

def clean_country(country):
    try:
        return {
            'Tanzania': 'United_Republic_of_Tanzania',
            # 'Western Sahara': 'Western_Sahara',
            'United States': 'United_States_of_America',
            'Papua New Guinea': 'Papua_New_Guinea',
            'Democratic Republic of the Congo': 'Democratic_Republic_of_the_Congo',
            'Dominican Republic': 'Dominican_Republic',
            'Russian Federation': 'Russia',
            # 'Falkland Islands': 'Falkland_Islands_(Malvinas)',
            'French Southern and Antarctic Lands': None,
            'Timor-Leste': 'Timor_Leste',
            'South Africa': 'South_Africa',
            'Lesotho': None,
            'Costa Rica': 'Costa_Rica',
            'El Salvador': 'El_Salvador',
            'Puerto Rico': 'Puerto_Rico',
            'Côte d\'Ivoire': 'Cote_dIvoire',
            'Guinea-Bissau': 'Guinea_Bissau',
            'Sierra Leone': 'Sierra_Leone',
            'Burkina Faso': 'Burkina_Faso',
            'Central African Republic': 'Central_African_Republic',
            'Republic of the Congo': 'Congo',
            'Equatorial Guinea': 'Equatorial_Guinea',
            'eSwatini': 'Eswatini',
            'The Gambia': 'Gambia',
            'United Arab Emirates': 'United_Arab_Emirates',
            'Vanuatu': None,
            'Lao PDR': 'Laos',
            'Dem. Rep. Korea': None,
            'Republic of Korea': 'South_Korea',
            'Turkmenistan': None,
            'New Caledonia': 'New_Caledonia',
            'Solomon Islands': None,
            'New Zealand': 'New_Zealand',
            'Sri Lanka': 'Sri_Lanka',
            'United Kingdom': 'United_Kingdom',
            'Brunei Darussalam': 'Brunei_Darussalam',
            'Czech Republic': 'Czechia',
            'Saudi Arabia': 'Saudi_Arabia',
            'Antarctica': None,
            'Northern Cyprus': None,
            'Somaliland': None,
            'Bosnia and Herzegovina': 'Bosnia_and_Herzegovina',
            'Macedonia': 'North_Macedonia',
            'Trinidad and Tobago': 'Trinidad_and_Tobago',
            'South Sudan': 'South_Sudan',

            'Anguilla': None,
            'Bonaire, Saint Eustatius and Saba': None,
            'Eritrea': None,
            'Falkland Islands': None,
            'Western Sahara': None,
        }[country]
    except KeyError:
        return country

if __name__ == '__main__':
    main()