# Analysis-of-Country-GDP-and-Forecast-for-the-USA
First of all, we started by getting the necessary libraries and our data set from its place.
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import geopandas
import pycountry
import plotly.express as ex

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("gdp_csv.csv")

df.head()
df.tail()
df.describe().T


df.isnull().sum()
No missing values.
Sum = df['Value'].groupby(df['Country Code']).sum()
first_15 = Sum.sort_values(ascending=True)[:15]
first_15

This code block is used to display the data in the “first_15” variable as a bar plot, plot() function takes the value “bar” into the “kind” parameter to determine the chart type. In addition, the xlim parameter limits the minimum and maximum values ​​to be displayed on the graph. However, this code will throw an error because the xlim parameter is set to an incorrect value. The color of the bars is determined by the color parameter. In this code, the bar color is set to red.
Finally, the chart is displayed with the “plt.show()” function.
first_15.plot(kind = 'bar',xlim=10, color='red')
plt.show()

df["Country Name"].unique()
array([‘Arab World’, ‘Caribbean small states’,
‘Central Europe and the Baltics’, ‘Early-demographic dividend’,
‘East Asia & Pacific’,
‘East Asia & Pacific (excluding high income)’,
‘East Asia & Pacific (IDA & IBRD countries)’, ‘Euro area’,
‘Europe & Central Asia’,
‘Europe & Central Asia (excluding high income)’,
‘Europe & Central Asia (IDA & IBRD countries)’, ‘European Union’,
‘Fragile and conflict affected situations’,
‘Heavily indebted poor countries (HIPC)’, ‘High income’,
‘IBRD only’, ‘IDA & IBRD total’, ‘IDA blend’, ‘IDA only’,
‘IDA total’, ‘Late-demographic dividend’,
‘Latin America & Caribbean’,
‘Latin America & Caribbean (excluding high income)’,
‘Latin America & the Caribbean (IDA & IBRD countries)’,
‘Least developed countries: UN classification’,
‘Low & middle income’, ‘Low income’, ‘Lower middle income’,
‘Middle East & North Africa’,
‘Middle East & North Africa (excluding high income)’,
‘Middle East & North Africa (IDA & IBRD countries)’,
‘Middle income’, ‘North America’, ‘OECD members’,
‘Other small states’, ‘Pacific island small states’,
‘Post-demographic dividend’, ‘Pre-demographic dividend’,
‘Small states’, ‘South Asia’, ‘South Asia (IDA & IBRD)’,
‘Sub-Saharan Africa’, ‘Sub-Saharan Africa (excluding high income)’,
‘Sub-Saharan Africa (IDA & IBRD countries)’, ‘Upper middle income’,
‘World’, ‘Afghanistan’, ‘Albania’, ‘Algeria’, ‘American Samoa’,
‘Andorra’, ‘Angola’, ‘Antigua and Barbuda’, ‘Argentina’, ‘Armenia’,
‘Aruba’, ‘Australia’, ‘Austria’, ‘Azerbaijan’, ‘Bahamas, The’,
‘Bahrain’, ‘Bangladesh’, ‘Barbados’, ‘Belarus’, ‘Belgium’,
‘Belize’, ‘Benin’, ‘Bermuda’, ‘Bhutan’, ‘Bolivia’,
‘Bosnia and Herzegovina’, ‘Botswana’, ‘Brazil’,
‘Brunei Darussalam’, ‘Bulgaria’, ‘Burkina Faso’, ‘Burundi’,
‘Cabo Verde’, ‘Cambodia’, ‘Cameroon’, ‘Canada’, ‘Cayman Islands’,
‘Central African Republic’, ‘Chad’, ‘Channel Islands’, ‘Chile’,
‘China’, ‘Colombia’, ‘Comoros’, ‘Congo, Dem. Rep.’, ‘Congo, Rep.’,
‘Costa Rica’, “Cote d’Ivoire”, ‘Croatia’, ‘Cuba’, ‘Cyprus’,
‘Czech Republic’, ‘Denmark’, ‘Djibouti’, ‘Dominica’,
‘Dominican Republic’, ‘Ecuador’, ‘Egypt, Arab Rep.’, ‘El Salvador’,
‘Equatorial Guinea’, ‘Eritrea’, ‘Estonia’, ‘Ethiopia’,
‘Faroe Islands’, ‘Fiji’, ‘Finland’, ‘France’, ‘French Polynesia’,
‘Gabon’, ‘Gambia, The’, ‘Georgia’, ‘Germany’, ‘Ghana’, ‘Greece’,
‘Greenland’, ‘Grenada’, ‘Guam’, ‘Guatemala’, ‘Guinea’,
‘Guinea-Bissau’, ‘Guyana’, ‘Haiti’, ‘Honduras’,
‘Hong Kong SAR, China’, ‘Hungary’, ‘Iceland’, ‘India’, ‘Indonesia’,
‘Iran, Islamic Rep.’, ‘Iraq’, ‘Ireland’, ‘Isle of Man’, ‘Israel’,
‘Italy’, ‘Jamaica’, ‘Japan’, ‘Jordan’, ‘Kazakhstan’, ‘Kenya’,
‘Kiribati’, ‘Korea, Rep.’, ‘Kosovo’, ‘Kuwait’, ‘Kyrgyz Republic’,
‘Lao PDR’, ‘Latvia’, ‘Lebanon’, ‘Lesotho’, ‘Liberia’, ‘Libya’,
‘Liechtenstein’, ‘Lithuania’, ‘Luxembourg’, ‘Macao SAR, China’,
‘Macedonia, FYR’, ‘Madagascar’, ‘Malawi’, ‘Malaysia’, ‘Maldives’,
‘Mali’, ‘Malta’, ‘Marshall Islands’, ‘Mauritania’, ‘Mauritius’,
‘Mexico’, ‘Micronesia, Fed. Sts.’, ‘Moldova’, ‘Monaco’, ‘Mongolia’,
‘Montenegro’, ‘Morocco’, ‘Mozambique’, ‘Myanmar’, ‘Namibia’,
‘Nauru’, ‘Nepal’, ‘Netherlands’, ‘New Caledonia’, ‘New Zealand’,
‘Nicaragua’, ‘Niger’, ‘Nigeria’, ‘Northern Mariana Islands’,
‘Norway’, ‘Oman’, ‘Pakistan’, ‘Palau’, ‘Panama’,
‘Papua New Guinea’, ‘Paraguay’, ‘Peru’, ‘Philippines’, ‘Poland’,
‘Portugal’, ‘Puerto Rico’, ‘Qatar’, ‘Romania’,
‘Russian Federation’, ‘Rwanda’, ‘Samoa’, ‘San Marino’,
‘Sao Tome and Principe’, ‘Saudi Arabia’, ‘Senegal’, ‘Serbia’,
‘Seychelles’, ‘Sierra Leone’, ‘Singapore’, ‘Slovak Republic’,
‘Slovenia’, ‘Solomon Islands’, ‘Somalia’, ‘South Africa’,
‘South Sudan’, ‘Spain’, ‘Sri Lanka’, ‘St. Kitts and Nevis’,
‘St. Lucia’, ‘St. Vincent and the Grenadines’, ‘Sudan’, ‘Suriname’,
‘Swaziland’, ‘Sweden’, ‘Switzerland’, ‘Syrian Arab Republic’,
‘Tajikistan’, ‘Tanzania’, ‘Thailand’, ‘Timor-Leste’, ‘Togo’,
‘Tonga’, ‘Trinidad and Tobago’, ‘Tunisia’, ‘Turkey’,
‘Turkmenistan’, ‘Tuvalu’, ‘Uganda’, ‘Ukraine’,
‘United Arab Emirates’, ‘United Kingdom’, ‘United States’,
‘Uruguay’, ‘Uzbekistan’, ‘Vanuatu’, ‘Venezuela, RB’, ‘Vietnam’,
‘Virgin Islands (U.S.)’, ‘West Bank and Gaza’, ‘Yemen, Rep.’,
‘Zambia’, ‘Zimbabwe’], dtype=object)
countries = ['Arab World', 'Caribbean small states',
       'Central Europe and the Baltics', 'Early-demographic dividend',
       'East Asia & Pacific',
       'East Asia & Pacific (excluding high income)',
       'East Asia & Pacific (IDA & IBRD countries)', 'Euro area',
       'Europe & Central Asia',
       'Europe & Central Asia (excluding high income)',
       'Europe & Central Asia (IDA & IBRD countries)', 'European Union',
       'Fragile and conflict affected situations',
       'Heavily indebted poor countries (HIPC)', 'High income',
       'IBRD only', 'IDA & IBRD total', 'IDA blend', 'IDA only',
       'IDA total', 'Late-demographic dividend',
       'Latin America & Caribbean',
       'Latin America & Caribbean (excluding high income)',
       'Latin America & the Caribbean (IDA & IBRD countries)',
       'Least developed countries: UN classification',
       'Low & middle income', 'Low income', 'Lower middle income',
       'Middle East & North Africa',
       'Middle East & North Africa (excluding high income)',
       'Middle East & North Africa (IDA & IBRD countries)',
       'Middle income', 'North America', 'OECD members',
       'Other small states', 'Pacific island small states',
       'Post-demographic dividend', 'Pre-demographic dividend',
       'Small states', 'South Asia', 'South Asia (IDA & IBRD)',
       'Sub-Saharan Africa', 'Sub-Saharan Africa (excluding high income)',
       'Sub-Saharan Africa (IDA & IBRD countries)', 'Upper middle income',
       'World']
#We filtred the Countries tha they didn't exist in the map dictionnary
df_country = df.loc[~df['Country Name'].isin(countries)]
len(df_country['Country Name'].unique())

df_country = df_country.replace('United States','United States of America')
Country name editing was done with replace()
A line chart is created showing the GDP values ​​of the countries in the df_country data frame over the years. Using the line() function, a separate line is drawn for each country on the graph and names showing which country it belongs to are added. A title is added on the chart using the annotations and update_layout() functions, and the chart is displayed with the fig.show() function.
The line_group parameter allows a separate line to be drawn for each country, and the hover_name parameter specifies the names of which country each line belongs to.
A list named annotations is created and a title is added on the chart using the update_layout() function.
annotations = []
fig = ex.line(df_country, x="Year", y="Value", color="Country Name",
              line_group="Country Name", hover_name="Country Name")
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='GDP over the years (1960 - 2016)',
                              font=dict(family='Arial',
                                        size=30),
                              showarrow=False))
fig.update_layout(annotations=annotations)
fig.show()

The world map data from the “naturalearth_lowres” dataset in the geopandas library is assigned to the variable named world. Then, the df_country dataframe created earlier is combined with the world dataset using the merge() function. This merge operation is performed by matching the country names in the name column with the country names in the Country Name column.
With the how parameter of the merge() function, the type of merge operation is specified and the “left” value is used to merge the df_country data frame and the world data set.
world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))

df_country_final = world.merge(df_country, how="left", left_on=['name'], right_on=['Country Name'])

world.name.unique()
array([‘Fiji’, ‘Tanzania’, ‘W. Sahara’, ‘Canada’,
‘United States of America’, ‘Kazakhstan’, ‘Uzbekistan’,
‘Papua New Guinea’, ‘Indonesia’, ‘Argentina’, ‘Chile’,
‘Dem. Rep. Congo’, ‘Somalia’, ‘Kenya’, ‘Sudan’, ‘Chad’, ‘Haiti’,
‘Dominican Rep.’, ‘Russia’, ‘Bahamas’, ‘Falkland Is.’, ‘Norway’,
‘Greenland’, ‘Fr. S. Antarctic Lands’, ‘Timor-Leste’,
‘South Africa’, ‘Lesotho’, ‘Mexico’, ‘Uruguay’, ‘Brazil’,
‘Bolivia’, ‘Peru’, ‘Colombia’, ‘Panama’, ‘Costa Rica’, ‘Nicaragua’,
‘Honduras’, ‘El Salvador’, ‘Guatemala’, ‘Belize’, ‘Venezuela’,
‘Guyana’, ‘Suriname’, ‘France’, ‘Ecuador’, ‘Puerto Rico’,
‘Jamaica’, ‘Cuba’, ‘Zimbabwe’, ‘Botswana’, ‘Namibia’, ‘Senegal’,
‘Mali’, ‘Mauritania’, ‘Benin’, ‘Niger’, ‘Nigeria’, ‘Cameroon’,
‘Togo’, ‘Ghana’, “Côte d’Ivoire”, ‘Guinea’, ‘Guinea-Bissau’,
‘Liberia’, ‘Sierra Leone’, ‘Burkina Faso’, ‘Central African Rep.’,
‘Congo’, ‘Gabon’, ‘Eq. Guinea’, ‘Zambia’, ‘Malawi’, ‘Mozambique’,
‘eSwatini’, ‘Angola’, ‘Burundi’, ‘Israel’, ‘Lebanon’, ‘Madagascar’,
‘Palestine’, ‘Gambia’, ‘Tunisia’, ‘Algeria’, ‘Jordan’,
‘United Arab Emirates’, ‘Qatar’, ‘Kuwait’, ‘Iraq’, ‘Oman’,
‘Vanuatu’, ‘Cambodia’, ‘Thailand’, ‘Laos’, ‘Myanmar’, ‘Vietnam’,
‘North Korea’, ‘South Korea’, ‘Mongolia’, ‘India’, ‘Bangladesh’,
‘Bhutan’, ‘Nepal’, ‘Pakistan’, ‘Afghanistan’, ‘Tajikistan’,
‘Kyrgyzstan’, ‘Turkmenistan’, ‘Iran’, ‘Syria’, ‘Armenia’, ‘Sweden’,
‘Belarus’, ‘Ukraine’, ‘Poland’, ‘Austria’, ‘Hungary’, ‘Moldova’,
‘Romania’, ‘Lithuania’, ‘Latvia’, ‘Estonia’, ‘Germany’, ‘Bulgaria’,
‘Greece’, ‘Turkey’, ‘Albania’, ‘Croatia’, ‘Switzerland’,
‘Luxembourg’, ‘Belgium’, ‘Netherlands’, ‘Portugal’, ‘Spain’,
‘Ireland’, ‘New Caledonia’, ‘Solomon Is.’, ‘New Zealand’,
‘Australia’, ‘Sri Lanka’, ‘China’, ‘Taiwan’, ‘Italy’, ‘Denmark’,
‘United Kingdom’, ‘Iceland’, ‘Azerbaijan’, ‘Georgia’,
‘Philippines’, ‘Malaysia’, ‘Brunei’, ‘Slovenia’, ‘Finland’,
‘Slovakia’, ‘Czechia’, ‘Eritrea’, ‘Japan’, ‘Paraguay’, ‘Yemen’,
‘Saudi Arabia’, ‘Antarctica’, ‘N. Cyprus’, ‘Cyprus’, ‘Morocco’,
‘Egypt’, ‘Libya’, ‘Ethiopia’, ‘Djibouti’, ‘Somaliland’, ‘Uganda’,
‘Rwanda’, ‘Bosnia and Herz.’, ‘North Macedonia’, ‘Serbia’,
‘Montenegro’, ‘Kosovo’, ‘Trinidad and Tobago’, ‘S. Sudan’],
dtype=object)
df_country_final.isnull().sum()

A linemap is created based on the Value column in the df_country_final dataframe. The size of the graphic is determined by the figsize parameter. By setting the legend parameter to True, the explanations of the data in the graph are displayed. With the legend_kwds parameter, the properties of the descriptions are determined. These properties are determined by the label parameter and the title of the explanation, and the orientation parameter is determined by the direction of the explanations.
df_country_final.plot('Value',figsize=(20,14),legend=True,
                           legend_kwds={"label":"Gdp By Countrie", "orientation":"horizontal"});

A data frame named gdp is created by re-reading the file. With the pd.to_datetime() function, “Year” column is converted to date format and Date column is created. Then, using the set_index() function, the Date column is set as the index of the data frame. With the loc function, rows with “Country Name” column “United States” are selected and assigned to the gdp data frame.
gdp = pd.read_csv("gdp_csv.csv")

gdp['Date'] = pd.to_datetime(gdp.Year, format='%Y')
gdp.set_index('Date', inplace=True)
gdp = gdp.loc[gdp["Country Name"] == "United States"]
gdp.head()

ARIMA (Autoregressive Integrated Moving Average) and SARIMA (Seasonal Autoregressive Integrated Moving Average) models are common models used for time series analysis. These models allow previous values ​​and errors in a time series to be used to predict current values.
ARIMA models use a combination of the terms autoregression (AR), moving average (MA), and difference (I). The term AR makes a prediction based on previous values of a time series. The term MA represents the error predicted by the model, and the model’s inclusion of this error term allows the model to make more accurate predictions. The difference term takes the difference of values in a given time series and fixes the statistical properties of the series.
The SARIMA models are an extension of the ARIMA models and are used to model data that includes seasonal patterns. A seasonal time series data is a dataset that shows a particular seasonal effect over a year or longer. SARIMA models incorporate seasonal elements in addition to ARIMA models. This makes forecasts more accurate, taking into account the variability and effects of seasonal elements.
Functions are called for the ARIMA and SARIMAX models using the statsmodels library. Then, using the SARIMAX() function, a SARIMAX model is created for the gdp.Value series and the parameters of the model are determined with the order parameter. The model created with the fit() function is trained and the results are assigned to the result_AR variable.
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(gdp.Value, order=(3,1,3))
result_AR = model.fit()
