#!/usr/bin/env python

"""
Get Weather Underground data between two dates.
Ref: code from Gabriel Guillocheau
"""

import requests
import pandas as pd
from datetime import datetime
from datetime import timedelta
from datetime import date
import numpy as np
from collections import ChainMap
import argparse

__author__ = "Paul Bui Quang"
__copyright__ = "Copyright 2021, Vegetal Signals"

#parser = argparse.ArgumentParser(description="Get raw Weather Underground\
                                 #data between two dates")
#parser.add_argument('-d', '--startdate', nargs='+', type=int,
                    #help='Give the START date : YEAR MONTH DAY')
#parser.add_argument('-D', '--enddate', nargs='+', type=int,
                    #help='Give the START date : YEAR MONTH DAY')
#parser.add_argument('-o', '--output', action='store',
                    #help='Give the output file name')
#parser.add_argument('-S', '--station', action='store', default='INMES40',
                    #help='Give the output file name')


#args = parser.parse_args()
#start_date = datetime(*args.startdate)
#end_date = datetime(*args.enddate)
#out_file = args.output
#station = args.station

start_date = datetime (2021,5,1)
end_date = datetime.today() - timedelta(days=1)
out_file = "meteo.csv"
station = "ITOILE2"


def convert_hour(hour_12_system):
    """Convert time from 12h system to 24h sytem."""
    in_time = datetime.strptime(hour_12_system, "%I:%M %p")
    classic_hour = datetime.strftime(in_time, "%H:%M")
    return(classic_hour)


def convert_temperature(T):
    """Convert temperature in °F to °C (fuck imperial system)."""
    if T == "--":  # handle missing value
        return(np.nan)
    T = float(T.split('\xa0')[0])
    T_c = round((T - 32) * 5 / 9, 2)
    return(T_c)


def convert_speed(miph):
    """Convert speed in miles per hour (miph) to m/s."""
    if miph == "--":  # handle missing value
        return(np.nan)
    else:
        miph = float(miph.split('\xa0')[0])
        mps = round(miph / 2.237, 2)
        return(mps)


def convert_precipitation(inch):
    """Convert rain fall in inch to mm."""
    if inch == "--":  # handle missing value
        return(np.nan)
    inch = float(inch.split('\xa0')[0])
    mm = round(inch * 25.4, 2)
    return(mm)


def clean_humidity(H):
    """Remove unit in humidity."""
    if H == "--":  # handle missing value
        return(np.nan)
    H = int(H.split('\xa0')[0])
    return(H)


def extract_data(current_date, station=station):
    """
    Remove unit in humidity.

    Args:
        current_date (datetime): Only
    Returns:
        int(humidity)
    """
    hour_weather_dict = {}
    cur_date = datetime.strftime(current_date, "%Y-%m-%d")
    url = "https://www.wunderground.com/dashboard/pws/"
    url += "{1}/table/{0}/{0}/daily".format(cur_date, station)
    print(url)
    html = requests.get(url).content
    if 'No data available' in str(html):
        return(hour_weather_dict)
    df_list = pd.read_html(html)
    df = df_list[-1].iloc[1:]
    df = df.drop(['Dew Point', 'Wind', 'Gust', 'Pressure', 'Precip. Accum.',
                 'UV', 'Solar'], axis=1)
    df['Date'] = current_date
    # Convert columns in a good format
    df['Time'] = df['Time'].apply(convert_hour)
    df['Temperature'] = df['Temperature'].apply(convert_temperature)
    df['Speed'] = df['Speed'].apply(convert_speed)
    df['Precip. Rate.'] = df['Precip. Rate.'].apply(convert_precipitation)
    df['Humidity'] = df['Humidity'].apply(clean_humidity)
    #
    return(df)


def record_weather(start_date, end_date, out_file, station):
    """
    Extract weather data day by day, then save it in a CSV file.
    """
    daily_weather_df_list = []
    date_list = [start_date + timedelta(days=x) for x in
                 range((end_date - start_date).days + 1)]
    for date in date_list:
        daily_weather_df = extract_data(date)
        daily_weather_df_list.append(daily_weather_df)
    result_df = pd.concat(daily_weather_df_list)
    # change column names
    result_df.rename(columns={
        'Date':'DATE',
        'Time':'TIME',
        'Precip. Rate.':'RAIN_FALL',
        'Temperature':'TEMPERATURE',
        'Humidity':'HUMIDITY',
        'Speed':'WIND_SPEED'
    }, inplace=True)
    result_df.to_csv(out_file)
    print(f'File {out_file} saved!')


if __name__ == "__main__":
    record_weather(start_date, end_date, out_file, station)