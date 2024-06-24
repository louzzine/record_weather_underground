#%% 
 
# author: OUZZINE LAMIAA
# date: 17/05/2022
# update : 07/07/2022
# description="This service monitor weather"

#%% 

import datetime as dt
import math
from pickle import FALSE

import numpy as np
import pandas as pd
import pytz
import requests
from dateutil import tz
from pymongo import MongoClient
from suntime import Sun

#%%
# data base

mongo_client = MongoClient (
            host = '10.0.0.16',
            port = 27017,
            username = 'vs_hmi_reader',
            password = 'CBNFjgRQWoetywyvGGDoX8qbW1fqrka82qtca9MCx9XmYIMGNDn9qr14PLW97V2e',
            authSource = 'vs92836'
            )

db = mongo_client["INTRANET"]
mapping_data = db["mapping"]
df_mapping = pd.DataFrame(list(mapping_data.find()))
df_mapping = df_mapping.drop(columns =['_id','device_id','device_sn','modality','variety','organization_id'])
#df_mapping.drop(['_id','device_id','device_sn','modality'],1,inplace=True)
df_mapping.drop_duplicates()


#%%
#VOLUME_PATH = os.environ["VOLUME_PATH"]
#INPUT_FILENAME_DEFAULT = os.environ["INPUT_FILENAME_DEFAULT"]
#%%
# functions

def add_datetime_column(df):
    df['DATETIME'] = [
        dt.datetime.strptime(d + ' ' + h, '%Y-%m-%d %H')
        for d, h in zip(df['DATE'].astype(str), df['HOUR'].astype(str))
    ]
    return df
    
def convert_hour(hour_12_system):
    """Convert time from 12h system to 24h sytem."""
    in_time = dt.datetime.strptime(hour_12_system, "%I:%M %p")
    classic_hour = dt.datetime.strftime(in_time, "%H:%M")
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


def extract_data(start_date, station):
    
    # hour_weather_dict = {}
    cur_date = dt.datetime.strftime(start_date, "%Y-%m-%d")
    url = "https://www.wunderground.com/dashboard/pws/"
    url += "{1}/table/{0}/{0}/daily".format(cur_date, station)
    print(url)
    html = requests.get(url).content
    if 'No data available' in str(html):
        # return(hour_weather_dict)
        return None
    try:
        df_list = pd.read_html(html)
    except:
        return None
    df = df_list[-1].iloc[1:]
    df = df.drop(['Dew Point', 'Wind', 'Gust', 'Pressure', 'Precip. Rate.',
                 'UV', 'Solar'], axis=1)
    df['Date'] = start_date
    # Convert columns in a good format
    df['Time'] = df['Time'].apply(convert_hour)
    df['Temperature'] = df['Temperature'].apply(convert_temperature)
    df['Speed'] = df['Speed'].apply(convert_speed)
    df['Precip. Accum.'] = df['Precip. Accum.'].apply(convert_precipitation)
    df['Humidity'] = df['Humidity'].apply(clean_humidity)
    #
    return(df)

#%%
def record_weather(start_date, end_date, station):
    """
    Extract weather data day by day, then save it in a CSV file.
    """
    daily_weather_df_list = []
    date_list = [start_date + dt.timedelta(days=x) for x in
                 range((end_date - start_date).days + 1)]
    for date in date_list:
        daily_weather_df = extract_data(date, station)
        if daily_weather_df is not None:
            daily_weather_df_list.append(daily_weather_df)
    if len(daily_weather_df_list) == 0:
        return
    result_df = pd.concat(daily_weather_df_list)
    # change column names
    result_df.rename(columns={
        'Date':'DATE',
        'Time':'TIME',
        'Precip. Accum.':'RAIN_FALL',
        'Temperature':'TEMPERATURE',
        'Humidity':'HUMIDITY',
        'Speed':'WIND_SPEED'
    }, inplace=True)
    return result_df


#%%
# %%
# read input file 
#input_filename = INPUT_FILENAME_DEFAULT
#input_filepath = os.path.join(VOLUME_PATH, input_filename)
end_date = dt.datetime.today() - dt.timedelta(days=1)
print(f'End date: {end_date.date()}')

#%%
mongo_client = MongoClient (
    host = '10.0.0.16',
    port = 27017,
    username = 'VS_exposome',
    password = 'vYSO7RXwApcHlWLvUZ21sK3tKBNJISif1Iiw8RA09cvilOERfrLjdOHR8LfGVR98',
    authSource = 'EXPOSOME'
    )
    
#client = MongoClient("mongodb://VS_exposome:vYSO7RXwApcHlWLvUZ21sK3tKBNJISif1Iiw8RA09cvilOERfrLjdOHR8LfGVR98@10.0.0.16:27017")
db = mongo_client["EXPOSOME"]
hourly_weather_data = db["hourly_weather_data"]
daily_weather_data = db["daily_weather_data"]
#f = open(input_filepath)
#json_data = json.load(f)
#%%

for a in df_mapping.index: 
#for a in range (10,11) : 
    station = df_mapping["station_id"][a]
    station = ''.join(station)
    parcel_id = df_mapping["parcel_id"][a]
    parcel_id = ''.join(parcel_id)
    print(f'Station ID: {station}')
    print(f'Parcel ID: {parcel_id}')
    # detection de la dernière date enregitrée
    # Making a Connection with MongoClient

    latest_Date = daily_weather_data.find({'station_id' : station,'parcel_id' : parcel_id})
    
    results = list(latest_Date)
    last = len(results)
    if len(results) == 0:
        start_date = dt.datetime(2021,8,31)
    else :
        start_date = results[last-1]['date']
    start_date = start_date + dt.timedelta(days=1)
    print(f'Start date: {start_date.date()}')
    if df_mapping["bud_burst_date"][a] == '':
        budburst_date = dt.datetime(2022,4,1)
        budburst_date = pd.to_datetime(budburst_date,format='%Y-%m-%d')
    
    df_mapping["bud_burst_date"][a] = pd.to_datetime(df_mapping["bud_burst_date"][a]).date()
    budburst_date = df_mapping["bud_burst_date"][a]
    budburst_date = budburst_date
    print(f'Bud burst date: {budburst_date}')
    lat = df_mapping["latitude"][a]
    lon = df_mapping["longitude"][a]  # longitude of the measurement site (ISAINTSA78) 
    z = df_mapping["elevation"][a]  # z = elevation above sea level

    print(f'Latitude, longiture, elevation: {lat}, {lon}, {z}')
    delta = end_date - start_date
    if(delta.days < 1):
        print("The parcel_id " + parcel_id +" is already up to date in the Database")
    else : 
        data_meteo = record_weather(start_date, end_date, station)
        if data_meteo is None or len(data_meteo) < 1:
            continue
        print("DONE")
     # --------------------- per Hour --------------------------------------
# %%
        data_meteo_h = pd.DataFrame (data_meteo)
        HOUR = data_meteo_h[["TIME"]]
        TEMPERATURE = data_meteo_h [["TEMPERATURE"]]
        RAIN_FALL = data_meteo_h [["RAIN_FALL"]]
        WIND_SPEED = data_meteo_h [["WIND_SPEED"]]
        RELATIVE_HUMIDITY = data_meteo_h [["HUMIDITY"]]
        #data_meteo['DATE'] = pd.to_datetime(data_meteo['DATE'], format='%Y-%m-%d')
        #DATE = data_meteo [["DATE"]]
        data_meteo_h.loc[:,'DATE'] = pd.to_datetime(data_meteo_h.DATE.astype(str)+' '+data_meteo_h.TIME.astype(str))
        data_meteo_h['DATE'] = pd.to_datetime(data_meteo_h['DATE'],utc=FALSE)
        data_meteo_h['DATE'] = data_meteo_h['DATE'] .dt.tz_convert("Europe/Paris")
        del data_meteo_h['TIME']
    # %%
        # Per HOUR
        meteo_per_hour_LAST = data_meteo_h.groupby([data_meteo_h['DATE'].dt.date.rename("DATE"), data_meteo_h['DATE'].dt.hour.rename("HOUR")],as_index=True).agg ( RAIN_FALL_LAST = ('RAIN_FALL','last')
        )
        last_Rain_Hour =  meteo_per_hour_LAST['RAIN_FALL_LAST'].to_numpy()
        meteo_per_hour = data_meteo_h.groupby([data_meteo_h['DATE'].dt.date.rename("DATE"), data_meteo_h['DATE'].dt.hour.rename("HOUR")],as_index=True).agg ( RAIN_FALL = ('RAIN_FALL','last'),TEMPERATURE_AVG = ('TEMPERATURE' ,'mean'),TEMPERATURE_MAX = ('TEMPERATURE' ,'max'),TEMPERATURE_MIN = ('TEMPERATURE' ,'min'),RELATIVE_HUMIDITY = ('HUMIDITY' ,'mean'),WIND_SPEED = ('WIND_SPEED' ,'mean'))
        meteo_per_hour = meteo_per_hour.reset_index()
        meteo_per_hour.columns = ['DATE', 'HOUR', 'RAIN_FALL', 'TEMPERATURE_AVG', 'TEMPERATURE_MAX','TEMPERATURE_MIN','RELATIVE_HUMIDITY','WIND_SPEED']  
        last_Rain_HOUR_corr= meteo_per_hour['RAIN_FALL'].to_numpy()
        Corrected_RAIN_HOUR = np.zeros((len(last_Rain_HOUR_corr),), dtype=float)
        for i in range(1,(len(last_Rain_HOUR_corr)-1)):
            Corrected_RAIN_HOUR[0] = last_Rain_Hour[0]
            if(i % 24 == 0): 
                Corrected_RAIN_HOUR [i] = last_Rain_HOUR_corr[i]
            else :
                Corrected_RAIN_HOUR [i] = last_Rain_HOUR_corr[i] - last_Rain_Hour[i-1]
                if(Corrected_RAIN_HOUR [i] <0 ):
                    Corrected_RAIN_HOUR [i] = -1*Corrected_RAIN_HOUR [i]
        Corrected_RAIN_HOUR[np.isnan(Corrected_RAIN_HOUR)] = 0
        meteo_per_hour['RAIN_FALL'] = Corrected_RAIN_HOUR
        meteo_per_hour = meteo_per_hour.round(3)
        meteo_per_hour = add_datetime_column(meteo_per_hour)
        meteo_per_hour = meteo_per_hour.set_index('DATETIME').resample('H').mean().interpolate('time')
        meteo_per_hour = meteo_per_hour.reset_index(level=[0])
        duration = len(meteo_per_hour ['DATETIME'])
    # %%
        # calcul Mean saturation vapour pressure
        e0 = np.zeros((duration,), dtype=float)
        for i in range(0,duration):
            e0[i] = 0.6108 * math.exp((17.27*meteo_per_hour['TEMPERATURE_AVG'][i]) / (meteo_per_hour['TEMPERATURE_AVG'][i]+237.3))
        
        # calcul Slope of saturation vapour pressure curve (D )
        delta = np.zeros((duration,), dtype=float)
        for i in range(0,duration):
            delta[i] = (4098*e0[i]) / pow((meteo_per_hour['TEMPERATURE_AVG'][i]+237.3),2)
        # Atmospheric Pressure (P)
        P = 101.3 * pow(((293-0.065*z)/293),5.26)
        #Psychrometric constant
        psy = 0.00065*P
        # Delta Term (DT) (auxiliary calculation for Radiation Term)
        # The delta term is used to calculate the “Radiation Term” of the overall ET equation (Eq. 33)
        DT = np.zeros((duration,), dtype=float)
        for i in range(0,duration):
            DT[i] = delta[i] /(delta[i]+psy*(1+0.34*meteo_per_hour['WIND_SPEED'][i]))
        # Psi Term (PT) (auxiliary calculation for Wind Term) The psi term is used to calculate the “Wind Term” of the overall ETo equation [Eq. 34]
        PT = np.zeros((duration,), dtype=float)
        for i in range(0,duration):
            PT[i] = psy / (delta[i]+psy*(1+0.34*meteo_per_hour['WIND_SPEED'][i]))
        #  Temperature Term (TT) (auxiliary calculation for Wind Term)
        TT = np.zeros((duration,), dtype=float)
        for i in range(0,duration):
            TT[i] = (900/(meteo_per_hour['TEMPERATURE_AVG'][i]+273))*meteo_per_hour['WIND_SPEED'][i]
        # Determine daily extraterrestrial radiation. Ra
        g_sc = 0.082  # Solar constant (MJ /m² /min)
        # %%
        # Determine daily extraterrestrial radiation. Ra
        J_start = meteo_per_hour ['DATETIME'][0].timetuple().tm_yday
        
        J_end = meteo_per_hour ['DATETIME'][(duration-1)] .timetuple().tm_yday

        dr = np.zeros((duration,), dtype=float)

        #dr = range(0,len)
        k = 0

        for i in range (0, duration):
                J = meteo_per_hour ['DATETIME'][i].timetuple().tm_yday
                dr[k]= 1+0.033 * math.cos(2 * math.pi /365 *J )
            # print(k)
                k = k+1
        # %%
        dd = np.zeros((duration,), dtype=float)
        k = 0
        
        for i in range (0 ,duration):
                J = meteo_per_hour ['DATETIME'][i].timetuple().tm_yday
                dd[k]= 0.409*math.sin((((2*math.pi)/365)*J)-1.39)
                k = k+1
                
        phi = np.deg2rad(lat)
        ws = np.zeros((duration,), dtype=float)
        Sc = np.zeros((duration,), dtype=float)
        b = np.zeros((duration,), dtype=float)
        w1 = np.zeros((duration,), dtype=float)
        w2 = np.zeros((duration,), dtype=float)
        ra = np.zeros((duration,), dtype=float) #Extraterrestrial radiation for hourly or shorter periods (Ra)
        Lz = 15 # longitude of the centre of the local time zone [degrees west of Greenwich]. 
            # For example, Lz = 75, 90, 105 and 120° for the Eastern, Central, Rocky Mountain and Pacific time zones (United States) and Lz = 0° for Greenwich, 330° for Cairo (Egypt), and 255° for Bangkok (Thailand),
        g_sc = 0.082  # Solar constant (MJ /m² /min)
        for i in range (0,(duration)) :
                J = meteo_per_hour ['DATETIME'][i].timetuple().tm_yday
                b[i] = 2 * math.pi * (J - 81) / 364
                Sc[i] = 0.1645 *math.sin(2*b[i]) - 0.1255*math.cos(b[i]) - 0.025 *math.sin(b[i])
                #ws[i] = math.pi / 12 *((meteo_per_hour['HOUR'][i] + 0.06667 * (Lz -lon) +Sc[i])-12)
                ws[i] = math.acos(-math.tan(phi)*math.tan(dd[i]))
                t1 = 1 #t1 length of the calculation period [hour]: i.e., 1 for hourly period or 0.5 for a 30-minute period.
                w1[i] = ws[i] - (t1*(math.pi/24))  #(equation 29)
                w2[i] = ws[i] + (t1*(math.pi /24)) #(equation 30)
            # Extraterrestrial radiation for daily periods (Ra) eq 28 (FAO)
            # For hourly or shorter periods the solar time angle at the beginning and end of the period should be considered when calculating Ra
                ra[i] = ((12*60)/math.pi) * g_sc *dr[i]*((w2[i]-w1[i])*math.sin(phi)*math.sin(dd[i])+math.cos(phi)*math.cos(dd[i])*(math.sin(w2[i]-math.sin(w1[i]))))
                if ra[i] < 0 :
                    ra[i] = 0

    # %%
    #Determine the net radiation.
        rs = np.zeros((duration,), dtype=float)
        rns = np.zeros((duration,), dtype=float)
        rnl = np.zeros((duration,), dtype=float)
        rn = np.zeros((duration,), dtype=float)
        for i in range (0,(duration)) :
            if ra[i] == 0 :
                rs[i] = 0
            else :
            #rs solar radiation  eq 50
                rs[i] =  0.16*np.sqrt(meteo_per_hour['TEMPERATURE_MAX'][i]-meteo_per_hour['TEMPERATURE_MIN'][i])*ra[i]
        #Compute net shortwave radiation (rns)
            rns[i] = (1-0.23)*rs[i]
        #Compute net longwave radiation (rnl) (Eq. 39)
            rnl[i] = (0.34-0.14*np.sqrt(e0[i]))*(1.35*rs[i]/(0.75*ra[i])-0.35)
        #rn is the difference between the incoming rns and the outgoing rnl
            rn[i] = rns[i]-rnl[i]

    # %%
    #Rng To express the net radiation (Rn) in equivalent of evaporation (mm) (Rng);
        rng = np.zeros((duration,), dtype=float)
        for i in range (0,duration):
                rng[i] = 0.408*rn[i]

    # %%
    # calcul VPD par heur
        gam = 0.067 # Link to altitude (Table 2.2)
        e_max = np.zeros((duration,), dtype=float)
        e_min = np.zeros((duration,), dtype=float)
        es = np.zeros((duration,), dtype=float)
        ea = np.zeros((duration,), dtype=float)
        hourly_VPD = np.zeros((duration,), dtype=float)
        hourly_GDD = np.zeros((duration,), dtype=float)
        for i in range (0,duration) :
            e_min[i] = 0.6108 * math.exp((17.27*meteo_per_hour['TEMPERATURE_MIN'][i])/(meteo_per_hour['TEMPERATURE_MIN'][i]+237.3))
            e_max[i] = 0.6108 * math.exp((17.27*meteo_per_hour['TEMPERATURE_MAX'][i])/(meteo_per_hour['TEMPERATURE_MAX'][i]+237.3))
            es[i] = (e_min[i] + e_max[i])/2
            ea[i] = (meteo_per_hour['RELATIVE_HUMIDITY'][i]/100)*((e_max[i]+e_min[i])/2)
            hourly_VPD[i] = es[i] - ea[i]
    # %% ETP par heure
    #référence : Differentiation of computed sum of hourly and daily reference evapotranspiration in a semi-arid climate (Bahram Bakhtiari,2017)
        hourly_ETP = np.zeros((duration,), dtype=float)
        sun = Sun(lat, lon)
        for i in range (0,duration) :
            sun_rise = sun.get_local_sunrise_time(meteo_per_hour['DATETIME'][i])
            sun_rise = sun_rise.replace(tzinfo=None) 
            sun_set = sun.get_local_sunset_time(meteo_per_hour['DATETIME'][i])
            sun_set = sun_set.replace(tzinfo=None) 
        #For hourly or shorter periods
        # Ghr :For hourly (or shorter) calculations, G beneath a dense cover of grass does not correlate well with air temperature. 
    
            if sun_rise < meteo_per_hour['DATETIME'][i].to_pydatetime() < sun_set:
            # Hourly G can be approximated during daylight periods as
                Ghr = 0.1 * rn[i]
            else :
            # during nighttime periods as:
                Ghr = 0.5 * rn[i]
            hourly_ETP[i] = ((0.408*delta[i]*(rn[i]-Ghr))+gam*(37/(meteo_per_hour['TEMPERATURE_AVG'][i]+273)*meteo_per_hour['WIND_SPEED'][i]*(es[i]-ea[i])))/(delta[i]+gam*(1+0.4*meteo_per_hour['WIND_SPEED'][i]))
    
    # %%
    # Calcul déficit hydrique
        u = dt.timedelta(days=19)
        hourly_deficit = np.zeros((duration,), dtype=float)
        hourly_bilan_hydrique = np.zeros((duration,), dtype=float)
        date_start_deficit = budburst_date + u
        for i in range (0,(duration-1)) :
            if meteo_per_hour['DATETIME'][i] < date_start_deficit :
                hourly_deficit [i] = np.NaN
                hourly_bilan_hydrique [i] = np.NaN
            if meteo_per_hour['DATETIME'][i] == date_start_deficit :
                hourly_deficit[i] = 0
            else :
                hourly_bilan_hydrique [i+1] =  meteo_per_hour ['RAIN_FALL'][i+1] - hourly_ETP[i+1] 
                if hourly_deficit [i] == np.NaN:
                    latest_hourly_deficit = hourly_weather_data.find({'station_id' : station,'parcel_id' : parcel_id})
                    results_deficit = list(latest_hourly_deficit)
                    last_deficit = len(results_deficit )
                    hourly_deficit[i]  = results_deficit[last_deficit-1]['WATER_DEFICIT']
                hourly_deficit [i+1] = hourly_bilan_hydrique [i+1] + hourly_deficit [i]
        
        # %%
        # Save in dataframe   
        del meteo_per_hour['HOUR']
        tz = pytz.timezone('Europe/Paris')
        for i in range (0,duration):
            meteo_per_hour['DATETIME'][i] = meteo_per_hour['DATETIME'][i].replace(tzinfo=tz) 
        
        weather_augmentated_hourly = meteo_per_hour
        weather_augmentated_hourly = weather_augmentated_hourly.set_axis(['datetime', 'rain_fall', 'temperature_avg', 'temperature_max', 'temperature_min','relative_humidity','wind_speed'], axis=1, inplace=False)
        #weather_augmentated_hourly ['datetime'] = meteo_per_hour['DATETIME']
        weather_augmentated_hourly ['station_id'] = station
        weather_augmentated_hourly ['parcel_id'] = parcel_id 
        weather_augmentated_hourly ['hourly_etp'] = hourly_ETP
        weather_augmentated_hourly ['hourly_vpd'] = hourly_VPD
        #weather_augmentated_hourly ['hourly_GDD'] = hourly_GDD
        weather_augmentated_hourly ['water_deficit'] = hourly_deficit
        weather_augmentated_hourly ['longitude'] = lon
        weather_augmentated_hourly ['latitude'] = lat
        weather_augmentated_hourly ['timezone'] = 'Europe/Paris'
        weather_augmentated_hourly ['weather_source'] = 'weather underground'

        # convert datafram to dict
        weather_augmentated_hourly_dict = weather_augmentated_hourly.to_dict("records")
        # %%

        hourly_weather_data.insert_many(weather_augmentated_hourly_dict)

        print("Hourly save : DONE")
        # --------------------------- per day ---------------------------------
        # %%
        print("per day start")
        data_meteo_d = pd.DataFrame (data_meteo)
        HOUR = data_meteo_d[["TIME"]]
        TEMPERATURE = data_meteo_d [["TEMPERATURE"]]
        RAIN_FALL = data_meteo_d [["RAIN_FALL"]]
        WIND_SPEED = data_meteo_d [["WIND_SPEED"]]
        RELATIVE_HUMIDITY = data_meteo_d [["HUMIDITY"]]
        data_meteo_d['DATE'] = pd.to_datetime(data_meteo_d['DATE'], format='%Y-%m-%d')
        #DATE = data_meteo_d ["DATE"].dt.date
        date_time = pd.concat([data_meteo_d['DATE'], HOUR], join = 'outer', axis = 1)
        
        # %%
        # Per DAY
        #meteo_per_day = data_meteo_d.groupby('DATE',as_index=False).aggregate({'RAIN_FALL':['sum'],'TEMPERATURE':['mean','max','min'],'HUMIDITY':'mean','WIND_SPEED':'mean'})
        meteo_per_day = data_meteo_d.groupby(data_meteo_d ["DATE"].dt.date).agg ( RAIN_FALL = ('RAIN_FALL','last'),TEMPERATURE_AVG = ('TEMPERATURE' ,'mean'),TEMPERATURE_MAX = ('TEMPERATURE' ,'max'),TEMPERATURE_MIN = ('TEMPERATURE' ,'min'),RELATIVE_HUMIDITY = ('HUMIDITY' ,'mean'),WIND_SPEED = ('WIND_SPEED' ,'mean'))
        meteo_per_day = meteo_per_day.reset_index(level=[0])


        meteo_per_day.columns = ['DATE',  'RAIN_FALL', 'TEMPERATURE_AVG', 'TEMPERATURE_MAX','TEMPERATURE_MIN','RELATIVE_HUMIDITY','WIND_SPEED']
        meteo_per_day['DATE'] = pd.to_datetime(meteo_per_day['DATE'], format='%Y-%m-%d')
        #meteo_per_day = meteo_per_day.set_index('DATE').resample('D').mean().interpolate('time')
        meteo_per_day = meteo_per_day.set_index('DATE').resample('D').mean()

        meteo_per_day = meteo_per_day.reset_index(level=[0])
        duration = len(meteo_per_day ['DATE'])

        # %%
        # calcul Mean saturation vapour pressure
        e0 = np.zeros((duration,), dtype=float)
        for i in range(0,duration):
            e0[i] = 0.6108 * math.exp((17.27*meteo_per_day['TEMPERATURE_AVG'][i]) / (meteo_per_day['TEMPERATURE_AVG'][i]+237.3))
        # calcul Slope of saturation vapour pressure curve (D )
        delta = np.zeros((duration,), dtype=float)
        for i in range(0,duration):
            delta[i] = (4098*e0[i]) / pow((meteo_per_day['TEMPERATURE_AVG'][i]+237.3),2)
        # Atmospheric Pressure (P)

        P = 101.3 * pow(((293-0.065*z)/293),5.26)
        #Psychrometric constant
        psy = 0.00065*P
        # Delta Term (DT) (auxiliary calculation for Radiation Term)
        # The delta term is used to calculate the “Radiation Term” of the overall ET equation (Eq. 33)
        DT = np.zeros((duration,), dtype=float)
        for i in range(0,duration):
            DT[i]= delta[i] /(delta[i]+psy*(1+0.34*meteo_per_day['WIND_SPEED'][i]))
        # Psi Term (PT) (auxiliary calculation for Wind Term) The psi term is used to calculate the “Wind Term” of the overall ETo equation [Eq. 34]
        PT = np.zeros((duration,), dtype=float)
        for i in range(0,duration):
            PT[i] = psy / (delta[i]+psy*(1+0.34*meteo_per_day['WIND_SPEED'][i]))
        #  Temperature Term (TT) (auxiliary calculation for Wind Term)
        TT = np.zeros((duration,), dtype=float)
        for i in range(0,duration):
            TT[i] = (900/(meteo_per_day['TEMPERATURE_AVG'][i]+273))*meteo_per_day['WIND_SPEED'][i]
        # Determine daily extraterrestrial radiation. Ra
        g_sc = 0.082  # Solar constant (MJ /m² /min)
        # %%
        # Determine daily extraterrestrial radiation. Ra
        J_start = meteo_per_day ['DATE'][0].timetuple().tm_yday
        
        J_end = meteo_per_day ['DATE'][(duration-1)] .timetuple().tm_yday
        #if J_end < J_start:
        #   J_end = J_end + 365

        dr = np.zeros((duration,), dtype=float)
        #dr = range(0,len)
        k = 0
        #for i in range (J_start ,(J_end)+1) :
        for i in range (0, duration):
            J = meteo_per_day ['DATE'][i].timetuple().tm_yday
            dr[k]= 1+0.033 * math.cos(2 * math.pi /365 *J )
        # print(k)
            k = k+1
        
        # %%
        dd = np.zeros((duration,), dtype=float)
        k = 0
        for i in range (0, duration):
            J = meteo_per_day ['DATE'][i].timetuple().tm_yday
            dd[k]= 0.409*math.sin((((2*math.pi)/365)*J)-1.39)
            k = k+1
        # %%
        phi = np.deg2rad(lat)
        ws = np.zeros((duration,), dtype=float)
        for i in range (0,(duration)) :
            ws[i] = math.acos(-math.tan(phi)*math.tan(dd[i]))

        # #Extraterrestrial radiation for daily periods (Ra) eq 21 (FAO)
        g_sc = 0.082  # Solar constant (MJ /m² /min)
        ra = np.zeros((duration,), dtype=float)
        for i in range (0,duration) :
            ra[i] = ((24*60)/math.pi)*g_sc*dr[i]*(ws[i]*math.sin(phi)*math.sin(dd[i])+math.cos(phi)*math.cos(dd[i])*math.sin(ws[i]))
            if ra[i] < 0 :
                ra[i] = 0

        # %%
        #Determine the net radiation.
        rs = np.zeros((duration,), dtype=float)
        rns = np.zeros((duration,), dtype=float)
        rnl = np.zeros((duration,), dtype=float)
        rn = np.zeros((duration,), dtype=float)
        for i in range (0,(duration)) :
            if ra[i] == 0 :
                rs[i] = 0
            else :
            #rs solar radiation  eq 50
                rs[i] =  0.19*np.sqrt(meteo_per_day['TEMPERATURE_MAX'][i]-meteo_per_day['TEMPERATURE_MIN'][i])*ra[i]
            #Compute net shortwave radiation (rns)
            rns[i] = (1-0.23)*rs[i]
            #Compute net longwave radiation (rnl) (Eq. 39)
            rnl[i] = (0.34-0.14*np.sqrt(e0[i]))*(1.35*rs[i]/(0.75*ra[i])-0.35)
            #rn is the difference between the incoming rns and the outgoing rnl
            rn[i] = rns[i]-rnl[i]

        # %%
        #Rng To express the net radiation (Rn) in equivalent of evaporation (mm) (Rng);
        rng = np.zeros((duration,), dtype=float)
        for i in range (0,duration):
            rng[i] = 0.408*rn[i]

        # %%
        # calcul ETP par jour 
        gam = 0.067 # Link to altitude (Table 2.2)
        e_max = np.zeros((duration,), dtype=float)
        e_min = np.zeros((duration,), dtype=float)
        es = np.zeros((duration,), dtype=float)
        ea = np.zeros((duration,), dtype=float)
        daily_ETP = np.zeros((duration,), dtype=float)
        daily_VPD = np.zeros((duration,), dtype=float)
        daily_GDD = np.zeros((duration,), dtype=float)
        for i in range (0,duration) :
            e_min[i] = 0.6108 * math.exp((17.27*meteo_per_day['TEMPERATURE_MIN'][i])/(meteo_per_day['TEMPERATURE_MIN'][i]+237.3))
            e_max[i] = 0.6108 * math.exp ((17.27*meteo_per_day['TEMPERATURE_MAX'][i])/(meteo_per_day['TEMPERATURE_MAX'][i]+237.3))
            es[i] = (e_min[i] +e_max[i])/2
            ea[i] = (meteo_per_day['RELATIVE_HUMIDITY'][i]/100)*((e_max[i]+e_min[i])/2)
            daily_VPD[i] = es[i] - ea[i]
            daily_ETP[i] = ((0.408*delta[i]*rn[i])+gam*(900/(meteo_per_day['TEMPERATURE_AVG'][i]+273)*meteo_per_day['WIND_SPEED'][i]*(es[i]-ea[i])))/(delta[i]+gam*(1+0.4*meteo_per_day['WIND_SPEED'][i]))
            #daily_GDD[i] = sum(meteo_per_day['TEMPERATURE_AVG'][1:i-10])

        # %%
        # Calcul déficit hydrique
        u = dt.timedelta(days=19)
        daily_deficit = np.zeros((duration,), dtype=float)
        daily_bilan_hydrique = np.zeros((duration,), dtype=float)
        date_start_deficit = budburst_date + u
        for i in range (0,(duration-1)) :
            if meteo_per_day['DATE'][i] < date_start_deficit :
                daily_deficit [i] = np.NaN
                daily_bilan_hydrique [i] = np.NaN
            if meteo_per_day['DATE'][i] == date_start_deficit :
                daily_deficit[i] = 0
                daily_bilan_hydrique [i+1] = meteo_per_day ['RAIN_FALL'][i+1] - daily_ETP[i+1] 
                daily_deficit [i+1] = daily_bilan_hydrique [i+1] + daily_deficit [i]
            else :
                daily_bilan_hydrique [i+1] = daily_ETP[i+1] - meteo_per_day ['RAIN_FALL'][i+1]
                daily_deficit [i+1] = daily_bilan_hydrique [i+1] + daily_deficit [i]




        # %%
        # Save in dataframe  
        weather_augmentated_daily = meteo_per_day 
        weather_augmentated_daily = weather_augmentated_daily.set_axis(['date', 'rain_fall', 'temperature_avg', 'temperature_max', 'temperature_min','relative_humidity','wind_speed'], axis=1, inplace=False)
        #weather_augmentated_daily ['date'] = meteo_per_day ['DATE']
        weather_augmentated_daily ['station_id'] = station
        weather_augmentated_daily ['parcel_id'] = parcel_id 
        weather_augmentated_daily ['daily_etp'] = daily_ETP
        weather_augmentated_daily ['daily_vpd'] = daily_VPD
        #weather_augmentated_daily ['DAILY_GDD'] = daily_GDD
        weather_augmentated_daily ['water_deficit'] = daily_deficit
        weather_augmentated_daily ['longitude'] = lon
        weather_augmentated_daily ['latitude'] = lat
        weather_augmentated_daily ['timezone'] = 'Europe/Paris'
        weather_augmentated_daily ['weather_source'] = 'weather underground'

        # %%
        # convert datafram to dict
        weather_augmentated_daily_dict = weather_augmentated_daily.to_dict("records")
        # %%
        # Making a Connection with MongoClient

        mongo_client = MongoClient (
            host = '10.0.0.16',
            port = 27017,
            username = 'VS_exposome',
            password = 'vYSO7RXwApcHlWLvUZ21sK3tKBNJISif1Iiw8RA09cvilOERfrLjdOHR8LfGVR98',
            authSource = 'EXPOSOME'
        )
            
        #client = MongoClient("mongodb://VS_exposome:vYSO7RXwApcHlWLvUZ21sK3tKBNJISif1Iiw8RA09cvilOERfrLjdOHR8LfGVR98@10.0.0.16:27017")
        db = mongo_client["EXPOSOME"]
        daily_weather_data= db["daily_weather_data"]
        daily_weather_data.insert_many(weather_augmentated_daily_dict)

        print("Daily save : DONE")
