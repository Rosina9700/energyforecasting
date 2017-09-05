# Script setup
import pandas as pd
import xarray as xr
import numpy as np
import requests
import logging
import yaml
import json
import os
import hashlib
import sqlalchemy

from datetime import datetime
from calendar import monthrange
from opendap_download.multi_processing_download import DownloadManager
import math
from functools import partial
import re
import getpass
from datetime import datetime, timedelta
import dateutil.parser

# Set up a log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('notebook')

def translate_lat_to_geos5_native(latitude):
    """
    The source for this formula is in the MERRA2
    Variable Details - File specifications for GEOS pdf file.
    The Grid in the documentation has points from 1 to 361 and 1 to 576.
    The MERRA-2 Portal uses 0 to 360 and 0 to 575.
    latitude: float Needs +/- instead of N/S
    """
    return ((latitude + 90) / 0.5)

def translate_lon_to_geos5_native(longitude):
    """See function above"""
    return ((longitude + 180) / 0.625)

def find_closest_coordinate(calc_coord, coord_array):
    """
    Since the resolution of the grid is 0.5 x 0.625, the 'real world'
    coordinates will not be matched 100% correctly. This function matches
    the coordinates as close as possible.
    """
    # np.argmin() finds the smallest value in an array and returns its
    # index. np.abs() returns the absolute value of each item of an array.
    # To summarize, the function finds the difference closest to 0 and returns
    # its index.
    index = np.abs(coord_array-calc_coord).argmin()
    return coord_array[index]

def translate_year_to_file_number(year):
    """
    The file names consist of a number and a meta data string.
    The number changes over the years. 1980 until 1991 it is 100,
    1992 until 2000 it is 200, 2001 until 2010 it is  300
    and from 2011 until now it is 400.
    """
    file_number = ''

    if year >= 1980 and year < 1992:
        file_number = '100'
    elif year >= 1992 and year < 2001:
        file_number = '200'
    elif year >= 2001 and year < 2011:
        file_number = '300'
    elif year >= 2011:
        file_number = '400'
    else:
        raise Exception('The specified year is out of range.')

    return file_number

def generate_url_params(parameter, time_para, lat_para, lon_para):
    """Creates a string containing all the parameters in query form"""
    parameter = map(lambda x: x + time_para, parameter)
    parameter = map(lambda x: x + lat_para, parameter)
    parameter = map(lambda x: x + lon_para, parameter)

    return ','.join(parameter)

def generate_download_links(download_years, base_url, dataset_name, url_params):
    """
    Generates the links for the download.
    download_years: The years you want to download as array.
    dataset_name: The name of the data set. For example tavg1_2d_slv_Nx
    """
    urls = []
    for y in download_years:
    # build the file_number
        y_str = str(y)
        file_num = translate_year_to_file_number(download_year)
        for m in range(1,13):
            # build the month string: for the month 1 - 9 it starts with a leading 0.
            # zfill solves that problem
            m_str = str(m).zfill(2)
            # monthrange returns the first weekday and the number of days in a
            # month. Also works for leap years.
            _, nr_of_days = monthrange(y, m)
            for d in range(1,nr_of_days+1):
                d_str = str(d).zfill(2)
                # Create the file name string
                file_name = 'MERRA2_{num}.{name}.{y}{m}{d}.nc4'.format(
                    num=file_num, name=dataset_name,
                    y=y_str, m=m_str, d=d_str)
                # Create the query
                query = '{base}{y}/{m}/{name}.nc4?{params}'.format(
                    base=base_url, y=y_str, m=m_str,
                    name=file_name, params=url_params)
                urls.append(query)
    return urls



def extract_date(data_set):
    """
    Extracts the date from the filename before merging the datasets.
    """
    try:
        # The attribute name changed during the development of this script
        # from HDF5_Global.Filename to Filename.
        if 'HDF5_GLOBAL.Filename' in data_set.attrs:
            f_name = data_set.attrs['HDF5_GLOBAL.Filename']
        elif 'Filename' in data_set.attrs:
            f_name = data_set.attrs['Filename']
        else:
            raise AttributeError('The attribute name has changed again!')

        # find a match between "." and ".nc4" that does not have "." .
        exp = r'(?<=\.)[^\.]*(?=\.nc4)'
        res = re.search(exp, f_name).group(0)
        # Extract the date.
        y, m, d = res[0:4], res[4:6], res[6:8]
        date_str = ('%s-%s-%s' % (y, m, d))
        data_set = data_set.assign(date=date_str)
        return data_set

    except KeyError:
        # The last dataset is the one all the other sets will be merged into.
        # Therefore, no date can be extracted.
        return data_set

def calculate_datetime(d_frame):
    """
    Calculates the accumulated hour based on the date.
    """
    cur_date = datetime.strptime(d_frame['date'], '%Y-%m-%d')
    hour = int(d_frame['time'])
    delta = abs(cur_date - start_date).days
    date_time_value = (delta * 24) + (hour)
    return date_time_value

def converting_timeformat_to_ISO8601(row):
    """Generates datetime according to ISO 8601 (UTC)"""
    date = dateutil.parser.parse(row['date'])
    hour = int(row['time'])
    # timedelta from the datetime module enables the programmer
    # to add time to a date.
    date_time = date + timedelta(hours = hour)
    return str(date_time.isoformat()) + 'Z'  # MERRA2 datasets have UTC time zone.

def calculate_windspeed(d_frame, idx_u, idx_v):
    """
    Calculates the windspeed. The returned unit is m/s
    """
    um = float(d_frame[idx_u])
    vm = float(d_frame[idx_v])
    speed = math.sqrt((um ** 2) + (vm ** 2))
    return round(speed, 2)

def get_sha_hash(path, blocksize=65536):
    sha_hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        buffer = f.read(blocksize)
        while len(buffer) > 0:
            sha_hasher.update(buffer)
            buffer = f.read(blocksize)
        return sha_hasher.hexdigest()


if __name__=='__main__':
    # User input of timespan
    download_year = 2016
    # Create the start date 2016-01-01
    download_start_date = str(download_year) + '-01-01'

    # User input of coordinates
    # ------
    # Example: Germany (lat/lon)
    # Northeastern point: 55.05917°N, 15.04361°E
    # Southwestern point: 47.27083°N, 5.86694°E

    # It is important to make the southwestern coordinate lat_1 and lon_1 since
    # the MERRA-2 portal requires it!
    # Southwestern coordinate
    # lat_1, lon_1 = 47.27083, 5.86694 Germany
    # Northeastern coordinate
    # lat_2, lon_2 = 55.05917, 15.04361 Germany

    # Southwestern coordinate
    lat_1, lon_1 = 47.27083, 5.86694
    # Northeastern coordinate
    lat_2, lon_2 = 55.05917, 15.04361

    # The arrays contain the coordinates of the grid used by the API.
    # The values are from 0 to 360 and 0 to 575
    lat_coords = np.arange(0, 361, dtype=int)
    lon_coords = np.arange(0, 576, dtype=int)

    # Translate the coordinates that define your area to grid coordinates.
    lat_coord_1 = translate_lat_to_geos5_native(lat_1)
    lon_coord_1 = translate_lon_to_geos5_native(lon_1)
    lat_coord_2 = translate_lat_to_geos5_native(lat_2)
    lon_coord_2 = translate_lon_to_geos5_native(lon_2)


    # Find the closest coordinate in the grid.
    lat_co_1_closest = find_closest_coordinate(lat_coord_1, lat_coords)
    lon_co_1_closest = find_closest_coordinate(lon_coord_1, lon_coords)
    lat_co_2_closest = find_closest_coordinate(lat_coord_2, lat_coords)
    lon_co_2_closest = find_closest_coordinate(lon_coord_2, lon_coords)

    # Check the precision of the grid coordinates. These coordinates are not lat/lon.
    # They are coordinates on the MERRA-2 grid.
    log.info('Calculated coordinates for point 1: ' + str((lat_coord_1, lon_coord_1)))
    log.info('Closest coordinates for point 1: ' + str((lat_co_1_closest, lon_co_1_closest)))
    log.info('Calculated coordinates for point 2: ' + str((lat_coord_2, lon_coord_2)))
    log.info('Closest coordinates for point 2: ' + str((lat_co_2_closest, lon_co_2_closest)))



    requested_params = ['U2M', 'U10M', 'U50M', 'V2M', 'V10M', 'V50M', 'DISPH', 'SWGDN', 'SWTDN','T2M', 'RHOA', 'PS']
    requested_time = '[0:1:23]'
    # Creates a string that looks like [start:1:end]. start and end are the lat or
    # lon coordinates define your area.
    requested_lat = '[{lat_1}:1:{lat_2}]'.format(
                    lat_1=lat_co_1_closest, lat_2=lat_co_2_closest)
    requested_lon = '[{lon_1}:1:{lon_2}]'.format(
                    lon_1=lon_co_1_closest, lon_2=lon_co_2_closest)

    parameter = generate_url_params(requested_params, requested_time,
                                    requested_lat, requested_lon)

    BASE_URL = 'https://goldsmr4.sci.gsfc.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/'
    generated_URL = generate_download_links([download_year], BASE_URL,
                                            'tavg1_2d_slv_Nx',
                                            parameter)

    # See what a query to the MERRA-2 portal looks like.
    log.info(generated_URL[0])

    # Download data (one file per day and dataset) with links to local directory.
    # Username and password for MERRA-2 (NASA earthdata portal)
    username = input('Username: ')
    password = getpass.getpass('Password:')

    # The DownloadManager is able to download files. If you have a fast internet
    # connection, setting this to 2 is enough. If you have slow wifi, try setting
    # it to 4 or 5. If you download too fast, the data portal might ban you for a
    # day.
    NUMBER_OF_CONNECTIONS = 5

    # The DownloadManager class is defined in the opendap_download module.
    download_manager = DownloadManager()
    download_manager.set_username_and_password(username, password)
    download_manager.download_path = 'download_wind'
    download_manager.download_urls = generated_URL

    # If you want to see the download progress, check the download folder you
    # specified
    %time download_manager.start_download(NUMBER_OF_CONNECTIONS)

    # Download time approx. 20+ min.

    # Roughness data is in a different data set. The parameter is called Z0M.
    roughness_para = generate_url_params(['Z0M'], requested_time,
                                         requested_lat, requested_lon)
    ROUGHNESS_BASE_URL = 'https://goldsmr4.sci.gsfc.nasa.gov/opendap/MERRA2/M2T1NXFLX.5.12.4/'
    roughness_links = generate_download_links([download_year], ROUGHNESS_BASE_URL,
                                              'tavg1_2d_flx_Nx', roughness_para)

    download_manager.download_path = 'download_roughness'
    download_manager.download_urls = roughness_links

    # If you want to see the download progress, check the download folder you
    # specified.
    %time download_manager.start_download(NUMBER_OF_CONNECTIONS)

    # Download time approx. 12+ min.

    # Parameters SWGDN and SWTDN
    radiation_para = generate_url_params(['SWGDN', 'SWTDN'], requested_time,
                                         requested_lat, requested_lon)
    RADIATION_BASE_URL = 'https://goldsmr4.sci.gsfc.nasa.gov/opendap/MERRA2/M2T1NXRAD.5.12.4/'
    radiation_links = generate_download_links([download_year], RADIATION_BASE_URL,
                                             'tavg1_2d_rad_Nx', radiation_para)

    download_manager.download_path = 'download_radiation'
    download_manager.download_urls = radiation_links

    %time download_manager.start_download(NUMBER_OF_CONNECTIONS)

    # Download time approx. 8+ min.

    # Parameter T2M (i.e. the temperature 2 meters above displacement height)
    temperature_para = generate_url_params(['T2M'], requested_time,
                                         requested_lat, requested_lon)
    TEMPERATURE_BASE_URL = 'http://goldsmr4.sci.gsfc.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/'
    temperature_links = generate_download_links([download_year], TEMPERATURE_BASE_URL,
                                             'tavg1_2d_slv_Nx', temperature_para)

    download_manager.download_path = 'download_temperature'
    download_manager.download_urls = temperature_links

    %time download_manager.start_download(NUMBER_OF_CONNECTIONS)

    # Download time approx. 13+ min.

    # Parameter RHOA
    density_para = generate_url_params(['RHOA'], requested_time,
                                         requested_lat, requested_lon)
    DENSITY_BASE_URL = 'http://goldsmr4.sci.gsfc.nasa.gov/opendap/MERRA2/M2T1NXFLX.5.12.4/'
    density_links = generate_download_links([download_year], DENSITY_BASE_URL,
                                             'tavg1_2d_flx_Nx', density_para)

    download_manager.download_path = 'download_density'
    download_manager.download_urls = density_links

    %time download_manager.start_download(NUMBER_OF_CONNECTIONS)

    # Download time approx. 13+ min.



    # Parameters PS
    pressure_para = generate_url_params(['PS'], requested_time,
                                         requested_lat, requested_lon)
    PRESSURE_BASE_URL = 'http://goldsmr4.sci.gsfc.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/'
    pressure_links = generate_download_links([download_year], PRESSURE_BASE_URL,
                                             'tavg1_2d_slv_Nx', pressure_para)

    download_manager.download_path = 'download_pressure'
    download_manager.download_urls = pressure_links

    %time download_manager.start_download(NUMBER_OF_CONNECTIONS)

    # Download time approx. 15+ min.

    # The dimensions map the MERRA2 grid coordinates to lat/lon. The coordinates
    # to request are 0:360 wheare as the other coordinates are 1:361
    requested_lat_dim = '[{lat_1}:1:{lat_2}]'.format(
                        lat_1=lat_co_1_closest, lat_2=lat_co_2_closest)
    requested_lon_dim = '[{lon_1}:1:{lon_2}]'.format(
                        lon_1=lon_co_1_closest , lon_2=lon_co_2_closest )

    lat_lon_dimension_para = 'lat' + requested_lat_dim + ',lon' + requested_lon_dim

    # Creating the download url.
    dimension_url = 'https://goldsmr4.sci.gsfc.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/2014/01/MERRA2_400.tavg1_2d_slv_Nx.20140101.nc4.nc4?'
    dimension_url = dimension_url + lat_lon_dimension_para
    download_manager.download_path = 'dimension_scale'
    download_manager.download_urls = [dimension_url]

    # Since the dimension is only one file, we only need one connection.
    %time download_manager.start_download(1)

    file_path = os.path.join('dimension_scale', DownloadManager.get_filename(
        dimension_url))

    with xr.open_dataset(file_path) as ds_dim:
        df_dim = ds_dim.to_dataframe()

    lat_array = ds_dim['lat'].data.tolist()
    lon_array = ds_dim['lon'].data.tolist()

    # The log output helps evaluating the precision of the received data.
    log.info('Requested lat: ' + str((lat_1, lat_2)))
    log.info('Received lat: ' + str(lat_array))
    log.info('Requested lon: ' + str((lon_1, lon_2)))
    log.info('Received lon: ' + str(lon_array))


    file_path = os.path.join('download_wind', '*.nc4')

    try:
        with xr.open_mfdataset(file_path, concat_dim='date',
                               preprocess=extract_date) as ds_wind:
            print(ds_wind)
            df_wind = ds_wind.to_dataframe()

    except xr.MergeError as e:
        print(e)

    df_wind.reset_index(inplace=True)

    start_date = datetime.strptime(download_start_date, '%Y-%m-%d')
    df_wind['date_time_hours'] = df_wind.apply(calculate_datetime, axis=1)

    df_wind['date_utc'] = df_wind.apply(converting_timeformat_to_ISO8601, axis=1)

    # partial is used to create a function with pre-set arguments.
    calc_windspeed_2m = partial(calculate_windspeed, idx_u='U2M', idx_v='V2M')
    calc_windspeed_10m = partial(calculate_windspeed, idx_u='U10M', idx_v='V10M')
    calc_windspeed_50m = partial(calculate_windspeed, idx_u='U50M', idx_v='V50M')

    df_wind['v_2m'] = df_wind.apply(calc_windspeed_2m, axis=1)
    df_wind['v_10m']= df_wind.apply(calc_windspeed_10m, axis=1)
    df_wind['v_50m'] = df_wind.apply(calc_windspeed_50m, axis=1)

    file_path = os.path.join('download_roughness', '*.nc4')
    with xr.open_mfdataset(file_path, concat_dim='date',
                           preprocess=extract_date) as ds_rough:
        df_rough = ds_rough.to_dataframe()

    df_rough.reset_index(inplace=True)

    file_path = os.path.join('download_radiation', '*.nc4')
    try:
        with xr.open_mfdataset(file_path, concat_dim='date',
                               preprocess=extract_date) as ds_rad:
            print(ds_rad)
            df_rad = ds_rad.to_dataframe()

    except xr.MergeError as e:
        print(e)
    df_rad.reset_index(inplace=True)

    file_path = os.path.join('download_temperature', '*.nc4')
    try:
        with xr.open_mfdataset(file_path, concat_dim='date',
                               preprocess=extract_date) as ds_temp:
            print(ds_temp)
            df_temp = ds_temp.to_dataframe()
    except xr.MergeError as e:
        print(e)
    df_temp.reset_index(inplace=True)

    file_path = os.path.join('download_density', '*.nc4')
    try:
        with xr.open_mfdataset(file_path, concat_dim='date',
                               preprocess=extract_date) as ds_dens:
            print(ds_dens)
            df_dens = ds_dens.to_dataframe()
    except xr.MergeError as e:
        print(e)
    df_dens.reset_index(inplace=True)

    file_path = os.path.join('download_pressure', '*.nc4')
    try:
        with xr.open_mfdataset(file_path, concat_dim='date',
                               preprocess=extract_date) as ds_pres:
            print(ds_pres)
            df_pres = ds_pres.to_dataframe()
    except xr.MergeError as e:
        print(e)
    df_pres.reset_index(inplace=True)

    df = pd.merge(df_wind, df_rough, on=['date', 'lat', 'lon', 'time'])
    df = pd.merge(df, df_rad, on=['date', 'lat', 'lon', 'time'])
    df = pd.merge(df, df_temp, on=['date', 'lat', 'lon', 'time'])
    df = pd.merge(df, df_dens, on=['date', 'lat', 'lon', 'time'])
    df = pd.merge(df, df_pres, on=['date', 'lat', 'lon', 'time'])

    # Calculate height for h1 (displacement height +2m) and h2 (displacement height
    # +10m).
    df['h1'] = df.apply((lambda x:int(x['DISPH']) + 2), axis=1)
    df['h2'] = df.apply((lambda x:int(x['DISPH']) + 10), axis=1)

    df.drop('DISPH', axis=1, inplace=True)
    df.drop(['time', 'date'], axis=1, inplace=True)
    df.drop(['U2M', 'U10M', 'U50M', 'V2M', 'V10M', 'V50M'], axis=1, inplace=True)

    df['lat'] = df['lat'].apply(lambda x: lat_array[int(x)])
    df['lon'] = df['lon'].apply(lambda x: lon_array[int(x)])

    rename_map = {'date_time_hours': 'cumulated hours',
              'date_utc': 'timestamp',
              'v_2m': 'v1',
              'v_10m': 'v2',
              'Z0M': 'z0',
              'T2M': 'T',
              'RHOA': 'rho',
              'PS': 'p'
             }

    df.rename(columns=rename_map, inplace=True)

    # Change order of the columns
    columns = ['timestamp', 'cumulated hours', 'lat', 'lon',
            'v1', 'v2', 'v_50m',
            'h1', 'h2', 'z0', 'SWTDN', 'SWGDN', 'T', 'rho', 'p']
    df = df[columns]

    df.to_csv('weather_data_GER_2016.csv', index=False)

    # Write the results to sqlite database. Using the chunksize parameter makes
    # this cell not use so much memory. If the parameter is not set, the to_sql
    # function will try to write all rows at the same time. This uses too much
    # memory. If you have a lot of memory, you can remove the parameter or increase
    # it to speed this process up. If you have memory problemes, try decreasing the
    # chunksize.
    engine = sqlalchemy.create_engine(
        'sqlite:///' + 'weather_data_GER_2016.sqlite')

    df.to_sql('weather_data_GER_2016',
                 engine,
                 if_exists="replace",
                 chunksize=100000,
                 index=False
                 )

    # Here we define meta data of the resulting data package.
    # The meta data follows the specification at:
    # http://dataprotocols.org/data-packages/

    metadata = """
    name: opsd-weather-data
    title: Weather data
    long_description: >-
        Weather data differ significantly from the other data types used resp.
        provided by OPSD in that the sheer size of the data packages greatly
        exceeds OPSD's capacity to host them in a similar way as feed-in
        timeseries, power plant data etc. While the other data packages also
        offer a complete one-klick download of the bundled data packages with
        all relevant data this is impossible for weather datasets like MERRA-2 due
        to their size (variety of variables, very long timespan, huge geographical
        coverage etc.). It would make no sense to mirror the data from the NASA
        servers.
        Instead we choose to provide a documented methodological script
        (as a kind of tutorial). The method describes one way to automatically
        obtain the desired weather data from the MERRA-2 database and simplifies
        resp. unifies alternative manual data obtaining methods in a single
        script.
        It is recommended to study the the "Step-by-step user guide" (developer use
        case) on this platform to learn how to run the script.
        The data package contains a sample dataset for Germany and the year 2016
    version: "2017-07-03"
    keywords: [Open Power System Data, MERRA-2, wind, solar, temperature, density,
                pressure]
    geographical-scope: Worldwide (German sample dataset for 2016)
    description: Script for the download of MERRA-2 weather data
    resources:
        - path: weather_data_GER_2016.csv
          format: csv
          encoding: UTF-8
          schema:
            fields:
                - name: timestamp
                  type: date-time
                  format: YYYY-MM-DDTHH:MM:SSZ
                  description: Start of timeperiod in Coordinated Universal Time
                - name: cumulated hours
                  type: number
                  format: integer
                  description: summarized number of hours for the timeperiod of the dataset
                - name: lat
                  type: geopoint
                  format: lat
                  description: Latitude coordinates
                - name: lon
                  type: geopoint
                  format: lon
                  description: Longitude coordinates
                - name: v1
                  type: number
                  format: float
                  description: wind speed 2 meters above displacement height
                - name: v2
                  type: number
                  format: float
                  description: wind speed 10 meters above displacement height
                - name: v_50m
                  type: number
                  format: float
                  description: wind speed 50 meters above ground
                - name: h1
                  type: number
                  format: float
                  description: height above ground corresponding to v1
                - name: h2
                  type: number
                  format: integer
                  description: height above ground corresponding to v2
                - name: z0
                  type: number
                  format: integer
                  description: roughness length
                - name: SWTDN
                  type: number
                  format: float
                  description: total top-of-the-atmosphere horizontal radiation
                - name: SWGDN
                  type: number
                  format: float
                  description: total ground horizontal radiation
                - name: T
                  type: number
                  format: float
                  description: Temperature 2 meters above displacement height
                - name: rho
                  type: number
                  format: float
                  description: air density at surface
                - name: p
                  type: number
                  format: float
                  description: air pressure at surface

    licenses:
        - type: MIT license
          url: http://www.opensource.org/licenses/MIT
    sources:
        - name: MERRA-2
          web: https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/
          source: National Aeronautics and Space Administration - Goddard Space Flight Center
    contributors:
        - name: Martin Jahn
          email: martin.jahn@uni-flensburg.de
        - name: Jan Urbansky

    views: True
    documentation: https://github.com/Open-Power-System-Data/weather_data/blob/2017-07-05/main.ipynb
    last_changes: corrected typos, slight modifications (file names)
    """

    metadata = yaml.load(metadata)

    datapackage_json = json.dumps(metadata, indent=4, separators=(',', ': '))

    with open('datapackage.json', 'w') as f:
        f.write(datapackage_json)

    output_path = ''

    files = [
        'weather_data_GER_2016.csv',
        'weather_data_GER_2016.sqlite']

    with open(os.path.join(output_path, 'checksums.txt'), 'w') as f:
        for file_name in files:
            file_hash = get_sha_hash(os.path.join(output_path, file_name))
            f.write('{},{}\n'.format(file_name, file_hash))
