# This is an example of developing a script locally with the West Nile Virus data to share on Kaggle
# Once you have a script you're ready to share, paste your code into a new script at:
#	https://www.kaggle.com/c/predict-west-nile-virus/scripts/new

# Code is borrowed from this script: https://www.kaggle.com/users/213536/vasco/predict-west-nile-virus/west-nile-heatmap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def draw_affected_area(input_data,output_dir, mode=None):
    mapdata = np.loadtxt("../west_nile/input/mapdata_copyright_openstreetmap_contributors.txt")
    lon_lat_box = (-88, -87.5, 41.6, 42.1)
    aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
    date_list = input_data['Date'].drop_duplicates().values
    year_list, year_month_list = parse_date(date_list)

    if mode == 'year':
        for year in year_list:
            plt.figure(figsize=(10, 14))
            plt.imshow(mapdata,
                       cmap=plt.get_cmap('gray'),
                       extent=lon_lat_box,
                       aspect=aspect)
            present_locations = input_data[['Longitude', 'Latitude']].loc[input_data['Date'].apply(lambda x: year in x) & input_data['WnvPresent'] == 1].drop_duplicates().values
            non_present_locations = input_data[['Longitude', 'Latitude']].loc[input_data['Date'].apply(lambda x: year in x) & input_data['WnvPresent'] == 0].drop_duplicates().values

            plt.scatter(non_present_locations[:, 0], non_present_locations[:, 1], marker='o', color='blue')
            plt.scatter(present_locations[:, 0], present_locations[:, 1], marker='x', color='red')
            plt.title(year+'-predict')
            plt.savefig(output_dir+ '/'+ year +'-predict.png')
    elif mode == 'year_month':

        for year_month in year_month_list:
            plt.figure(figsize=(10, 14))
            plt.imshow(mapdata,
                       cmap=plt.get_cmap('gray'),
                       extent=lon_lat_box,
                       aspect=aspect)
            present_locations = input_data[['Longitude', 'Latitude']].loc[input_data['Date'].apply(lambda x: year_month in x) & input_data['WnvPresent'] == 1].drop_duplicates().values
            non_present_locations = input_data[['Longitude', 'Latitude']].loc[input_data['Date'].apply(lambda x: year_month in x) & input_data['WnvPresent'] == 0].drop_duplicates().values

            plt.scatter(non_present_locations[:, 0], non_present_locations[:, 1], marker='o', color='blue')
            plt.scatter(present_locations[:, 0], present_locations[:, 1], marker='x', color='red')
            plt.title(year_month + '-predict')
            plt.savefig(output_dir + '/' + year_month + '-predict.png')
    else:
        present_locations = input_data[['Longitude', 'Latitude']].loc[input_data[
                'WnvPresent'] == 1].drop_duplicates().values
        non_present_locations = input_data[['Longitude', 'Latitude']].loc[input_data[
                'WnvPresent'] == 0].drop_duplicates().values
        plt.figure(figsize=(10, 14))
        plt.imshow(mapdata,
                   cmap=plt.get_cmap('gray'),
                   extent=lon_lat_box,
                   aspect=aspect)
        plt.scatter(non_present_locations[:, 0], non_present_locations[:, 1], marker='o', color='blue')
        plt.scatter(present_locations[:, 0], present_locations[:, 1], marker='x', color='red')
        plt.title('predict')
        plt.savefig(output_dir + '/predict.png')

def parse_date(date_list):
    year_list = []
    year_month_list = []
    for date in date_list:
        date = date.split('-')
        year_list.append(date[0])
        year_month_list.append(date[0]+'-'+date[1])
    return list(set(year_list)),list(set(year_month_list))