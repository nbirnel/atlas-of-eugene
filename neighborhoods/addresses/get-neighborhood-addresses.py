#!/usr/bin/env python
# coding: utf-8


from sys import argv

import geopandas as gpd

requested = argv[1]

neighborhoods = gpd.read_file(
    "data/city-of-eugene/Eugene_Neighborhoods_-_HUB.geojson"
)
neighborhood = neighborhoods[neighborhoods["NAME"] == requested]

addresses = gpd.read_file(
    "data/city-of-eugene/Eugene_Addresses_-_HUB.geojson", mask=neighborhood
).sort_values(
    by=[
        "street_name",
        "street_type_code",
        "pre_direction_code",
        "house_nbr",
        "house_suffix_code",
        "unit_id",
    ]
)
addresses['state'] = 'OR'

format_requested = requested.replace(" ", "-")
addresses[
    [
        "concat_address",
        "city_name",
        'state',
        "five_digit_zip_code",
        "four_digit_zip_code",
    ]
].to_csv(f"{format_requested}-addresses.csv", index=False)
