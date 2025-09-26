#!/usr/bin/env python
# coding: utf-8

import sys

import geopandas as gpd

neighborhood = sys.argv[1]
print(neighborhood)
neighborhood_format = neighborhood.replace(" ", "_")

neighborhoods = gpd.read_file(
    "data/city-of-eugene/Eugene_Neighborhoods_-_HUB.geojson"
)
#neighborhood = "Downtown Neighborhood Association"
#neighborhood = "Northeast Neighbors"
mask = neighborhoods[neighborhoods["NAME"] == neighborhood]

addresses = gpd.read_file(
    "data/city-of-eugene/Eugene_Addresses_-_HUB.geojson", mask=mask
)
addresses = addresses.fillna("")

addresses["first_line"] = (
    addresses[
        [
            "house_nbr",
            "house_suffix_code",
            "pre_direction_code",
            "street_name",
            "street_type_code",
        ]
    ]
    .astype(str)
    .apply(lambda x: " ".join(x), axis=1)
    .replace("  *", " ", regex=True)
)
addresses["second_line"] = (
    addresses[["unit_type_code", "unit_id"]]
    .astype(str)
    .apply(lambda x: " ".join(x), axis=1)
    .replace("  *", " ", regex=True)
    .replace("^ *$", "", regex=True)
)

addresses[
    [
        "concat_address",
        "first_line",
        "second_line",
        "city_name",
        "five_digit_zip_code",
        "four_digit_zip_code",
    ]
].to_csv(f"{neighborhood_format}_addresses.csv", index=False)

