from functools import cache
import pickle

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import contains, intersects, make_valid

from . import landowners

neighborhood_abbreviations = [
    "ABC",
    "ANA",
    "CYNA",
    "CWN",
    "CAN",
    "DNA",
    "FNA",
    "FAN",
    "GIN",
    "HNA",
    "ICCO",
    "JWN",
    "LHVC",
    "NEN",
    "RRCO",
    "SCCO",
    "SUNA",
    "SEN",
    "SHiNA",
    "TNA",
    "WEC",
    "WUN",
    "WCC",
]

mfh_types = [
    "APART",
    "Apartment, High Rise, Shell",
    "Apartments - High Rise",
    "Dormitory",
    "Labor Dormitory",
    "Mixed Retail w/Residential",
    "Multiple Res",
    "Multiple Res Senior-Low Rise",
    "Multiple Res-Assisted Living",
    "Multiple Res-Retirement Com",
    "Rooming House",
]

acres_per_sq_meter = 0.000247105


def single_median(series):
    return series.mode().loc[0]

def str_as_float(string):
    if string == "":
        return float(0)
    return float(string)


def str_as_int(string):
    if string == "":
        return int(0)
    return int(string)


def format_percent(f: float, smart=False, ndigits=None) -> str:
    if smart:
        if f == 0.0:
            pass
        elif f < 0.0001:
            ndigits = 3
        elif f < 0.001:
            ndigits = 2
        elif f < 0.01:
            ndigits = 1
    try:
        return f"{round(f*100, ndigits)}%"
    except ValueError:
        return ""


def format_thousands(n) -> str:
    return f"{round(n/10**3):,}K"


def format_dollars(dollars):
    return f"${round(dollars):,}"


def format_million_dollars(dollars):
    return f"${round(dollars/10**6):,}M"


def add_display_columns(df, ratios=True):

    df["assessed_value_100k"] = df["assessed_value"] / 10**5
    df["real_market_value_100k"] = df["real_market_value"] / 10**5

    df["Acreage"] = df["acreage"].round(2)

    dollar_columns = [
        "assessed_value",
        "max_assessed_value",
        "real_market_value",
    ]

    if ratios is True:

        dollar_columns += [
            "assessed_value_per_acre",
            "real_market_value_per_acre",
        ]

        df["assessed_value_per_acre"] = df["assessed_value"] / df["acreage"]

        df["real_market_value_per_acre"] = (
            df["real_market_value"] / df["acreage"]
        )

        df["assessed_value_per_acre_100k"] = (
            df["assessed_value_per_acre"] / 10**5
        ).round(2)

        df["real_market_value_per_acre_100k"] = (
            df["real_market_value_per_acre"] / 10**5
        ).round(2)

        df["assessed_value_as_percentage_of_real_market_value"] = (
            df["assessed_value"] / df["real_market_value"]
        ).round(2)

        df["Assessed Value as Percentage of Real Market Value"] = (
            df["assessed_value_as_percentage_of_real_market_value"]
        ).apply(format_percent)

    for column in dollar_columns:
        new_column = " ".join(column.split("_")).title()
        df[new_column] = df[column].astype(int).map("${:,d}".format)

    return df


def modify_aggregate_columns(df, asint=("Year Built",)):
    df = add_display_columns(df, ratios=False)
    for col in asint:
        df[col] = df[col].astype(int)
    display_columns = [
        col
        for col in df.columns
        if col.lower() != col and col != "Neighborhood"
    ]
    new_cols = [f"Median {col}" for col in display_columns]
    rename = dict(zip(display_columns, new_cols))
    return df.rename(columns=rename)


@cache
def get_wards():
    gdf = gpd.read_file("data/city-of-eugene/Eugene_Wards_-_HUB.geojson")
    gdf.geometry = gdf.geometry.apply(make_valid)
    gdf["ward_number"] = gdf.ward_number.astype('category')
    return gdf.rename(
        columns={"councilor": "Councilor"}
    )


def get_ward(geom):
    wards = get_wards()

    for n in wards.itertuples():
        if contains(n.geometry, geom):
            return n.ward_number
    for n in wards.itertuples():
        if intersects(n.geometry, geom):
            return n.ward_number
    return np.nan


@cache
def get_neighborhoods(sort_column="POP_2000"):
    if sort_column == "POP_2000":
        ascending = False
    else:
        ascending = True

    nas = gpd.read_file(
        "data/city-of-eugene/Eugene_Neighborhoods_-_HUB.geojson"
    ).sort_values(by="NAME")
    gdf = nas[nas["NAME"] != "OUT"].copy()
    gdf.rename(columns={"NAME": "neighborhood"}, inplace=True)
    gdf["neighborhood_abbreviation"] = neighborhood_abbreviations
    gdf["acres"] = gdf.to_crs("EPSG:32610").area * acres_per_sq_meter
    gdf["population_per_acre"] = gdf.POP_2000 / gdf.acres
    gdf["Acres"] = gdf.acres.map(lambda x: round(x, 2))
    gdf["Population per Acre"] = gdf.population_per_acre.map(
        lambda x: round(x, 2)
    )

    return gdf.sort_values(by=sort_column, ascending=ascending)


def get_neighborhood(geom):
    neighborhoods = get_neighborhoods()

    for n in neighborhoods.itertuples():
        if contains(n.geometry, geom):
            return n.neighborhood
    for n in neighborhoods.itertuples():
        if intersects(n.geometry, geom):
            return n.neighborhood
    return np.nan


def get_pickle(picklefile):
    with open(picklefile, "rb") as f:
        return pickle.load(f)


def get_city_gdf():
    return gpd.read_file(
        "data/city-of-eugene/Eugene_City_Limits_-_HUB.geojson"
    ).to_crs("EPSG:4269")



def get_city_limits():
    pass


def get_addresses_gdf(regenerate=False):
    if regenerate is False:
        try:
            return get_pickle("cache/addresses_gdf.pickle")
        except:
            pass
    gdf = gpd.read_file("data/city-of-eugene/Eugene_Addresses_-_HUB.geojson")
    with open("cache/addresses_gdf.pickle", "wb") as f:
        pickle.dump(gdf, f, pickle.HIGHEST_PROTOCOL)
    return gdf


def get_addresses(regenerate=False):
    if regenerate is False:
        try:
            return get_pickle("cache/addresses.pickle")
        except:
            pass
    df = pd.read_csv("data/city-of-eugene/Eugene_Addresses_-_HUB.csv")
    with open("cache/addresses.pickle", "wb") as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)
    return df


def get_eugene_density(regenerate=False):
    if regenerate is False:
        try:
            return get_pickle("cache/eugene_density_gdf.pickle")
        except:
            pass
    oregon = gpd.read_file(
        "data/us-census/tl_2024_41_tabblock20/tl_2024_41_tabblock20.shp",
        columns=[
            "GEOID20",
            "HOUSING20",
            "COUNTYFP20",
            "POP20",
            "geometry",
        ],
    )

    fips = {
        "OR": "41",
        "Lane": "039",
        "Eugene": "4123850",
    }

    lane = oregon[oregon.COUNTYFP20 == fips["Lane"]].copy()
    del oregon

    city_limits = make_valid(get_city_gdf().geometry[0])

    gdf = lane[lane.geometry.intersects(city_limits)].copy()
    del lane

    gdf.drop(columns="COUNTYFP20")
    gdf["acres"] = gdf.to_crs("EPSG:32610").area * acres_per_sq_meter
    gdf["population_per_acre"] = gdf.POP20 / gdf.acres
    gdf["units_per_acre"] = gdf.HOUSING20 / gdf.acres

    gdf["Acres"] = gdf.acres.map(lambda x: round(x, 2))
    gdf["Population per Acre"] = gdf.population_per_acre.map(
        lambda x: round(x, 2)
    )
    gdf["Housing Units per Acre"] = gdf.units_per_acre.map(
        lambda x: round(x, 2)
    )

    employment_gdf = gpd.read_file(
        "data/otm/census-otm-employment-2025.zip",
        layer=0,
        columns=["id", "c000", "geometry"],
    )

    employment_gdf.rename(columns={"c000": "Total Employed"}, inplace=True)
    employment_gdf["Total Employed"] = employment_gdf[
        "Total Employed"
    ].astype(int)

    gdf = gdf.merge(
        employment_gdf[["id", "Total Employed"]],
        left_on="GEOID20",
        right_on="id",
    )

    gdf["jobs_per_acre"] = gdf["Total Employed"] / gdf["acres"]
    gdf["Jobs per Acre"] = gdf.jobs_per_acre.map(lambda x: round(x, 2))

    gdf["Population + Jobs"] = gdf["Total Employed"] + gdf.POP20
    gdf["population_and_jobs_per_acre"] = gdf["Population + Jobs"] / gdf.acres
    gdf["Population + Jobs per Acre"] = gdf.population_and_jobs_per_acre.map(
        lambda x: round(x, 2)
    )
    gdf["population_and_logs_per_acre_log"] = np.log(
        gdf.population_and_jobs_per_acre
    )

    unit_guidelines = {
        "little service": 0,
        "frequent bus min": 10,
        "BRT min": 15,
        "rail min": 20,
        "frequent bus target": 30,
        "BRT target": 40,
        "rail target": 75,
    }
    unit_edges = list(unit_guidelines.values()) + [10**3]

    population_guidelines = {
        "little service": 0,
        "BRT target": 17,
        "rail target": 55,
    }
    population_edges = list(population_guidelines.values()) + [10**3]

    gdf["Unit Density Guidelines"] = pd.cut(
        gdf.units_per_acre,
        unit_edges,
        labels=unit_guidelines.keys(),
    )

    gdf["Population Density Guidelines"] = pd.cut(
        gdf.population_per_acre,
        population_edges,
        labels=population_guidelines.keys(),
    )

    gdf["Population and Job Density Guidelines"] = pd.cut(
        gdf.population_and_jobs_per_acre,
        population_edges,
        labels=population_guidelines.keys(),
    )

    with open("cache/eugene_density_gdf.pickle", "wb") as f:
        pickle.dump(gdf, f, pickle.HIGHEST_PROTOCOL)

    return gdf


def get_census_2020_gdf(regenerate=False):
    if regenerate is False:
        try:
            return get_pickle("cache/census_2020_gdf.pickle")
        except:
            pass
    oregon = gpd.read_file(
        "data/us-census/tl_2024_41_tabblock20/tl_2024_41_tabblock20.shp"
    )
    lane_fp = "039"
    lane = oregon[oregon.COUNTYFP20 == lane_fp]
    del oregon
    city_gdf = gpd.read_file(
        "data/city-of-eugene/Eugene_City_Limits_-_HUB.geojson"
    ).to_crs("EPSG:4269")
    city_limits = make_valid(city_gdf.geometry[0])
    return lane[lane.geometry.intersects(city_limits)].copy()


def get_account_taxpayer_lot(regenerate=False):
    if regenerate is False:
        try:
            return get_pickle("cache/account_tax_payer_lot.pickle")
        except:
            pass

    df = pd.read_csv(
        "data/scrape-lane-county/account_tax_payer_lot.csv",
        keep_default_na=False,
        dtype={"account_number": int, "prop_class": "category"},
        converters={
            "acreage": str_as_float,
            "tca": str_as_int,
        },
        usecols=[
            "account_number",
            "tax_payer",
            "map_and_tax_lot_number",
            "situs_address",
            "acreage",
            "prop_class",
            "situs_city_state_zip",
            "mailing_address_1",
            "mailing_address_2",
            "mailing_address_3",
            "mailing_city_state_zip",
        ],
    )

    df = df[df.situs_city_state_zip.str.contains("EUGENE")]

    # df.drop(columns="situs_city_state_zip", inplace=True)

    df.rename(
        columns={
            "account_number": "account",
            "map_and_tax_lot_number": "maptaxlot",
        },
        inplace=True,
    )
    # df.dropna(inplace=True)
    df = df[df.acreage != 0]
    df["catenated_mailing_address"] = (
        df.mailing_address_1
        + df.mailing_address_2
        + df.mailing_address_3
        + df.mailing_city_state_zip
    )

    with open("cache/account_tax_payer_lot.pickle", "wb") as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

    return df


def get_assessments(regenerate=False):
    if regenerate is False:
        try:
            return get_pickle("cache/assessments.pickle")
        except:
            pass

    df = pd.read_csv(
        "data/scrape-lane-county/assessments.csv",
        dtype={
            "account_id": int,
            "assessed_value": int,
            "max_assessed_value": int,
            "real_market_value": int,
        },
    )
    df = df.rename(columns={"account_id": "account"})
    df.dropna(inplace=True)
    df.sort_values(by="account", inplace=True)

    with open("cache/assessments.pickle", "wb") as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

    return df


def get_properties(regenerate=False):
    """
    Return properties DataFrame, sorted by maptaxlot
    """
    if regenerate is False:
        try:
            return get_pickle("cache/properties.pickle")
        except:
            pass

    df = pd.read_csv(
        "data/scrape-lane-county/properties.csv",
        usecols=[
            "map_and_tax_lot",
            "owner",
        ],
    )
    df = df.rename(columns={"map_and_tax_lot": "maptaxlot"})
    df.dropna(inplace=True)
    df.sort_values(by="maptaxlot", inplace=True)

    with open("cache/properties.pickle", "wb") as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

    return df


def get_tax_data(regenerate=False):
    """
    Return tax data DataFrame, sorted by maptaxlot
    """
    if regenerate is False:
        try:
            return get_pickle("cache/tax_data.pickle")
        except:
            pass

    last_assessment = get_assessments().groupby("account").last()

    df = pd.merge(
        get_account_taxpayer_lot(),
        last_assessment,
        how="left",
        left_on="account",
        right_index=True,
    ).sort_values(by="maptaxlot")

    df = add_display_columns(df)
    with open("cache/tax_data.pickle", "wb") as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

    return df


def get_assessed_gdf(regenerate=False):
    """
    Return GeoDataFrame of taxlots, ordered by neighborhood.
    """
    if regenerate is False:
        try:
            return get_pickle("cache/assessed_gdf.pickle")
        except:
            pass

    gdf = gpd.read_file(
        "data/city-of-eugene/Eugene_Taxlots_-_HUB.geojson",
        columns=[
            "maptaxlot",
            "map",
            "mapacres",
            "geometry",
        ],
    )
    gdf["maptaxlot"] = gdf["maptaxlot"].astype(int)

    gdf = gdf.sort_values(by="maptaxlot").merge(
        get_tax_data(), on="maptaxlot", how="left"
    )

    gdf = gdf.merge(get_properties(), on="maptaxlot", how="left")

    gdf.dropna(subset="assessed_value", inplace=True)
    gdf.account = gdf.account.astype(int)
    gdf["neighborhood"] = gdf.geometry.apply(get_neighborhood)
    gdf["ward"] = gdf.geometry.apply(get_ward)

    gdf = gdf.sort_values(by="neighborhood").merge(
        get_neighborhoods(sort_column="neighborhood")[
            ["neighborhood", "neighborhood_abbreviation"]
        ],
        on="neighborhood",
    )

    gdf.drop_duplicates(subset=["maptaxlot"], inplace=True)
    ## Here is what we want to do, but it is slow:
    #    gdf["full_address"] = gdf[
    #        [
    #            "mailing_address_1",
    #            "mailing_address_2",
    #            "mailing_address_3",
    #            "mailing_city_state_zip",
    #        ]
    #    ].apply(lambda x: " ".join(x.dropna()), axis=1)

    ## So instead we do this:
    gdf["mailing_address"] = (
        gdf[
            [
                "maptaxlot",
                "mailing_address_1",
                "mailing_address_2",
                "mailing_address_3",
                "mailing_city_state_zip",
            ]
        ]
        .set_index("maptaxlot")
        .stack()
        .groupby(level=0, sort=False)
        .agg(" ".join)
        .values
    )

    gdf["mailing_address"] = gdf.mailing_address.str.replace(
        "  *", " ", regex=True
    )

    gdf["owner_group"] = gdf.owner
    gdf["taxpayer_group"] = gdf.tax_payer
    for section in [landowners.private, landowners.public]:
        for key, values in section.items():
            owner = gdf["owner_group"]
            gdf["owner_group"] = owner.mask(owner.isin(values), key)
            taxpayer = gdf["taxpayer_group"]
            gdf["taxpayer_group"] = taxpayer.mask(taxpayer.isin(values), key)

    gdf["Owner Group"] = gdf["owner_group"]
    gdf["Taxpayer Group"] = gdf["taxpayer_group"]
    with open("cache/assessed_gdf.pickle", "wb") as f:
        pickle.dump(gdf, f, pickle.HIGHEST_PROTOCOL)

    return gdf


def get_taxmap_gdf(regenerate=False):
    if regenerate is False:
        try:
            return get_pickle("cache/taxmap_gdf.pickle")
        except:
            pass
    gdf = get_assessed_gdf()[
        [
            "map",
            "mapacres",
            "acreage",
            "assessed_value",
            "max_assessed_value",
            "real_market_value",
            "geometry",
        ]
    ].dissolve(by="map", aggfunc="sum", as_index=False)
    gdf = gdf[gdf.assessed_value != 0.0]
    gdf = add_display_columns(gdf)
    gdf.rename(columns={"map": "Map"}, inplace=True)

    with open("cache/taxmap_gdf.pickle", "wb") as f:
        pickle.dump(gdf, f, pickle.HIGHEST_PROTOCOL)

    return gdf


def get_neighborhood_gdf(regenerate=False):
    if regenerate is False:
        try:
            return get_pickle("cache/neighborhood_gdf.pickle")
        except:
            pass

    in_neighborhood = (
        get_assessed_gdf()[
            [
                "mapacres",
                "acreage",
                "assessed_value",
                "max_assessed_value",
                "real_market_value",
                "neighborhood",
            ]
        ]
        .dropna(subset="neighborhood")
        .sort_values(by="neighborhood")
    )

    by_na = in_neighborhood.groupby("neighborhood").sum()
    by_na = add_display_columns(by_na)

    gdf = get_neighborhoods(sort_column="neighborhood")[
        ["neighborhood", "geometry"]
    ].merge(by_na, how="left", left_on="neighborhood", right_index=True)

    gdf.rename(columns={"neighborhood": "Neighborhood"}, inplace=True)

    with open("cache/neighborhood_gdf.pickle", "wb") as f:
        pickle.dump(gdf, f, pickle.HIGHEST_PROTOCOL)

    return gdf


def get_residential_buildings(regenerate=False):
    if regenerate is False:
        try:
            return get_pickle(
                "cache/assessments/residential_buildings.pickle"
            )
        except:
            pass

    df = (
        pd.read_csv(
            "data/scrape-lane-county/residential_buildings.csv",
            usecols=[
                "taxlot",
                "year_built",
                "basement_floor_base",
                "first_floor_base",
                "second_floor_base",
                "attic_floor_base",
                "total_floor_base",
                "basement_garage",
                "attached_garage",
                "detached_garage",
                "attached_carport",
            ],
        )
        .dropna(subset=["taxlot", "year_built"])
        .rename(columns={"taxlot": "maptaxlot"})
        .sort_values(by="maptaxlot")
    )

    df["Year Built"] = df["year_built"].astype(int)

    with open("cache/residential_buildings.pickle", "wb") as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

    return df


def get_single_family_housing_gdf(regenerate=False):
    if regenerate is False:
        try:
            return get_pickle("cache/single_family_housing.pickle")
        except:
            pass

    assessed_gdf = get_assessed_gdf()
    gdf = assessed_gdf[
        assessed_gdf.prop_class.isin(
            [
                "10 Miscellaneous Residential",
                #'100 Residential Vacant',
                "101 Residential Improved",
                "106 Residential Improved Waterfront",
                #'109 Residential Improved Manufactured Structure',
                "121 Residential Commercial Zoned Improved",
                "131 Residential Industrial Zoned Improved",
                #'139 Residential Industrial Zoned Manufactured Structure',
                #'190 Residential Potential Development Vacant',
                #'191 Residential Potential Development Improved',
                #'196 Residential Waterfront Vacant',
                #'199 Residential Potential Development Manufactured Structure',
                #'210 Commercial Residential Zoned Vacant',
                "211 Commercial Residential Zoned Improved",
                "216 Commercial Residential Zoned Waterfront",
            ]
        )
    ].copy()

    gdf = add_display_columns(gdf)

    gdf = gdf.sort_values(by="maptaxlot")
    gdf = gdf.merge(get_residential_buildings(), on="maptaxlot")

    gdf["Has Garage"] = gdf[
        [
            "basement_garage",
            "attached_garage",
            "detached_garage",
            "attached_carport",
        ]
    ].any(axis="columns")

    gdf["garage_sq_ft"] = (
        gdf.basement_garage.fillna(0)
        + gdf.attached_garage.fillna(0)
        + gdf.detached_garage.fillna(0)
        + gdf.attached_carport.fillna(0)
    ).replace(0.0, np.nan)

    gdf["Buildings Footprint"] = (
        gdf.first_floor_base.fillna(value=0)
        + gdf.detached_garage.fillna(value=0)
    ).astype(int)

    square_feet_per_acre = 43560
    gdf["Lot Square Footage"] = (gdf.acreage * square_feet_per_acre).astype(
        int
    )
    gdf["lot_coverage"] = (
        gdf["Buildings Footprint"] / gdf["Lot Square Footage"]
    )
    gdf["Lot Coverage"] = gdf.lot_coverage.apply(format_percent)

    gdf["Decade Built"] = (gdf["Year Built"] // 10 * 10).astype(str)

    gdf["Tenancy"] = gdf.situs_address.str.slice(
        stop=8
    ) == gdf.mailing_address_1.str.slice(stop=8)
    gdf["Tenancy"] = (
        gdf["Tenancy"]
        .map(lambda x: {False: "Rented", True: "Owner Occupied"}.get(x))
        .astype("category")
    )

    def get_floors(row):
        if not pd.isnull(row.attic_floor_base):
            return 3
        if not pd.isnull(row.second_floor_base):
            return 2
        return 1

    gdf["floor_number"] = gdf.apply(get_floors, axis=1)
    gdf["description"] = "Single Family Housing"
    gdf["1"] = 1

    with open("cache/single_family_housing.pickle", "wb") as f:
        pickle.dump(gdf, f, pickle.HIGHEST_PROTOCOL)

    return gdf


def get_single_family_housing_neighborhood_median_gdf(regenerate=False):
    if regenerate is False:
        try:
            return get_pickle(
                "cache/single_family_housing_neighborhood_median_gdf.pickle"
            )
        except:
            pass
    df = (
        get_single_family_housing_gdf()[
            [
                "acreage",
                "assessed_value",
                "max_assessed_value",
                "real_market_value",
                "neighborhood",
                "Year Built",
            ]
        ]
        .dropna(subset="neighborhood")
        .groupby("neighborhood", as_index=False)
        .median()
        .rename(columns={"neighborhood": "Neighborhood"})
    )

    gdf = get_neighborhood_gdf()[["Neighborhood", "geometry"]].merge(
        df, on="Neighborhood"
    )

    gdf = modify_aggregate_columns(gdf)

    with open(
        "cache/single_family_housing_neighborhood_median_gdf.pickle", "wb"
    ) as f:
        pickle.dump(gdf, f, pickle.HIGHEST_PROTOCOL)

    return gdf


def get_commercial_buildings_gdf(regenerate=False):
    if regenerate is False:
        try:
            return get_pickle("cache/commercial_buildings.pickle")
        except:
            pass

    df = (
        pd.read_csv(
            "data/scrape-lane-county/commercial_improvements.csv",
            usecols=[
                "taxlot",
                "year_built",
                "description",
                "floor_number",
                "sq_ft",
                "grade",
                "occupancy_number",
                "effective_year_built",
            ],
            dtype={"description": "category", "taxlot": int},
        )
        .dropna()
        .rename(columns={"taxlot": "maptaxlot"})
        .sort_values(by="maptaxlot")
    )

    df["floor_number"] = df["floor_number"].replace("B", "1")
    df["floor_number"] = df["floor_number"].replace("M1", "0")
    df["floor_number"] = df["floor_number"].astype(int)
    df["Year Built"] = df["year_built"].astype(int)
    df["Decade Built"] = (df["Year Built"] // 10 * 10).astype(str)

    gdf = get_assessed_gdf().sort_values(by="maptaxlot")
    gdf = gdf.merge(df, on="maptaxlot", how="inner")
    gdf["1"] = 1

    with open("cache/commercial_buildings.pickle", "wb") as f:
        pickle.dump(gdf, f, pickle.HIGHEST_PROTOCOL)

    return gdf


def get_non_residential_gdf(regenerate=False):
    if regenerate is False:
        try:
            get_pickle("cache/non_residential_gdf.pickle")
        except:
            pass

    gdf = get_commercial_buildings_gdf()
    gdf = gdf[~gdf.description.isin(mfh_types)]

    with open("cache/non_residential_gdf.pickle", "wb") as f:
        pickle.dump(gdf, f, pickle.HIGHEST_PROTOCOL)

    return gdf


def get_multi_family_housing_gdf(regenerate=False):

    if regenerate is False:
        try:
            return get_pickle("cache/multi_family_housing_gdf.pickle")
        except:
            pass

    gdf = get_commercial_buildings_gdf()
    gdf = gdf[gdf.description.isin(mfh_types)]

    with open("cache/multi_family_housing_gdf.pickle", "wb") as f:
        pickle.dump(gdf, f, pickle.HIGHEST_PROTOCOL)

    return gdf


def boxplot_by_neighborhoods(dfl: list, fmt, outliers=True):
    """
    Accept dfl (list of pandas.DataFrame),
    fmt (an fstring format).
    Set outliers to False to not display outliers.
    Display a Boxplots for each neighborhood.
    """

    dfl = dfl[::-1]
    fig, ax = plt.subplots()

    _ = ax.boxplot(
        dfl,
        orientation="horizontal",
        tick_labels=neighborhood_abbreviations[::-1],
        flierprops={
            "markersize": 1,
            "markerfacecolor": "lightblue",
            "markeredgecolor": "lightblue",
        },
        showfliers=outliers,
    )
    labels = [
        fmt(t.get_text().replace("−", "-"))
        for t in ax.get_xticklabels()[1:-1]
    ]
    ax.set_xticks(ax.get_xticks()[1:-1], labels=labels)

    fig.set_figwidth(10)
    fig.set_figheight(8)

    plt.show()


def explore(
    gdf: gpd.GeoDataFrame,
    column=None,
    caption="",
    tooltip=False,
    title=None,
    legend=False,
    cmap="YlOrRd",
    **kwargs,
) -> folium.folium.Map:
    """
    Accept geopandas, kwargs.
    """

    legend_kwds = kwargs.get("legend_kwds", {}).update({"caption": caption})
    m = gdf.explore(
        cmap=cmap,
        tiles="cartodb positron",
        legend=legend,
        column=column,
        legend_kwds={"caption": caption},
        tooltip=tooltip,
        zoom_start=12,
        **kwargs,
    )

    if title:
        map_title = title
        title_html = f'<h1 align="center" >{map_title}</h1>'
        m.get_root().html.add_child(folium.Element(title_html))
    return m
