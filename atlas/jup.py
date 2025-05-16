private_mail = (
    private_gdf[["mailing_address", "real_market_value", "acreage"]]
    .groupby("mailing_address")
    .sum()
    .reset_index()
    .sort_values(by="real_market_value", ascending=False)
)
private_mail["Real Market Value"] = private_mail.real_market_value.astype(
    int
).map("${:,d}".format)
private_mail["Mailing Address"] = private_mail.mailing_address
private_mail["Acreage"] = private_mail.acreage
private_mail[["Mailing Address", "Acreage", "Real Market Value"]].head(20)
