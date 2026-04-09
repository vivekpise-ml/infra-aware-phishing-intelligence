import pandas as pd
import time
import os

from src.infrastructure.infra_utils import (
    extract_domain,
    get_ip,
    get_whois_info,
    get_asn_info
)


def enrich_dataset(input_path, output_path, limit=None):

    df = pd.read_csv(input_path)

    results = []

    for i, row in df.iterrows():

        if limit and i >= limit:
            break

        url = row["url"]
        label = row["label"]

        print(f"[{i}] Processing: {url}")

        domain = extract_domain(url)
        ip = get_ip(domain)

        age, registrar = get_whois_info(domain)
        asn, country = get_asn_info(ip)

        results.append({
            "url": url,
            "label": label,
            "domain": domain,
            "domain_age_days": age,
            "registrar": registrar,
            "ip": ip,
            "asn": asn,
            "country": country
        })

        time.sleep(0.5)  # avoid rate limit

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)

    print("✅ Enriched dataset saved →", output_path)


if __name__ == "__main__":
    enrich_dataset(
        input_path="data/kaggle/balanced_urls.csv",
        output_path="data/kaggle/enriched_urls.csv",
        limit=500   # start small!
    )