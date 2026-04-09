import socket
import whois
import tldextract
import requests
from datetime import datetime


def extract_domain(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"


def get_ip(domain):
    try:
        return socket.gethostbyname(domain)
    except:
        return None


def get_whois_info(domain):
    try:
        w = whois.whois(domain)

        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        if creation_date:
            age_days = (datetime.now() - creation_date).days
        else:
            age_days = None

        registrar = w.registrar

        return age_days, registrar

    except:
        return None, None
    


def get_asn_info(ip):
    if not ip:
        return None, None

    try:
        url = f"http://ip-api.com/json/{ip}"
        res = requests.get(url, timeout=5).json()

        asn = res.get("as", None)
        country = res.get("country", None)

        return asn, country

    except:
        return None, None