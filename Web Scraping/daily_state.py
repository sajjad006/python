import json
import requests

def state_daily_data(state_code):
    url_daily = 'https://api.covid19india.org/states_daily.json'
    request_daily = requests.get(url_daily)
    data_daily = request_daily.json()['states_daily']

    date      = [x["date"] for x in data_daily if x["status"] == 'Confirmed']
    confirmed = [x[state_code] for x in data_daily if x['status'] == 'Confirmed']
    recovered = [x[state_code] for x in data_daily if x['status'] == 'Recovered']
    deceased  = [x[state_code] for x in data_daily if x['status'] == 'Deceased']

    return {'date':date, 'confirmed': confirmed, 'recovered': recovered, 'deceased': deceased}


def district_daily_data(state, district):
    url_daily = 'https://api.covid19india.org/districts_daily.json'
    request_daily = requests.get(url_daily)
    data_daily = request_daily.json()['districtsDaily'][state][district]

    date      = [x["date"] for x in data_daily]
    active    = [x["active"] for x in data_daily]
    confirmed = [x["confirmed"] for x in data_daily]
    recovered = [x["recovered"] for x in data_daily]
    deceased  = [x["deceased"] for x in data_daily]

    return {'date': date, 'active': active, 'confirmed': confirmed, 'recovered': recovered, 'deceased': deceased}


print(state_daily_data('mh'))