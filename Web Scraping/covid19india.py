import json
import requests

# url_state = 'https://api.covid19india.org/data.json'
# request_state = requests.get(url_state)
# data_state = request_state.json()

# url_district = 'https://api.covid19india.org/v2/state_district_wise.json'
# request_district = requests.get(url_district)
# data_district = request_district.json()

# states = data_state['statewise'][1:]

# for state in states:
# 	district_raw = [obj for obj in data_district if obj['state'] == state['state']]
# 	districts = district_raw[0]['districtData']

# 	state['districts'] = districts

url = "https://pomber.github.io/covid19/timeseries.json"
countries = requests.get(url).json()
x = 'India'
r = [country for country in countries if country==x]
print(r)