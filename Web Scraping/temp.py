import json
import requests

data_state = json.loads(open('data.json').read())
data_district = json.loads(open('state_district_wise.json').read())

states = data_state['statewise'][1:]

for state in states:

	print('Name : ', state['state'])
	print('confirmed', state['confirmed'])
	print('deaths : ', state['deaths'])
	print('recovered : ', state['recovered'])
	print()

	district_raw = [obj for obj in data_district if obj['state'] == state['state']]
	districts = district_raw[0]['districtData']

	for district in districts:
		print(district['district'], district['active'], district['confirmed'], district['deceased'], district['recovered'])

	print()