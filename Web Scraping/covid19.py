from bs4 import BeautifulSoup
import pandas as pd
import requests
import sys

source = requests.get('https://www.worldometers.info/coronavirus/').text
soup = BeautifulSoup(source, 'lxml')

# soup_text = soup.encode(sys.stdout.encoding, errors='replace')
# soup2 = BeautifulSoup(soup_text, 'lxml')
# print(soup2)

# columns = (soup.find_all('th'))[:13]

# columns = ['Country', 'TotalCases', 'NewCases', 'TotalDeaths', 'NewDeaths', 'TotalRecovered', 'ActiveCases'	'Serious', 'Cases/1M', 'Deaths/1M', 'TotalTests', 'Tests/1M pop']
# df = pd.DataFrame(columns=columns)

# china 224
# country starts at index 9. 
countries = soup.find_all('tr')
world = countries[231]
countries = countries[9:224]
countries.append(world)
# countries = countries[:len(countries)-8]

# for country in countries:
# 	rows = country.find_all('td')
# 	# print(countries[9])
# 	for row in rows:
# 		print(row.text, end=', ')

# 	print()
def my_int(str):
    if str.isnumeric():
        return int(str)
    else:
        return 0

for country in countries:
	rows = country.find_all('td')
	# print(countries[9])
	rows = list(map(lambda x:str(x.text).replace(',', ''), rows))

	print(rows[1], rows[2])
	# print("country ", rows[1])
	# print("total", rows[2])
	# print("active", rows[7].replace('+', ''))
	# print("new", rows[3].replace('+', ''))
	# print("death", rows[4])
	# print("new death", rows[5].replace('+', ''))
	# print("recovered", rows[6])
	# print("test", rows[11])
	# print(row.text, end=', ')

	# name = rows[0].replace(' ', '_')
	# print(f"http://img.freeflagicons.com/thumb/glossy_square_icon/{name}/{name}_640.png".lower())

	# print(rows[0])

# activecase, deaths, name, newcase, newdeath, recovered, tests, totalcase