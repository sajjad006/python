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

# country starts at index 9. 
countries = soup.find_all('tr')
countries = countries[9:221]


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

	# print("country ", rows[0])
	# print("total", rows[1])
	# print("active", rows[6], type(rows[6]), rows[6].isnumeric())
	# print("new", rows[2].replace('+', ''))
	# print("death", rows[3], type(rows[3]), rows[3].strip().isnumeric())
	# print("new death", rows[4].replace('+', ''))
	# print("recovered", rows[5])
	# print("test", rows[10])
	# print(row.text, end=', ')

	name = rows[0].replace(' ', '_')
	print(f"http://img.freeflagicons.com/thumb/glossy_square_icon/{name}/{name}_640.png".lower())

	print(rows[0])

# activecase, deaths, name, newcase, newdeath, recovered, tests, totalcase