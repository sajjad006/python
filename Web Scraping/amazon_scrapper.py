from bs4 import BeautifulSoup
import requests

search = input("Enter search term:")
search = 'https://www.amazon.in/s?k={}'.format(search.replace(' ', '+'))

source = requests.get(search).text
soup = BeautifulSoup(source, 'lxml')

items = soup.find_all('div', class_='s-result-list')

print(items[0])