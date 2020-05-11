import urllib
import requests
from bs4 import BeautifulSoup

source = requests.get('https://www.indiatimes.com/').text
soup = BeautifulSoup(source, 'lxml')

articles = soup.find_all('div', class_='card-div')
articles = articles[0:len(articles)-1]

c = 0
for article in articles:
    title = article.a['title']
    url = article.a['href']
    img = article.img['src']

    r = requests.get(img, allow_redirects=True)
    open(f'{c}.jpg', 'wb').write(r.content)
    c+=1

    # print(url)
    text_source = requests.get(url).text
    text_soup = BeautifulSoup(text_source, 'lxml')

    div = text_soup.article.find_all('div', class_='left-container')
    # print(len(div))
    ps = div[0].find_all('p', class_=None)
    for p in ps:
    	print(p.text)
    break