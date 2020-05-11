from bs4 import BeautifulSoup
import requests
import urllib

source = requests.get('https://www.donboscoparkcircus.org/gallery.aspx').text
soup = BeautifulSoup(source, 'lxml')

link = soup.find('div', class_='newgallery').article.a['href']
source = requests.get('https://www.donboscoparkcircus.org/{}'.format(link)).text
soup = BeautifulSoup(source, 'lxml')
images = soup.find_all('img', class_='gallery_detail')

for image in images:
    img = image['src']
    src = 'www.donboscoparkcircus.org/{}'.format(img)