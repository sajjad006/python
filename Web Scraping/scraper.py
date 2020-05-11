from bs4 import BeautifulSoup
import requests
import csv

source = requests.get('http://coreyms.com').text
soup = BeautifulSoup(source, 'lxml')

csv_file = open('cms_scrape.csv', 'w')

csv_writer = csv.writer(csv_file)
csv_writer.writerow(['headline', 'summary', 'video_link'])

for article in soup.findAll('article'):

    heading = article.h2.a.text
    summary = article.div.p.text

    try:
        vid_src = article.find('iframe', class_='youtube-player')['src']
        vid_id = vid_src.split('/')[4]
        vid_id = vid_id.split('?')[0]

        ytb_link = 'https://www.youtube.com/watch?v={}'.format(vid_id)
    except:
        ytb_link=None


    print(heading, summary, ytb_link)
    print()

    csv_writer.writerow([heading, summary, ytb_link])

csv_file.close