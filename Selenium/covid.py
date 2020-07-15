from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time

PATH = 'C:\Program Files (x86)\chromedriver.exe'
driver = webdriver.Chrome(PATH)

driver.get("https://covid19-india-predictor.netlify.app/")

try:
    table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "tbody")))
    rows = driver.find_elements_by_tag_name("tr")
    
    for row in rows:
    	cells = row.find_elements_by_tag_name("td")

    	for cell in cells:
    		print(cell.text)
finally:
    driver.quit()
