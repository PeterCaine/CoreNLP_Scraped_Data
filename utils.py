from selenium import webdriver
from bs4 import BeautifulSoup
from tqdm import trange
import pickle
import time
import sys

def get_links_to_scrape(main_url, start, num_pages):
    '''
    goes to airline page, extracts html links to airlines for future scrape.
    '''
    driver = webdriver.Firefox()
    driver.get(main_url)
    driver.implicitly_wait(100)

    for n in trange(int(start)):
        button = driver.find_element_by_class_name("nav.next")
        button.click()
        time.sleep(2)

    url_list = []
    for n in trange(num_pages):
        urls = driver.find_elements_by_class_name('detailsLink')
        for url in urls[:10]:
            href = url.get_attribute('href')
            url_list.append(href)

        button = driver.find_element_by_class_name("nav.next")
        button.click()
        time.sleep(2)

    driver.quit()
    new_airlines = '\n'.join(url_list)
    with open('./airline_pickles/airlines.txt', 'a') as outfile:
        outfile.write(new_airlines)

    return url_list

def start_scrape(url, start=0):
    '''
    starts selenium web driver
    param: url - the page url of the page you wish to scrape
    param = int: if scrape crashes - change parameter (start) to page number where crash happened to restart from
    that point

    '''
    driver = webdriver.Firefox()
    driver.get(url)
    driver.implicitly_wait(100)

    for n in trange(start):
        button = driver.find_element_by_link_text("Next")
        button.click()
    return driver

def trip_advisor_scrape(driver, url, start=0, num_reviews=1500):
    basename = url.split("Reviews-")[-1]
    soup_list = []
    for n in trange(num_reviews):
        try:
            elements = driver.find_element_by_class_name('_36B4Vw6t')
            elements.click()
        except:
            pass
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        review_soup = soup.find_all('div', attrs={'class': 'Dq9MAugU T870kzTX LnVzGwUB'})
        soup_list.append(review_soup)
        try:
            button = driver.find_element_by_link_text("Next")
            button.click()
            time.sleep(1)
        except:
            sys.setrecursionlimit(10000)
            pickle.dump(soup_list, open(f'./airline_pickles/{basename}_{start}-{start + n}.pkl', 'wb'))
            break

    print('finished at review number: ', n+1)
    
    sys.setrecursionlimit(10000)
    pickle.dump(soup_list, open(f'./airline_pickles/{basename}_{start}-{start + n}.pkl', 'wb')) 