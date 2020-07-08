import pandas as pd
from bs4 import BeautifulSoup


def data_in_fives(file):
    '''
    takes: scraped datafile - pickle of review card as extracted by beautifulsoup
    Data is scraped as it is loaded - in groups of 5 reviews.
    extracts: username; month of post; rating; flight details; title of
    review and the  review text itself
    returns list of extracted information
    '''
    all_data_in_fives = []
    for n, line in enumerate(file):
        line = str(line)
        soup = BeautifulSoup(line, 'lxml')

        who_when_list = soup.find_all('div', {'class':"_2fxQ4TOx"}, limit=5)
        who = []
        when = []
        try:
            for who_when in who_when_list:

                who_when_split = who_when.text.split(' wrote a review ')
                who.append(who_when_split[0])
                when.append(who_when_split[1])
        except:
            print(n, 'who_when')
            pass

        ratings_list = soup.find_all('div', {'data-test-target': 'review-rating'})
        ratings = []
        try:
            for rating in ratings_list:
                rate = str(rating)
                ratings.append(rate[-17])
        except:

            print(n, 'ratings')
            pass



        flight_type_list = soup.find_all('div',{'class': 'hpZJCN7D'})
        
        for flight_type in flight_type_list:
            soup = BeautifulSoup(str(flight_type), 'lxml')
            three_elements = soup.find_all('div',{'class': '_3tp-5a1G'})
            for element in three_elements:
                flight = three_elements[0].text
                flight_type = three_elements[1].text
                seat_type = three_elements[2].text
            print (flight, '\t', flight_type, '\t', seat_type)




        flight_list = soup.find_all(  'div', {'class': 'hpZJCN7D'})
        flights = []
        try:
            for flight_type in flight_type_list:
                soup = BeautifulSoup(str(flight_type), 'lxml')
                three_elements = soup.find_all('div',{'class': '_3tp-5a1G'})
                flight = [element.text for element in three_elements]
                flights.append(', '.join(flight))
        except:
            print(n, 'flights')
            pass

        title_list = soup.find_all('div',{'data-test-target': 'review-title'})
        titles = []
        try:
            for title in title_list:
                titles.append(title.text)
        except:
            print(n, 'titles')
            pass

        text_list = soup.find_all('q',{'IRsGHoPm'})
        try:
            texts = []

            for text in text_list:
                texts.append(text.text)
        except:
            print(n, 'texts')
            pass

        all_data = zip(who, when, ratings, final_flights, titles, texts)
        all_data_in_fives.append(all_data)
    return all_data_in_fives


def dict_builder(all_data_in_fives):
    '''
    takes list of 5 review details; constructs 1 large dictionary effectively
    concatenating the data ready for dataframe construction
    '''
    keys = ['reviewer', 'review_date', 'rating', 'flight', 'title', 'text']
    final_dict = {}
    n = 0
    for all_data in all_data_in_fives:
        for items in all_data:
            final_dict[n] = dict(zip(keys, items))
            n += 1

    return final_dict


def dataframe_constructor(dict_out):
    '''
    takes dictionary of reviews and metadata and creates DataFrame
    duplicates are dropped.
    returns dataframe.
    '''
    df = pd.DataFrame(dict_out)
    df = df.T
    df.drop_duplicates(subset='text', inplace=True)
    return df