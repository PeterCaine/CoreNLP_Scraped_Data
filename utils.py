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

    #     print(n)
        line = str(line)
        soup = BeautifulSoup(line, 'lxml')

        who_when_list = soup.find_all('div',{'class':"_2fxQ4TOx"}, limit = 5)
        who = []
        when = []
        try:
            for who_when in who_when_list:
                who_when_split = who_when.text.split(' wrote a review ')
                who.append(who_when_split[0])
                when.append(who_when_split[1])
        except:
            print(n)
            pass

        ratings_list = soup.find_all('div', {'data-test-target':'review-rating'})
        ratings = []
        try:
            for rating in ratings_list:
                rate = str(rating)
                ratings.append(rate[-17])
        except:
            print(n, 'ratings')
            pass

        flight_type_list = soup.find_all('div',{'class': 'hpZJCN7D'})
        flights = []
        try:
            for flight_type in flight_type_list:
                flight_soup = BeautifulSoup(str(flight_type), 'lxml')
                three_elements = flight_soup.find_all('div',{'class': '_3tp-5a1G'})
                flight = [element.text for element in three_elements]
                flights.append (', '.join(flight))
        except:
            print(n, 'flights')
            pass


        title_list = soup.find_all('div',{'data-test-target': 'review-title'})
        titles = []
        for title in title_list:
            titles.append(title.text)

        text_list = soup.find_all('q', {'class':'IRsGHoPm'})
        texts = []
        try: 
            for text in text_list:
                texts.append(text.text)
        except:
            print(n, 'texts')
            pass

        all_data = zip(who, when, ratings, flights, titles, texts)
        all_data_in_fives.append(all_data)
    return all_data_in_fives


def dict_builder(all_data_in_fives):
    '''
    takes list of 5 review details; constructs 1 large dictionary effectively
    concatenating the data ready for datafframe construction
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


def stanford_to_csv(stanford_pp):
    '''
    takes stanford output file and returns a list of nested lists of stanford
    output per token
    '''
    processed_sentence_list = []
    for doc in stanford_pp:
        # this i+1 is review number - not sentence number
        output = [[i+1, word.id, word.text, word.lemma, word.upos, word.xpos,
                   word.head, word.deprel] for i, sent in enumerate(doc.sentences) for word in sent.words]
        processed_sentence_list.append(output)
    return processed_sentence_list


def add_pre_processed_col(stanford_pp, dataframe, basename):
    '''
    takes a dataframe of airline reviews and a pickled stanford doc of the text
    column
    returns modified dataframe with extra columns in which the text has been
    lemmatised, and reference to the airline and digits have been reduced and
    another column (no_NER) in which all NER labels have attempted to be removed

    '''
    stanford_ner = []
    stanford_lemmatized = []
    stanford_pos = []

    for doc in stanford_pp:
        lemma_list = [word.lemma.lower()
                      for sent in doc.sentences for word in sent.words
                      if word.upos != 'PUNCT']
        stanford_lemmatized.append(' '.join(lemma_list))
        ner_list = [(ent.text, ent.type) for sent in doc.sentences
                    for ent in sent.ents]
        stanford_ner.append(ner_list)
        pos_list = [word.upos for sent in doc.sentences for word in sent.words
                    if word.upos != 'PUNCT']
        stanford_pos.append(pos_list)

    dataframe['stanford_lemma'] = (stanford_lemmatized)
    
    return dataframe