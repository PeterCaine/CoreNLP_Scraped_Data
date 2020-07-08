# Trip-Advisor_Scrape
Scraping Trip Advisor using Python and Selenium

## Install

- download and install selenium (https://selenium-python.readthedocs.io/installation.html or pip install selenium)
- download and install geckodriver (for Firefox): https://github.com/mozilla/geckodriver/releases (this can be placed directly into the folder where you run your .py files from (or ipynb notebook, or added to path))

## Prompts
Scraping Trip Advisor requires a few decisions to be made on the part of the user:
- how many airlines; 
- how many reviews per airline. 
It also takes a while, so if you want to do a little this time and wish to continue where you left off last time, that adds more options.

At the beginning of the programme you are prompted to:
- enter the path to geckodriver (Linux/ Mac only - windows users should put driver in same directory as .py files otherwise add to PATH)
- add the number of Trip Advisor pages of companies to scrape (5 companies per page)
- add the start page (if you have previously scraped 10 pages, enter 11 to continue where you left off)
- enter number of reviews per company (experimentation showed that around 1500 was possible without connection issues - max values are prone to errors midway. The pickle files will still write out, but it's still annoying).
- at what point do you wish to start (the pickle files are written with the name of the airline and number of pages of reviews scraped (e.g. Adria-Airways-No-Longer-Operating_0-4.pkl)-you can continue by entering the next number if there are any more reviews to scrape).

## Output

- a pickle file is output containing a binary version of the raw html of each review. file "sample_pickle_extract.py" is an example of how to extract to dictionary info about:
    - reviewer, 
    - review date, 
    - overall rating, 
    - flight, 
    - title of review, 
    - text of review. 
- a txt file of the companies scraped is also output to the airline_pickles folder.
- large files (i.e. large numbers of scraped reviews) take a while to write out. Patience is required.

