import os, platform
from utils import get_links_to_scrape, start_scrape, trip_advisor_scrape

def main ():
    prompt = 'You can scrape en masse (indiscriminantly) or you can scrape select companies. Enter "i" for indiscriminate\
or "s" for select companies: '
    main_input = input(prompt)
    if main_input.lower() == 'i':
        main_url = 'https://www.tripadvisor.com/Airlines'
        prompt = "Add number of Trip Advisor pages of companies to scrape: (10 companies per page): "
        page_input = input(prompt)
        if page_input == '':
            page_input = 10
        else:
            page_input = int(page_input)

        
    else:
        prompt = 'enter url of airline you wish to scrape: '
        main_url = input(prompt)
        
    prompt = "Add start page (default = 0): "
    start_input = input (prompt)
    if start_input == '':
        start_input = 0
    else:
        start_input = int(start_input)

    prompt = "Enter number of review pages you wish to scrape for each airline (5 reviews per page): "
    num_reviews_input = input (prompt)

    prompt = "if scrape has crashed, change parameter (start) to page number where crash happened to restart from\
that point otherwise enter 0): "
    start = input(prompt)
    
    if main_input.lower() == 'i':
        urls = get_links_to_scrape(main_url, start=start_input, num_pages=page_input)
    else:
        urls = [main_url]


    for url in urls:
        driver = start_scrape(url, start=int(start_input))
        trip_advisor_scrape(driver, url, start=int(start), num_reviews=int(num_reviews_input))
        driver.quit()
        
if __name__ == "__main__":
    
    if platform.system() == "Linux" or platform.system() == "Linux2" or platform.system() == "Darwin":
        prompt = "Enter path to geckodriver (default is current working directory): "
        path_input = input(prompt)
        if path_input == '':
            path = os.getcwd()
            print('\n', 'Gecko driver is expected to be at: ', path, '\n')
        else:
            path = path_input
        os.environ["PATH"] += os.pathsep + path
    if not os.path.exists('./airline_pickles'):
        os.mkdir('airline_pickles')
    main()