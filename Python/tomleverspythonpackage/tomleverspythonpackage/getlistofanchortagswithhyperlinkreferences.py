from bs4 import BeautifulSoup
import requests
import sys

def get_list_of_anchor_tags_with_hyperlink_references(URL, username, password):
    session = requests.Session()
    session.auth = (username, password)
    auth = session.post('https://www.doi.org')
    response = session.get(URL)
    text = response.text
    beautiful_soup = BeautifulSoup(text, 'html')
    list_of_anchor_tags = beautiful_soup.find_all('a', href = True)
    for anchor_tag in list_of_anchor_tags:
        print(anchor_tag['href'])

if __name__ == "__main__":
    URL = sys.argv[1]
    username = sys.argv[2]
    password = sys.argv[3]
    get_list_of_anchor_tags_with_hyperlink_references(URL)