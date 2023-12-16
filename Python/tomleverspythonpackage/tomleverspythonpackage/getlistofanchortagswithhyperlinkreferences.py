# Gets list of anchor tags with hyperlink references given a URL
# Usage:
# python .\getlistofanchortagswithhyperlinkreferences.py https://www.google.com
# from tomleverspythonpackage.getlistofanchortagswithhyperlinkreferences import get_list_of_anchor_tags_with_hyperlink_references; get_list_of_anchor_tags_with_hyperlink_references('https://www.google.com')

import argparse
from bs4 import BeautifulSoup
import requests

def get_list_of_anchor_tags_with_hyperlink_references(URL):
    session = requests.Session()
    response = session.get(URL)
    text = response.text
    beautiful_soup = BeautifulSoup(text, 'html')
    list_of_anchor_tags = beautiful_soup.find_all('a', href = True)
    for anchor_tag in list_of_anchor_tags:
        print(anchor_tag['href'])

def parse_arguments():
    dictionary_of_arguments = {}
    parser = argparse.ArgumentParser(prog = 'Get List Of Anchor Tags With Hyperlink References', description = 'This program gets a list of anchor tags with hyperlink references.')
    parser.add_argument('URL', help = 'URL')
    args = parser.parse_args()
    URL = args.URL
    print(f'URL: {URL}')
    dictionary_of_arguments['URL'] = URL
    return dictionary_of_arguments

if __name__ == "__main__":
    dictionary_of_arguments = parse_arguments()
    get_list_of_anchor_tags_with_hyperlink_references(dictionary_of_arguments['URL'])