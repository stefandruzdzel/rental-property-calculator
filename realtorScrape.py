# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:23:00 2022

@author: Stefa
"""

from bs4 import BeautifulSoup
import requests

url = r'https://www.realtor.com/realestateandhomes-detail/M4623205196'
# url = r'https://www.trulia.com/p/pa/homestead/1911-mcclure-st-homestead-pa-15120--2013931957'
# url = r'https://www.zillow.com/homedetails/1911-McClure-St-Homestead-PA-15120/11293941_zpid/?view=public'
# url = r'https://www.redfin.com/PA/Homestead/1911-McClure-St-15120/home/74481122'
# url = r'https://www.homes.com/property/1911-mcclure-st-homestead-pa-15120/id-400022651471/'



session = requests.Session()
session.auth = ('user', 'pass')
session.headers.update({'x-test': 'true'})
session.get('https://httpbin.org/headers', headers={'x-test2': 'true'})

response = session.get('https://httpbin.org/cookies', cookies={'from-my': 'browser'})
print(response .text)

response = session.get('https://httpbin.org/cookies')
print(response.text)


headers = requests.utils.default_headers()
headers.update({
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
})

result = requests.get(url, headers = headers)
doc = BeautifulSoup(result.text, "html.parser")
tax = doc.find_all(text="Tax")
print(tax)
for i in tax:
    parent = i.parent
    print(parent)



"""
Purchase Price (optional)
Property Tax
Home Insurance


Other sites:
Monthly Rent

interest rate

"""