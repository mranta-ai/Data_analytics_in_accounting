## Processing unstructured data

### Extracting accounting data from different document types

#### JSON

JSON stands for JavaScript Object Notation. It is a popular format for transporting and storing data, especially in web-page management. Python has a built-in library for encoding/decoding JSON files. In the following are basic examples how to use it.

import json

The format of JSON is very similar to a Python dictionary.

json_example = '{ "Company":"Nokia", "Country":"Finland", "ROA":0.12}'

loads() turns JSONs to dictionaries.

result_dict = json.loads(json_example)

result_dict

result_dict['ROA']

dumps() can be used to change dictionaries to JSON objects.

pyth_dict = {"Company":"Apple", "Country":"USA", "ROA":0.17}

results_json = json.dumps(pyth_dict)

results_json # Notice the outer single quotation marks.

More specifically, dumps() will change Python objects into JSON objects with the following rules:
* Python dict to JSON object
* Python list to JSON array
* Python tuple to JSON array
* Python string to JSON string
* Python int to JSON number
* Python float to JSON number
* Python boolean to JSON boolean
* Python None to JSON null

Very often, data providers set up API services to connect applications to databases. Often, these API services will use XML or JSON formats to exchange data between the client and the server. Therefore, it is essential to know how to read these files.

With json-library, you can use the built-in open() function to open a json file and json.loads() to transfer it to a Python object.

fd = open('company_tickers.json')

comp_dict = json.loads(fd.read())

list(comp_dict.items())[:10]

Pandas has also functios to read JSON objects

import pandas as pd

json_df = pd.read_json('company_tickers.json')

json_df

json_df.transpose()

#### XML

https://docs.python.org/3/library/xml.etree.elementtree.html

pd.read_




#### XBLR

https://pypi.org/project/python-xbrl/

https://www.codeproject.com/Articles/1227268/Accessing-Financial-Reports-in-the-EDGAR-Database

https://www.codeproject.com/Articles/1227765/Parsing-XBRL-with-Python

from bs4 import BeautifulSoup
import requests
import sys

# Access page
cik = '0000051143'
type = '10-K'
dateb = '20160101'

# Obtain HTML for search page
base_url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={}&type={}&dateb={}"
edgar_resp = requests.get(base_url.format(cik, type, dateb))
edgar_str = edgar_resp.text

# Find the document link
doc_link = ''
soup = BeautifulSoup(edgar_str, 'html.parser')
table_tag = soup.find('table', class_='tableFile2')
rows = table_tag.find_all('tr')
for row in rows:
    cells = row.find_all('td')
    if len(cells) > 3:
        if '2015' in cells[3].text:
            doc_link = 'https://www.sec.gov' + cells[1].a['href']

# Exit if document link couldn't be found
if doc_link == '':
    print("Couldn't find the document link")
    sys.exit()

# Obtain HTML for document page
doc_resp = requests.get(doc_link)
doc_str = doc_resp.text

# Find the XBRL link
xbrl_link = ''
soup = BeautifulSoup(doc_str, 'html.parser')
table_tag = soup.find('table', class_='tableFile', summary='Data Files')
rows = table_tag.find_all('tr')
for row in rows:
    cells = row.find_all('td')
    if len(cells) > 3:
        if 'INS' in cells[3].text:
            xbrl_link = 'https://www.sec.gov' + cells[2].a['href']

# Obtain XBRL text from document
xbrl_resp = requests.get(xbrl_link)
xbrl_str = xbrl_resp.text

# Find and print stockholder's equity
soup = BeautifulSoup(xbrl_str, 'lxml')
tag_list = soup.find_all()
for tag in tag_list:
    if tag.name == 'us-gaap:stockholdersequity':
        print("Stockholder's equity: " + tag.text)

#### PDF

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
# From PDFInterpreter import both PDFResourceManager and PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
# Import this to raise exception whenever text extraction from PDF is not allowed
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.converter import PDFPageAggregator

def convert_pdfminer(fname):
        fp = open(fname, 'rb')
        parser = PDFParser(fp)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        text = ''
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
            layout = device.get_result()
            for lt_obj in layout:
                if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
                    text += lt_obj.get_text()
        return text

### Extracting data from the Internet

#### Social media

#### Structured data

test_df = pd.read_html('https://www.nordnet.fi/markkinakatsaus/osakekurssit?selectedTab=keyFigures&sortField=pe&sortOrder=asc&exchangeCountry=FI',decimal=',')

work_df = pd.DataFrame()

work_df['Name'] = test_df[0]['Nimi']

work_df['P/E']= [i for [i,j] in test_df[0]['P/S'].str.split(' ')]

work_df['P/S']= [i for [i,j] in test_df[0]['P/B'].str.split(' ')]

work_df['P/B']= [i for [i,j] in test_df[0]['Tulos/osake'].str.split(' ')]

temp_list = []
for value in test_df[0]['Osinkotuotto']:
    try:
        temp_list.append(value.split(' ')[0])
    except:
        temp_list.append(value)
        
work_df['Tulos/osake'] = temp_list

temp_list = []
for value in test_df[0]['Osinko/osake']:
    try:
        temp_list.append(value.split(' ')[0])
    except:
        temp_list.append(value)
        
work_df['Osinkotuotto'] = temp_list

temp_list = []
for value in test_df[0]['Unnamed: 11']:
    try:
        temp_list.append(value.split(' ')[0])
    except:
        temp_list.append(value)
        
work_df['Lainoitusaste'] = temp_list

work_df

work_df['P/E'] = work_df['P/E'].astype('float')

work_df['P/S'] = work_df['P/S'].astype('float')

work_df['P/B'] = work_df['P/B'].astype('float')

work_df['Tulos/osake'].replace('–',np.nan,inplace=True)

work_df['Tulos/osake'] = work_df['Tulos/osake'].astype('float')

work_df['Osinkotuotto'].replace('–',np.nan,inplace=True)

work_df['Osinkotuotto'] = work_df['Osinkotuotto'].astype('float')

work_df['Lainoitusaste'].replace('–',np.nan,inplace=True)

work_df['Lainoitusaste'] = work_df['Lainoitusaste'].astype('float')

work_df.hist(figsize=(10,10))
plt.show()

### Processing text data

#### Regular experssions

https://www.regular-expressions.info/quickstart.html

### Processing video and images

#### Satellite data

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

my_example_nc_file = 'S5P_NRTI_L2__NO2____20200309T105605_20200309T110105_12457_01_010302_20200309T114222.nc'
fh = Dataset(my_example_nc_file, mode='r')

print(fh.groups['PRODUCT'].variables['nitrogendioxide_tropospheric_column_precision'])

lons = fh.groups['PRODUCT'].variables['longitude'][:][0,:,:]
lats = fh.groups['PRODUCT'].variables['latitude'][:][0,:,:]
no2 = fh.groups['PRODUCT'].variables['nitrogendioxide_tropospheric_column_precision'][0,:,:]
print (lons.shape)
print (lats.shape)
print (no2.shape)

no2_units = fh.groups['PRODUCT'].variables['nitrogendioxide_tropospheric_column_precision'].units

no2_units

from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap

lon_0 = lons.mean()
lat_0 = lats.mean()

m = Basemap(width=3000000,height=2500000,
            resolution='l',projection='stere',lat_0=lat_0,lon_0=lon_0)

xi, yi = m(lons, lats)

plt.figure(figsize=(10,10))
cs = m.pcolor(xi,yi,np.squeeze(no2),norm=LogNorm(), cmap='jet')
m.drawparallels(np.arange(-80., 81., 10.), labels=[1,0,0,0], fontsize=10)
m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], fontsize=10)

# Add Coastlines, States, and Country Boundaries
m.drawcoastlines(linewidth=2.5)
m.drawcountries(linewidth=2.5)
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label(no2_units)
plt.savefig('test.png')

plt.figure(figsize=(10,10))
plt.imshow(no2,cmap='hot')

### Speech recognition and synthesis

### Feature engineering

