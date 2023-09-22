# Ignore the warnings
import warnings
# warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import os
import pickle as pk
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import datetime
from time import sleep
import matplotlib.pyplot as plt
import statsmodels.api as sm

#크롤링
import csv, json
from bs4 import BeautifulSoup
import requests
import aiohttp
import asyncio
import nest_asyncio
nest_asyncio.apply()
from emoji import core
import re
import sys


def get_contents_from_naverblogs(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    content = (soup.find('div', 'se-main-container') or
               soup.select_one('#postViewArea') or
               soup.select_one('div.se_component_wrap.sect_dsc.__se_component_area > div.se_component.se_paragraph.default > div > div > div > div > div > p'))

    if content:
        content_text = content.get_text(strip=True)
        content_text = core.replace_emoji(content_text, replace="")
        url_pattern = r"(http|https)?:\/\/[a-zA-Z0-9-\.]+\.[a-z]{2,}(\S*)?|www\.[a-zA-Z0-9-]+\.[a-z]{2,}(\S*)?|[a-zA-Z0-9-]+\.[a-z]{2,}(\S*)?|([a-zA-Z0-9-]+\.)?naver.com(\S*)?"
        content_text = re.sub(url_pattern, "", content_text)
        content_text = re.sub(r"\u200C|\u200b", "", content_text)
        content_text = ' '.join(content_text.split())
        return content_text
    
    return None

def parse_blog_data(soup, result_data):
    title_and_url_elems = soup.find_all("a", {"class": "api_txt_lines"})
    desc_elems = soup.find_all("div", {"class": "api_txt_lines"})
    date_elems = soup.find_all("span", {"class": "sub_time"})

    for i in range(len(title_and_url_elems)):
        title = title_and_url_elems[i].text
        url = title_and_url_elems[i]['href']
        desc = desc_elems[i].text
        date = date_elems[i].text
        result_data.append({'url': url, 'title': title, 'description': desc, 'date': date})
        
async def fetch_and_parse_blog_data(_url, data):
    async with aiohttp.ClientSession() as session:
        async with session.get(_url) as response:
            text = await response.text()
            soup = BeautifulSoup(text.replace("\\", ""), "html.parser")
            parse_blog_data(soup, data)

async def get_data_from_naverblogs(search_query, start, end, maxpage=1000, save_local=False):
    df_blogs = []
    urls = [f'https://s.search.naver.com/p/blog/search.naver?where=blog&sm=tab_pge&api_type=1&query={search_query}&rev=44&start={i * 30}&dup_remove=1&post_blogurl=&post_blogurl_without=&nso=so:dd,p:from{end}to{start}&nlu_query=r_category:29+27&dkey=0&source_query=&nx_search_query={search_query}&spq=0&_callback=viewMoreContents'
            for i in range(1, maxpage+1)]
    print(urls)
    
    # 개별 URL에 따른 데이터 추출 
    time_start = datetime.datetime.now()
    await asyncio.gather(*(fetch_and_parse_blog_data(urls[i], df_blogs) for i in range(len(urls))))
    
    # 정리
    df_blogs = pd.DataFrame(df_blogs)
    df_blogs.columns = ['URL', 'Title', 'Content_Short', 'Date']
    df_blogs = df_blogs[['Date', 'Title', 'Content_Short', 'URL']]
    
    # 본문 내용 추가                
    contents = []
    for idx in tqdm(range(df_blogs.shape[0])):
        url = df_blogs.iloc[idx]['URL']
        if url.startswith("https://blog.naver.com/"):
            frameaddr = "https://blog.naver.com/" + BeautifulSoup(requests.get(url).text, 'html.parser').find('iframe', id="mainFrame")['src']
            content_text = get_contents_from_naverblogs(frameaddr)
            contents.append(content_text)
        else:
            contents.append([])
    
    # 정리
    df_blogs = pd.concat([df_blogs, pd.DataFrame(contents, columns=['Content'])], axis=1)
    df_blogs = df_blogs[['Date', 'Title', 'Content_Short', 'Content', 'URL']]
    time_end = datetime.datetime.now()
    print('Blogs Info Extracting Time: ', time_end-time_start)
    print('Size of Blogs Data: ', df_blogs.shape[0])
    
    # 저장
    if save_local:
        folder_location = os.path.join(os.getcwd(), 'Data', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        save_name = 'NaverBlogs_{}_{}-{}_KK.csv'.format(search_query, df_blogs.Date.min()[:10], df_blogs.Date.max()[:10])
        df_blogs.to_csv(os.path.join(folder_location, save_name), index=False, encoding='utf-8-sig')

    return df_blogs
