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
import ray
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


### Date and Author: 20230731, Kyungwon Kim ###
### 여럿 user-agent 목록
### Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.101 Safari/537.36
### Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0
### Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:46.0) Gecko/20100101 Firefox/46.0
### Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.101 Safari/537.36
headers = {
    'authority': 'apis.naver.com',
    'accept': '*/*',
    'accept-language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    # Requests sorts cookies= alphabetically
    # 'cookie': 'ab.storage.userId.7af503ae-0c84-478f-98b0-ecfff5d67750=%7B%22g%22%3A%22browser-1625985144309-6%22%2C%22c%22%3A1626101500089%2C%22l%22%3A1626101500089%7D; ab.storage.deviceId.7af503ae-0c84-478f-98b0-ecfff5d67750=%7B%22g%22%3A%224cbe130c-6edd-d4aa-a78d-290b003c3592%22%2C%22c%22%3A1626101500094%2C%22l%22%3A1626101500094%7D; ASID=7992e0220000017aaa36664e0000004e; _ga=GA1.2.612969395.1626832328; ab.storage.sessionId.7af503ae-0c84-478f-98b0-ecfff5d67750=%7B%22g%22%3A%2228148006-e01d-7623-b7d1-b4fff0f59b4e%22%2C%22e%22%3A1627919390179%2C%22c%22%3A1627908091281%2C%22l%22%3A1627917590179%7D; MM_NEW=1; NFS=2; NNB=RDIIILNX6JCWE; nx_ssl=2; nid_inf=1665554565; NID_AUT=tP3V5ox533EjyAgkJ1JaqWEnPOhXs2hr3teD39pK972fuXqDWQZXoIOMzICJpa1A; NID_JKL=d393brIzilbjw+7TVvG0OW6Eo22+WIhQAfihItUdgbY=; _naver_usersession_=SPdJTrlTMrn8Udkyn58eo6HL; NID_SES=AAABwJaKJ5FjUAETXL8SAX2HKMUSTt3l8pPu49OSzbGzgKEEMN/ckpP4DbQVHQwTV1hVPWtbpP7Nomg0CbD8TtCpyOYbeq8+OpHb5eWbDsXXCeLHO4epgthLtbQHiBE8spXqEtx/h0D6MzxsIlN4pa8gz51jV+oWzQQNnpQCeaKKLaxcpMfhGXnZv4BK1Rg+TAgUFE9RtExcKjteTL2hB9tKT41C7antdQdhLfVXWUbsJ/q5b62iDZnnZUAANXHnWp/9RI2YyKSn70SVu4Bag+fxA/23OqjCHSbK5RMiNOQKV+Bs7uugaAsMKkH6lGBBIbNDkTXGZ4n1+KbqFwe1kV9oCaPJ+siwXESEqvY0jaLVNAqUATQZjnIMFIYwARw41FTuduxW1IOF7MdP7R3EqOvnqNir2lXW1UfRlHlOtMC4w/tXk8xqJR/HVlZrnltKkMZB5zfyDNvnt02jbOKJcORjmOeVvL+xoCdSXwZclfJzRkC31l43+9jSu4X8RPUfuJILRMHf2e1A0NU7Mwds7h+S//5AD0yUJlPtFFzLvriuD1SMTRXiSwN4pNWBi6UIsPzScRpyLMc8hUE8Bi8jJtGk4e0=; NDARK=N; page_uid=hrKUflprvN8ssNc4Muwssssss3R-382317; BMR=',
    'referer': 'https://n.news.naver.com/article/028/0002595736',
    'sec-ch-ua': '"Whale";v="3", " Not;A Brand";v="99", "Chromium";v="102"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'script',
    'sec-fetch-mode': 'no-cors',
    'sec-fetch-site': 'same-site',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
}

### 페이지 설정에 따른 URL 추출
def get_urls_from_navernews_bypage(search_query, start_pg, end_pg, sort=0):
    if start_pg == end_pg:
        start_page = str((start_pg-1) * 10 + 1)
        url = f"https://search.naver.com/search.naver?where=news&ie=utf8&sm=tab_pge&query={search_query}&sort={sort}&start={start_page}"
        return url
    else:
        urls = []
        for i in range(start_pg, end_pg + 1):
            page = str((i-1) * 10 + 1)
            url = f"https://search.naver.com/search.naver?where=news&ie=utf8&sm=tab_pge&query={search_query}&sort={sort}&start={page}"
            urls.append(url)
        return urls   
    
### 날짜 설정에 따른 URL 추출
def get_urls_from_navernews_bydate(search_query, start_date, end_date, sort=0, maxpage=1000):
    i, urls = 1, []
    while True:
        page = str((i-1) * 10 + 1)
        url = f"https://search.naver.com/search.naver?where=news&ie=utf8&sm=nws_hty&query={search_query}&sort={sort}&nso=so%3Ar%2Cp%3Afrom{start_date}to{end_date}&start={page}"
        urls.append(url)
        if i == maxpage+1:
            return urls
        else:
            i = i + 1
            
### 유효 URL 추출
def get_urls_from_navernews(search_query, start, end, sort=0, maxpage=1000, maxpage_count=True):
    # URL 불러오기
    if type(start) == int:
        url = get_urls_from_navernews_bypage(search_query, start, end, sort=sort)
    elif type(start) == str:
        url = get_urls_from_navernews_bydate(search_query, start, end, sort=sort, maxpage=maxpage)
        
    # 페이지수 counting에 따른 url 갯수 조정
    if maxpage_count:
        for pg in tqdm([url[idx] for idx in range(0, maxpage+1, 10)]):
            response = requests.get(pg, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            sleep(random.uniform(2, 4))
            maxpage_numbers = []
            for link in soup.select('.sc_page_inner > a'):
                maxpage_number = int(link.text)
                maxpage_numbers.append(maxpage_number)
            if maxpage_numbers != []: 
                maxpage_final = max(maxpage_numbers) 
            else: 
                break
        url = url[:maxpage_final]  
        print('Total Pages: ', len(url))
    
    return url

### URL에 담긴 댓글 추출
def get_comments_from_navernews(url):
    # setting
    url = url.split('?')[0]
    oid_1, oid_2 = url.split('/')[-1], url.split('/')[-2]

    i, comments = 1, [] #모든 댓글을 담는 리스트
    while True:
        params = {
            'ticket': 'news',
            'templateId': 'default_society',
            'pool': 'cbox5',
            'lang': 'ko',
            'country': 'KR',
            'objectId': f'news{oid_2},{oid_1}',
            'pageSize': '100',
            'indexSize': '10',
            'page': str(i),
            'currentPage': '0',
            'moreParam.direction': 'next',
            'moreParam.prev': '10000o90000op06guicil48ars',
            'moreParam.next': '1000050000305guog893h1re',
            'followSize': '100',
            'includeAllStatus': 'true',
        }

        response = requests.get('https://apis.naver.com/commentBox/cbox/web_naver_list_jsonp.json', params=params, headers=headers)
        response.encoding = "UTF-8-sig"
        res = response.text.replace("_callback(","")[:-2]
        temp=json.loads(res)
        try :
            comment = list(pd.DataFrame(temp['result']['commentList'])['contents'])
            for j in range(len(comment)):
                comments.append(comment[j])
            else:
                i+=1
        except :
            break

    return comments

### 시간표시 24H 기준으로 형식 변환
def get_update_timeformat(time_origin):
    try:
        # 영문화
        if '오전' in time_origin:
            time_origin = time_origin.replace('오전', 'AM')
        elif '오후' in time_origin:
            time_origin = time_origin.replace('오후', 'PM')
        # 형식 변환
        time_final = datetime.datetime.strptime(time_origin, '%Y.%m.%d. %p %I:%M')
        time_final = time_final.strftime('%Y-%m-%d %H:%M:%S')
        
        return time_final
    
    except ValueError:
        
        return time_origin  # 날짜 형식이 아닌 경우 그대로 반환

### URL에 담긴 댓글을 포함한 뉴스정보 추출
def get_navernews(search_query, start, end, sort=0, maxpage=1000, maxpage_count=False, save_local=False):
    # URL 불러오기
    url = get_urls_from_navernews(search_query, start, end, sort=sort, maxpage=maxpage, maxpage_count=maxpage_count)  
    
    # 개별 URL에 따른 데이터 추출 
    time_start = datetime.datetime.now()
    df_news = []
    for pg in url:
        response = requests.get(pg, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        sleep(random.uniform(2, 4)) # 요청 사이 무작위로 시간 간격을 두어(2~4초) 일정한 시간간격의 비정상적인 접근 방지
        
        # 테스트
        news_elements = soup.select('div.news_wrap.api_ani_send')
        article_elements = soup.select("div.group_news > ul.list_news > li div.news_area > div.news_info > div.info_group > a.info")
        for element, article in zip(news_elements, article_elements):
            ## 기사시간
            time = element.select_one('span.info').text.strip()
            if "news.naver.com" in article.attrs['href']:  
                article_response = requests.get(article.attrs['href'], headers=headers, verify=False)
                article_soup = BeautifulSoup(article_response.text, 'html.parser')
                try:
                    time_html = article_soup.select_one("div#ct> div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > div > span")
                    time = time_html.attrs['data-date-time']
                    time = get_update_timeformat(time)
                except AttributeError:
                    time = article_soup.select_one("#content > div.end_ct > div > div.article_info > span > em")
                    time = re.sub(pattern='<[^>]*>',repl='',string=str(time))
                    time = get_update_timeformat(time)
            ## 언론사
            try: 
                press = element.select_one('a.info.press').text.strip()
            except:
                press = None
            ## 카테고리
            if "news.naver.com" in article.attrs['href']:  
                category = article_soup.select_one('#_LNB > ul > li.Nlist_item._LNB_ITEM.is_active > a > span')
                if category != None: 
                    category = str(category).split('menu">')[1].split('</span>')[0]
            else:
                category = None
            ## 제목
            title = element.select_one('a.news_tit').text.strip()
            title = re.sub(pattern='<[^>]*>', repl='', string=str(title))
            ## 본문
            if "news.naver.com" in article.attrs['href']: 
                content = None
                content = article_soup.select("article#dic_area")    # "div#dic_area"에서 "article#dic_area"로 변경
                if content == None:
                    content = article_soup.select("#articeBody")
                content = ''.join(str(content))
                content = re.sub(pattern='<[^>]*>', repl='', string=content)
                content = content.replace("""[\n\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}""", '')
                content = content.replace('\n', ' ').replace('\t', ' ')
            else:
                content = None
            ## 댓글
            if "news.naver.com" in article.attrs['href']: 
                comment = get_comments_from_navernews(article.attrs['href'])
            else:
                comment = None            
            ## 한줄씩 붙이기
            df_news.append([time, press, category, title, content, comment, element.select_one('a.news_tit')['href'], article.attrs['href']])
        
    # 정리
    df_news = pd.DataFrame(df_news, columns=['Date', 'Press', 'Category', 'Title', 'Content', 'Comment', 'URL_Origin', 'URL_Naver'])
    time_end = datetime.datetime.now()
#     print('News Info Extracting Time: ', time_end-time_start)
#     print('Size of News Data: ', df_news.shape[0])
    
    # 저장
    if save_local:
        folder_location = os.path.join(os.getcwd(), 'Data', search_query, '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        datetime_info = df_news.Date[df_news.Date.apply(lambda x: len(x[:10]) == 10)]
        try:
            save_name = 'NaverNews_{}-{}_KK.csv'.format(datetime_info.min()[:10], datetime_info.max()[:10])
        except:
            save_name = 'NaverNews_{}-{}_KK.csv'.format(start, end)
        df_news.to_csv(os.path.join(folder_location, save_name), index=False, encoding='utf-8-sig')
    
    return df_news

### 입력 날짜 범위를 월별로 나누어서 시작과 종료 문자 생성
def date_generator(start, end):
    date_list = []
    year_list = list(range(int(start[:4]), int(end[:4])+1))
    # 연도가 시작과 종료가 같은 경우
    if len(year_list) == 1:
        year = year_list[0]
        month_list = list(range(int(start[4:6]), int(end[4:6])+1))
        for idx, month in enumerate(month_list):
            if idx == 0 and month == int(start[4:6]):
                day_start, day_end = start[6:], '31'
            elif idx == len(month_list)-1 and month == int(end[4:6]):
                day_start, day_end = '01', end[6:]
            else:
                day_start, day_end = '01', '31'
                
            # 정리
            if len(str(month)) == 1:
                start_date = str(year) + '0' + str(month) + day_start
                end_date = str(year) + '0' + str(month) + day_end
            else:
                start_date = str(year) + str(month) + day_start
                end_date = str(year) + str(month) + day_end
            
            date_list.append([start_date, end_date])
                
    # 연도의 시작과 종료가 다른 경우
    else:
        for idx, year in enumerate(year_list):
            # month
            if idx == 0:
                month_list = list(range(int(start[4:6]), 12+1))
            elif idx == len(year_list)-1:
                month_list = list(range(1, int(end[4:6])+1))
            else:
                month_list = list(range(1, 12+1))

            # day
            for month in month_list:
                if idx == 0 and month == int(start[4:6]):
                    day_start, day_end = start[6:], '31'
                elif idx == len(year_list)-1 and month == int(end[4:6]):
                    day_start, day_end = '01', end[6:]
                else:
                    day_start, day_end = '01', '31'

                # 정리
                if len(str(month)) == 1:
                    start_date = str(year) + '0' + str(month) + day_start
                    end_date = str(year) + '0' + str(month) + day_end
                else:
                    start_date = str(year) + str(month) + day_start
                    end_date = str(year) + str(month) + day_end

                date_list.append([start_date, end_date])
        
    return date_list

# 각 월별 뉴스정보 추출 후 모두 결합
def get_data_from_navernews(search_query, start, end, sort=0,
                            maxpage=1000, maxpage_count=False, save_local=False):
    # 날짜 생성
    time_start = datetime.datetime.now()
    date_list = date_generator(start, end)
    
    # 데이터 수집
    df_news = pd.DataFrame()
    for period in tqdm(date_list):
        # 각 월별 데이터 수집
        df = get_navernews(search_query=search_query, start=period[0], end=period[1], sort=sort, 
                           maxpage=maxpage, maxpage_count=maxpage_count, save_local=save_local)
        
        # 모든 데이터 결합
        if df.shape[0] != 0:
            df_news = pd.concat([df_news, df], axis=0)
            
    # 저장
    if save_local:
        folder_location = os.path.join(os.getcwd(), 'Data', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        datetime_info = df_news.Date[df_news.Date.apply(lambda x: len(x[:10]) == 10)]
        save_name = 'NaverNews_{}_{}-{}_KK.csv'.format(search_query, datetime_info.min()[:10], datetime_info.max()[:10])
        df_news.to_csv(os.path.join(folder_location, save_name), index=False, encoding='utf-8-sig')
    time_end = datetime.datetime.now()
    print('News Info Extracting Time: ', time_end-time_start)
    
    return df_news

# 병렬처리
@ray.remote
def get_data_from_navernewsParallel(search_query, start, end, sort=0,
                                    maxpage=1000, maxpage_count=False, save_local=False):
    # 날짜 생성
    time_start = datetime.datetime.now()
    date_list = date_generator(start, end)
    
    # 데이터 수집
    df_news = pd.DataFrame()
    for period in tqdm(date_list):
        # 각 월별 데이터 수집
        df = get_navernews(search_query=search_query, start=period[0], end=period[1], sort=sort, 
                           maxpage=maxpage, maxpage_count=maxpage_count, save_local=save_local)
        
        # 모든 데이터 결합
        if df.shape[0] != 0:
            df_news = pd.concat([df_news, df], axis=0)
           
    # 저장
    if save_local:
        folder_location = os.path.join(os.getcwd(), 'Data', 'Naver', '')
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)
        datetime_info = df_news.Date[df_news.Date.apply(lambda x: len(x[:10]) == 10)]
        save_name = 'NaverNews_{}_{}-{}_KK.csv'.format(search_query, datetime_info.min()[:10], datetime_info.max()[:10])
        df_news.to_csv(os.path.join(folder_location, save_name), index=False, encoding='utf-8-sig')
    time_end = datetime.datetime.now()
    print('News Info Extracting Time: ', time_end-time_start)
    
    return df_news