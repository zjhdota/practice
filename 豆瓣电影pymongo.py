# -*- encoding:utf-8 -*-
import requests
import pymongo
import json

connection = pymongo.MongoClient(host="127.0.0.1",port=27017)
douban = connection['douban']
moviedate =douban['movie']

url = 'https://movie.douban.com/j/search_subjects?type=movie&tag=%E8%B1%86%E7%93%A3%E9%AB%98%E5%88%86&sort=recommend&page_limit=200000000000&page_start=0'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
}
html = requests.get(url, headers=headers)
date = json.loads(html.text)
for dit in date['subjects']:
    rate = dit['rate']
    title = dit['title']
    url = dit['url']
    cover = dit['cover']
    movie = {
        'rate': rate,
        'title': title,
        'url': url,
        'cover': cover,
    }
    moviedate.insert(movie)

