# -*- encoding:utf-8 -*-
from concurrent.futures import ThreadPoolExecutor


def func():
    print('this is a thread{}'.format())

def start():
    pool = ThreadPoolExecutor(2)
    for i in range(100):
        pool.submit(func)

def main():
    start()

if __name__ == '__main__':
    main()
