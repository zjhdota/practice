# -*- encoding: utf-8 -*-
from multiprocessing import Process
from multiprocessing import Pool
import os

def func():
    print('this is a process {}'.format(os.getpid()))

def process():
    pool = Pool(processes=3)
    for i in range(1000):
        pool.apply_async(func)
    pool.close()
    pool.join()

def main():
    process()

if __name__ == '__main__':
    main()
