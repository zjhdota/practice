# -*- encoding: utf-8 -*-
from multiprocessing import Process,Pool,Queue

def func1(q):
    q.put('1')

def func2(q):
    print(q.get())

if __name__ == '__main__':

    q = Queue()
    p = Pool()

    p1 = Process(target=func1, args=(q,))
    p2 = Process(target=func2, args=(q,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
