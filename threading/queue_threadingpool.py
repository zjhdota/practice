# -*- encoding:utf-8 -*-
from queue import Queue
import threading
import os

class CustomThread(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.__queue = queue

    def run(self):
        while True:
            cmd = self.__queue.get()
            cmd()
            self.__queue.task_done()

lock = threading.Lock()

def func():
    lock.acquire()
    print('this is a thread {}'.format(threading.current_thread().name))
    lock.release()

def custom_pool():
    print('this is a thread {}'.format(threading.current_thread().name))
    queue = Queue(5)
    for i in range(queue.maxsize):
        t = CustomThread(queue)
        t.setDaemon(True)
        t.start()

    for i in range(200):
        queue.put(func)
    queue.join()

    print('this is a thread {}'.format(threading.current_thread().name))

def main():
    custom_pool()


if __name__ == '__main__':
    main()
