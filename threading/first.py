# -*- encoding:utf-8 -*-
import threading

def func():
    print('this is a thread {}'.format(threading.current_thread().name))

def start():
    t = threading.Thread(target=func)
    t.start()
    t.join()

def main():
    start()

if __name__ == '__main__':
        main()



