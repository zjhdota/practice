# -*- encoding: utf-8 -*-

import threading

def func(n):
    for i in range(100):
        print(str(i) + threading.current_thread().name)

t1 = threading.Thread(target=func, args=('1',))
t2 = threading.Thread(target=func, args=('2',))

threads = []
threads.append(t1)
threads.append(t2)

if __name__ == '__main__':
    for t in threads:
        t.setDaemon(True)
        t.start()
    t.join()
    print('this is a thread {}'.format(threading.current_thread().name))
