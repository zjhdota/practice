# -*- encoding:utf-8 -*-
def consumer():
    r = ''
    while True:
        n = yield r
        if not n:
            return
        print('[consumer] consuming {}'.format(n))
        r = '200 OK'

def produce(c):
    c.send(None)
    n = 0
    while n < 5:
        n = n + 1
        print('[produce] producting {}'.format(n))
        r = c.send(n)
        print('[produce] producting {}'.format(r))
    c.close()

c = consumer()
produce(c)
