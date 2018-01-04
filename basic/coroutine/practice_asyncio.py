# -*- encoding:utf-8 -*-
import datetime
import asyncio

@asyncio.coroutine
def hello():
    print('hello,world!')
    r = yield from asyncio.sleep(1)
    print('hello again')

loop = asyncio.get_event_loop()
loop.run_until_complete(hello())


async def do():
    print('waiting: {}'.format(2))

start = datetime.datetime.now()
loop = asyncio.get_event_loop()
loop.run_until_complete(do())
loop.close()
print(datetime.datetime.now()-start)
