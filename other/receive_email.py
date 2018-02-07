#!/usr/bin/env python3
import os
import shutil
import poplib
import logging
from email.parser import Parser
from email.header import decode_header
from email.utils import parseaddr

logging.basicConfig(level=logging.INFO)
poplib._MAXLINE=20480

def decode_str(s):
    if not s:
        return None
    value, charset = decode_header(s)[0]
    if charset:
        value = value.decode(charset)
    return value

def get_mails():
    prefix = ''# 需要下载的主题
    host = ''
    username = ''
    password = ''

    server = poplib.POP3_SSL(host,'995')
    server.user(username)
    server.pass_(password)
    # 获得邮件
    messages = [server.retr(i) for i in range(len(server.list()[1])-5, len(server.list()[1]) + 1)]
    messages = [b'\r\n'.join(mssg[1]).decode('gb2312') for mssg in messages]
    messages = [Parser().parsestr(mssg) for mssg in messages]
    print("===="*10)
    messages = messages[::-1]
    for message in messages:
        subject = message.get('Subject')
        subject = decode_str(subject)
        #如果标题匹配
        if subject and subject[:len(prefix)] == prefix:
            value = message.get('From')
            if value:
                hdr, addr = parseaddr(value)
                name = decode_str(hdr)
                value = u'%s <%s>' % (name, addr)
            logging.info("发件人: %s" % value)
            logging.info("标题:%s" % subject)
            for part in message.walk():
                filename = part.get_filename()
                filename = decode_str(filename)
                # 保存附件
                if filename:
                    file_path = filename
                    with open(file_path, 'wb') as fEx:
                        data = part.get_payload(decode=True)
                        fEx.write(data)
                        logging.info("附件%s已保存" % filename)
    server.quit()

if __name__ == '__main__':
    get_mails()

