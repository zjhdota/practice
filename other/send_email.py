import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

fromaddr = ""
toaddr = ""

msg = MIMEMultipart()
msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = '1234'
msg.attach(MIMEText(html, 'html'))

server = smtplib.SMTP(address,port)
server.login(username,password)

text = msg.as_string()
server.sendmail(fromaddr, toaddr, text)
