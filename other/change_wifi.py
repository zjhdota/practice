import os

os.system("netsh wlan disconnect")
os.system('netsh wlan connect ssid=teleinfo-2.4G name=teleinfo-2.4G"')
