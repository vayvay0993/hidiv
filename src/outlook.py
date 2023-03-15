import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import os

def send_email_with_attach(mail_server: str, sender: str, receivers:str, subject:str, content: str, file_path_list: str) -> None:
    message = MIMEMultipart()
    message['To'] = Header(';'.join(receivers), 'utf-8')
    message['Subject'] = Header(subject, 'utf-8')
    message.attach(MIMEText(content, 'plain', 'utf-8'))
    for file_path in file_path_list:
        filename = os.path.basename(file_path)
        att1 = MIMEText(open(file_path, 'rb').read(), 'base64', 'utf-8')
        att1["Content-Type"] = 'application/octet-stream'
        att1["Content-Disposition"] = 'attachment; filename="{}"'.format(filename)
        message.attach(att1)
    smtpObj = smtplib.SMTP(mail_server)
    smtpObj.sendmail(sender, receivers, message.as_string())    