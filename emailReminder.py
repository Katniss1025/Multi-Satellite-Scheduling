import smtplib
from email.mime.text import MIMEText
from utils import get_args


def send_email(sender, receiver, subject, message):
    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver

    try:
        # 请根据你的邮件服务商和授权信息进行相应的配置
        smtp_server = "smtp.126.com"
        smtp_port = 25
        username = "katniss5050@126.com"
        password = "HMSLUQXWBSDRIRUB"

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # 启用安全传输层协议
            server.login(username, password)
            server.sendmail(sender, receiver, msg.as_string())
        print("邮件已成功发送")
    except Exception as e:
        print("发送邮件时出现错误:", str(e))



