import smtplib, ssl
from email.mime.text import MIMEText
import logging

def notify(subject: str, body: str) -> None:
    port = 994
    server = "smtp.163.com"
    password = '<replace with your password>'
    sender_email = '<replace with your email address>'
    receiver_email = 'xianzma@gmail.com'
    message = MIMEText(body, "plain", "utf-8")
    message['Subject'] = subject
    message['To'] = receiver_email
    message['From'] = sender_email
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, [receiver_email], message.as_string())
    except Exception as ex:
        logging.exception(ex)
    else:
        logging.info('email is sent successfully')


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s: [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S',
        handlers=[
            # also output to console
            logging.StreamHandler(),
        ]
    )
    notify('notification test', 'I hope it succeeds!')

