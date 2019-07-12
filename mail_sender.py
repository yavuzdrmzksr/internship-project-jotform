import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from main import mail_text

sender_email = "tjotform@gmail.com"
receiver_email = "yavuzbirk@mailinator.com"
password = "xx1234xx"

message = MIMEMultipart("alternative")

message["From"] = sender_email
message["To"] = receiver_email

html,images,message["Subject"] = mail_text("../p30-(paid-users-who-upgraded-in-the-first-30-days)-(saas).csv")

part2 = MIMEText(html, "html")

message.attach(part2)

for i in range(len(images)):

    with open(images[i], 'rb') as f:
        # set attachment mime and file name, the image type is png
        mime = MIMEBase('image', 'png', filename=images[i])
        # add required header data:
        mime.add_header('Content-Disposition', 'attachment', filename=images[i])
        mime.add_header('X-Attachment-Id', str(i))
        mime.add_header('Content-ID', '<'+str(i)+'>')
        # read attachment file content into the MIMEBase object
        mime.set_payload(f.read())
        # encode with base64
        encoders.encode_base64(mime)
        # add MIMEBase object to MIMEMultipart object
        message.attach(mime)

# Create secure connection with server and send email
context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(
        sender_email, receiver_email, message.as_string()
    )
