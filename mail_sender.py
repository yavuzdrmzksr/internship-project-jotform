import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import fourier
import holt_winters
import lstm
import sarimax
from pandas import read_csv
from numpy import array
import warnings

###############################################################################

warnings.filterwarnings('ignore')

###############################################################################

def mail_values(filename):
    raw_seq = read_csv(filename, parse_dates =["Category"], index_col ="Category")
    raw_seq = raw_seq.resample('MS').mean()
    current_date=raw_seq.index[-1].year*12+raw_seq.index[-1].month
    raw_seq = array(raw_seq.interpolate()).flatten()

    last_years_months_1=[]
    last_years_months_2=[]
    last_year_and_next_6_months=[]

    for i in range(-13,6):
        if((current_date+i)%12)<9:
            pre="0"
        else:
            pre=""
        if(i>-13):
            last_year_and_next_6_months.append(str(int((current_date+i)/12))+"-"+pre+str(((current_date+i)%12)+1))
        if(i<-1):
            last_years_months_1.append(str(int((current_date+i)/12))+"-"+pre+str(((current_date+i)%12)+1))
        if(i>-13 and i<0):
            last_years_months_2.append(str(int((current_date+i)/12))+"-"+pre+str(((current_date+i)%12)+1))

    preds=[]

    mod = fourier.fit_model(raw_seq[:-13],12)
    rmse,mean,std,iqrlimit = fourier.last_years_rmse(raw_seq[:-1],mod)
    fourier.plot_last_year(raw_seq[:-1],mod,"Fourier Analysis Model - Test with Previous Data","fourier-test.png",last_years_months_1)
    mod = fourier.fit_model(raw_seq[:-1])
    next_month = fourier.next_month(mod)
    mod = fourier.fit_model(raw_seq)
    next_6_months = list(map(lambda x:round(x,2),fourier.next_6_months(mod)))
    z = (abs(next_month-raw_seq[-1])-mean)/std
    fourier.plot_next_6_months(raw_seq,mod,"Fourier Analysis Model - Predictions of Next 6 Months","fourier-pred.png",last_years_months_2,last_year_and_next_6_months)
    preds.append(("Fourier Analysis Model",round(rmse,2),"fourier-test.png",round(next_month,2),round(raw_seq[-1],2),round(abs(next_month-raw_seq[-1]),2),round(abs(next_month-raw_seq[-1])/raw_seq[-1]*100,2),next_6_months,"fourier-pred.png",z>3.5,abs(next_month-raw_seq[-1])>iqrlimit))

    mod = holt_winters.fit_model(raw_seq[:-13])
    rmse,mean,std,iqrlimit = holt_winters.last_years_rmse(raw_seq[:-1],mod)
    holt_winters.plot_last_year(raw_seq[:-1],mod,"Holt-Winters Model - Test with Previous Data","hw-test.png",last_years_months_1)
    mod = holt_winters.fit_model(raw_seq[:-1])
    next_month = holt_winters.next_month(mod)
    mod = holt_winters.fit_model(raw_seq)
    next_6_months = list(map(lambda x:round(x,2),holt_winters.next_6_months(mod)))
    z = (abs(next_month-raw_seq[-1])-mean)/std
    holt_winters.plot_next_6_months(raw_seq,mod,"Holt-Winters Model - Predictions of Next 6 Months","hw-pred.png",last_years_months_2,last_year_and_next_6_months)
    preds.append(("Holt-Winters Model",round(rmse,2),"hw-test.png",round(next_month,2),round(raw_seq[-1],2),round(abs(next_month-raw_seq[-1]),2),round(abs(next_month-raw_seq[-1])/raw_seq[-1]*100,2),next_6_months,"hw-pred.png",z>3.5,abs(next_month-raw_seq[-1])>iqrlimit))

    mod = lstm.fit_model(raw_seq[:-13])
    rmse,mean,std,iqrlimit = lstm.last_years_rmse(raw_seq[:-1],mod)
    lstm.plot_last_year(raw_seq[:-1],mod,"LSTM Network Model - Test with Previous Data","lstm-test.png",last_years_months_1)
    mod = lstm.fit_model(raw_seq[:-1])
    next_month = float(lstm.next_month(raw_seq[:-1],mod))
    mod = lstm.fit_model(raw_seq)
    next_6_months = list(map(lambda x:round(float(x),2),lstm.next_6_months(raw_seq,mod)))
    z = (abs(next_month-raw_seq[-1])-mean)/std
    lstm.plot_next_6_months(raw_seq,mod,"LSTM Network Model - Predictions of Next 6 Months","lstm-pred.png",last_years_months_2,last_year_and_next_6_months)
    preds.append(("LSTM Network Model",round(rmse,2),"lstm-test.png",round(next_month,2),round(raw_seq[-1],2),round(abs(next_month-raw_seq[-1]),2),round(abs(next_month-raw_seq[-1])/raw_seq[-1]*100,2),next_6_months,"lstm-pred.png",z>3.5,abs(next_month-raw_seq[-1])>iqrlimit))

    mod = sarimax.fit_model(raw_seq[:-13])
    rmse,mean,std,iqrlimit = sarimax.last_years_rmse(raw_seq[:-1],mod)
    sarimax.plot_last_year(raw_seq[:-1],mod,"Sarimax Model - Test with Previous Data","sarimax-test.png",last_years_months_1)
    mod = sarimax.fit_model(raw_seq[:-1])
    next_month = sarimax.next_month(mod)
    mod = sarimax.fit_model(raw_seq)
    next_6_months = list(map(lambda x:round(x,2),sarimax.next_6_months(mod)))
    z = (abs(next_month-raw_seq[-1])-mean)/std
    sarimax.plot_next_6_months(raw_seq,mod,"Sarimax Model - Predictions of Next 6 Months","sarimax-pred.png",last_years_months_2,last_year_and_next_6_months)
    preds.append(("Sarimax Model",round(rmse,2),"sarimax-test.png",round(next_month,2),round(raw_seq[-1],2),round(abs(next_month-raw_seq[-1]),2),round(abs(next_month-raw_seq[-1])/raw_seq[-1]*100,2),next_6_months,"sarimax-pred.png",z>3.5,abs(next_month-raw_seq[-1])>iqrlimit))

    return sorted(preds,key=lambda x : x[1]),last_year_and_next_6_months[-6:]

###############################################################################

def mail_text(filename,m1="m1.html",m2="m2.html",m3="m3.html",currency_symbol=""):
    vals,month_replace=mail_values(filename)
    h=open(m1,"r")
    text1 = h.read().replace("%FILENAME%",filename)
    images=[]

    c=0
    text3=""
    wars=[]
    for (model,rmse,testimg,pred,observation,error,perror,pred6,predimg,z_warning,iqr_warning) in vals:
        h=open(m2,"r")
        s = h.read()
        s=s.replace("%MODEL%",model)
        s=s.replace("%RMSE%",currency_symbol+'{:,}'.format(rmse))
        images.append(testimg)
        s=s.replace("%TESTIMG%","cid:"+str(c))
        c+=1
        s=s.replace("%PREDICTION%",currency_symbol+'{:,}'.format(pred))
        s=s.replace("%OBSERVATION%",currency_symbol+'{:,}'.format(observation))
        s=s.replace("%DIFFERENCE%",currency_symbol+'{:,}'.format(error))
        s=s.replace("%PDIFFERENCE%",'{:,}'.format(perror))
        for i in range(6):
            s=s.replace("%MONTH"+str(i)+"%",month_replace[i])
            s=s.replace("%PREDICTION"+str(i)+"%",currency_symbol+'{:,}'.format(pred6[i]))
        images.append(predimg)
        s=s.replace("%PREDIMG%","cid:"+str(c))
        c+=1
        text3+=s
        if(z_warning or iqr_warning):
            wars.append((model,z_warning,iqr_warning))

    if(not wars):
        text2 = "<h3>There are no warnings.<h3><br>"
    else:
        text2 = "<h3>Following model(s) should be checked:</h3><p>"
        for (model,z_warning,iqr_warning) in wars:
            text2+="&emsp;&emsp;&emsp;<b>"+model+"</b>"
            if(z_warning and iqr_warning):
                text2+=" (because z score of error is high and error is greater than sum of 3rd quartile and 1.5*IQR)<br>"
            elif(z_warning):
                text2+=" (because z score of error is high)<br>"
            else:
                text2+=" (because error is greater than sum of 3rd quartile and 1.5*IQR)<br>"
        text2 += "</p><br>"

    h=open(m3,"r")
    text4 = h.read()

    text = text1+text2+text3+text4

    return (text,images,filename+" Report")

###############################################################################

def send_mail(filename,currency_symbol=""):
    sender_email = "tjotform@gmail.com"
    receiver_email = "tjotform@gmail.com"
    password = "xx1234xx"

    message = MIMEMultipart("alternative")

    message["From"] = sender_email
    message["To"] = receiver_email

    html,images,message["Subject"] = mail_text(filename,currency_symbol=currency_symbol)

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
