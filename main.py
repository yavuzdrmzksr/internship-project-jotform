import fourier
import holt_winters
import lstm
import sarimax
from pandas import read_csv
from numpy import array
import warnings

warnings.filterwarnings('ignore')

def mail_values(filename):
    raw_seq = read_csv(filename, parse_dates =["Category"], index_col ="Category")
    raw_seq = raw_seq.resample('MS').mean()
    raw_seq = array(raw_seq.interpolate()).flatten()

    preds=[]

    mod = fourier.fit_model(raw_seq[:-13],12)
    rmse,mean,std,iqrlimit = fourier.last_years_rmse(raw_seq[:-1],mod)
    fourier.plot_last_year(raw_seq[:-1],mod,title="Fourier Analysis Model - Test with Previous Data",filename="fourier-test.png")
    mod = fourier.fit_model(raw_seq[:-1])
    next_month = fourier.next_month(mod)
    mod = fourier.fit_model(raw_seq)
    next_6_months = list(map(round,fourier.next_6_months(mod)))
    z = (abs(next_month-raw_seq[-1])-mean)/std
    fourier.plot_next_6_months(raw_seq,mod,title="Fourier Analysis Model - Predictions of Next 6 Months",filename="fourier-pred.png")
    preds.append(("Fourier Analysis Model",round(rmse),"fourier-test.png",round(next_month),round(raw_seq[-1]),round(abs(next_month-raw_seq[-1])),round(abs(next_month-raw_seq[-1])/raw_seq[-1]*100,2),next_6_months,"fourier-pred.png",z>3.5,abs(next_month-raw_seq[-1])>iqrlimit))

    mod = holt_winters.fit_model(raw_seq[:-13])
    rmse,mean,std,iqrlimit = holt_winters.last_years_rmse(raw_seq[:-1],mod)
    holt_winters.plot_last_year(raw_seq[:-1],mod,title="Holt-Winters Model - Test with Previous Data",filename="hw-test.png")
    mod = holt_winters.fit_model(raw_seq[:-1])
    next_month = holt_winters.next_month(mod)
    mod = holt_winters.fit_model(raw_seq)
    next_6_months = list(map(round,holt_winters.next_6_months(mod)))
    z = (abs(next_month-raw_seq[-1])-mean)/std
    holt_winters.plot_next_6_months(raw_seq,mod,title="Holt-Winters Model - Predictions of Next 6 Months",filename="hw-pred.png")
    preds.append(("Holt-Winters Model",round(rmse),"hw-test.png",round(next_month),round(raw_seq[-1]),round(abs(next_month-raw_seq[-1])),round(abs(next_month-raw_seq[-1])/raw_seq[-1]*100,2),next_6_months,"hw-pred.png",z>3.5,abs(next_month-raw_seq[-1])>iqrlimit))

    mod = lstm.fit_model(raw_seq[:-13])
    rmse,mean,std,iqrlimit = lstm.last_years_rmse(raw_seq[:-1],mod)
    lstm.plot_last_year(raw_seq[:-1],mod,title="LSTM Network Model - Test with Previous Data",filename="lstm-test.png")
    mod = lstm.fit_model(raw_seq[:-1])
    next_month = lstm.next_month(raw_seq[:-1],mod)
    mod = lstm.fit_model(raw_seq)
    next_6_months = list(map(round,lstm.next_6_months(raw_seq,mod)))
    z = (abs(next_month-raw_seq[-1])-mean)/std
    lstm.plot_next_6_months(raw_seq,mod,title="LSTM Network Model - Predictions of Next 6 Months",filename="lstm-pred.png")
    preds.append(("LSTM Network Model",round(rmse),"lstm-test.png",round(next_month),round(raw_seq[-1]),round(abs(next_month-raw_seq[-1])),round(abs(next_month-raw_seq[-1])/raw_seq[-1]*100,2),next_6_months,"lstm-pred.png",z>3.5,abs(next_month-raw_seq[-1])>iqrlimit))

    mod = sarimax.fit_model(raw_seq[:-13])
    rmse,mean,std,iqrlimit = sarimax.last_years_rmse(raw_seq[:-1],mod)
    sarimax.plot_last_year(raw_seq[:-1],mod,title="Sarimax Model - Test with Previous Data",filename="sarimax-test.png")
    mod = sarimax.fit_model(raw_seq[:-1])
    next_month = sarimax.next_month(mod)
    mod = sarimax.fit_model(raw_seq)
    next_6_months = list(map(round,sarimax.next_6_months(mod)))
    z = (abs(next_month-raw_seq[-1])-mean)/std
    sarimax.plot_next_6_months(raw_seq,mod,title="Sarimax Model - Predictions of Next 6 Months",filename="sarimax-pred.png")
    preds.append(("Sarimax Model",round(rmse),"sarimax-test.png",round(next_month),round(raw_seq[-1]),round(abs(next_month-raw_seq[-1])),round(abs(next_month-raw_seq[-1])/raw_seq[-1]*100,2),next_6_months,"sarimax-pred.png",z>3.5,abs(next_month-raw_seq[-1])>iqrlimit))

    return sorted(preds,key=lambda x : x[1])

###############################################################################

def mail_text(filename,m1="m1.html",m2="m2.html",m3="m3.html",currency_symbol=""):
    vals=mail_values(filename)
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
        s=s.replace("%RMSE%",currency_symbol+'{:,}'.format(int(rmse)))
        images.append(testimg)
        s=s.replace("%TESTIMG%","cid:"+str(c))
        c+=1
        s=s.replace("%PREDICTION%",currency_symbol+'{:,}'.format(int(pred)))
        s=s.replace("%OBSERVATION%",currency_symbol+'{:,}'.format(int(observation)))
        s=s.replace("%DIFFERENCE%",currency_symbol+'{:,}'.format(int(error)))
        s=s.replace("%PDIFFERENCE%",'{:,}'.format(perror))
        for i in range(6):
            s=s.replace("%PREDICTION"+str(i)+"%",currency_symbol+'{:,}'.format(int(pred6[i])))
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
