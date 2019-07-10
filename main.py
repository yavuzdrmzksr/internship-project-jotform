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
    raw_seq= array(raw_seq.interpolate()).flatten()

    preds=[]

    mod = fourier.fit_model(raw_seq[:-12])
    rmse = round(fourier.last_years_rmse(raw_seq,mod),2)
    fourier.plot_last_year(raw_seq,mod,title="Fourier Analysis Model - Test with Previous Data",filename="fourier-test.png")
    mod = fourier.fit_model(raw_seq[:-1])
    next_month = round(fourier.next_month(mod),2)
    mod = fourier.fit_model(raw_seq)
    next_6_months = list(map(lambda x:round(x,2),fourier.next_6_months(mod)))
    fourier.plot_next_6_months(raw_seq,mod,title="Fourier Analysis Model - Predictions of Next 6 Months",filename="fourier-pred.png")
    preds.append(("Fourier Analysis Model",rmse,"fourier-test.png",next_month,round(raw_seq[-1],2),round(abs(next_month-raw_seq[-1])/raw_seq[-1]*100,2),next_6_months,"fourier-pred.png"))

    mod = holt_winters.fit_model(raw_seq[:-12])
    rmse = round(holt_winters.last_years_rmse(raw_seq,mod),2)
    holt_winters.plot_last_year(raw_seq,mod,title="Holt-Winters Model - Test with Previous Data",filename="hw-test.png")
    mod = holt_winters.fit_model(raw_seq[:-1])
    next_month = round(holt_winters.next_month(mod),2)
    mod = holt_winters.fit_model(raw_seq)
    next_6_months = list(map(lambda x:round(x,2),holt_winters.next_6_months(mod)))
    holt_winters.plot_next_6_months(raw_seq,mod,title="Holt-Winters Model - Predictions of Next 6 Months",filename="hw-pred.png")
    preds.append(("Holt-Winters Model",rmse,"hw-test.png",next_month,round(raw_seq[-1],2),round(abs(next_month-raw_seq[-1])/raw_seq[-1]*100,2),next_6_months,"hw-pred.png"))

    mod = lstm.fit_model(raw_seq[:-12])
    rmse = round(lstm.last_years_rmse(raw_seq,mod),2)
    lstm.plot_last_year(raw_seq,mod,title="LSTM Network Model - Test with Previous Data",filename="lstm-test.png")
    mod = lstm.fit_model(raw_seq[:-1])
    next_month = round(lstm.next_month(raw_seq,mod),2)
    mod = lstm.fit_model(raw_seq)
    next_6_months = list(map(lambda x:round(x,2),lstm.next_6_months(raw_seq,mod)))
    lstm.plot_next_6_months(raw_seq,mod,title="LSTM Network Model - Predictions of Next 6 Months",filename="lstm-pred.png")
    preds.append(("LSTM Network Model",rmse,"lstm-test.png",next_month,round(raw_seq[-1],2),round(abs(next_month-raw_seq[-1])/raw_seq[-1]*100,2),next_6_months,"lstm-pred.png"))

    mod = sarimax.fit_model(raw_seq[:-12])
    rmse = round(sarimax.last_years_rmse(raw_seq,mod),2)
    sarimax.plot_last_year(raw_seq,mod,title="Sarimax Model - Test with Previous Data",filename="sarimax-test.png")
    mod = sarimax.fit_model(raw_seq[:-1])
    next_month = round(sarimax.next_month(mod),2)
    mod = sarimax.fit_model(raw_seq)
    next_6_months = list(map(lambda x:round(x,2),sarimax.next_6_months(mod)))
    sarimax.plot_next_6_months(raw_seq,mod,title="Sarimax Model - Predictions of Next 6 Months",filename="sarimax-pred.png")
    preds.append(("Sarimax Model",rmse,"sarimax-test.png",next_month,round(raw_seq[-1],2),round(abs(next_month-raw_seq[-1])/raw_seq[-1]*100,2),next_6_months,"sarimax-pred.png"))

    return sorted(preds,key=lambda x : x[1])

###############################################################################

def mail_text(filename,m1="m1.html",m2="m2.html",m3="m3.html"):
    vals=mail_values(filename)
    h=open(m1,"r")
    text = h.read().replace("%FILENAME%",filename)
    images=[]

    c=0
    for (model,rmse,testimg,pred,observation,error,pred6,predimg) in vals:
        h=open(m2,"r")
        s = h.read()
        s=s.replace("%MODEL%",model)
        s=s.replace("%RMSE%",str(rmse))
        images.append(testimg)
        s=s.replace("%TESTIMG%","cid:"+str(c))
        c+=1
        s=s.replace("%PREDICTION%",str(pred))
        s=s.replace("%OBSERVATION%",str(observation))
        s=s.replace("%DIFFERENCE%",str(error))
        for i in range(6):
            s=s.replace("%PREDICTION"+str(i)+"%",str(pred6[i]))
        images.append(predimg)
        s=s.replace("%PREDIMG%","cid:"+str(c))
        c+=1
        text+=s

    h=open(m3,"r")
    text += h.read()

    return (text,images,filename+" Report")
