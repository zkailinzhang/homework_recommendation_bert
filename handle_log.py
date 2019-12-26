
from dateutil import parser
from copy import deepcopy
import json, re, os

package_prefix = "20"

deprecated_prefix = ["1-", "2-"]

PATH = os.path.dirname(os.path.abspath(__file__))


import smtplib 
from email.header import Header
from email.mime.text import MIMEText    
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import datetime 


SENDER = 'zhangkl@knowbox.cn'

RECEIVERS = ['zhangkl@knowbox.cn']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱
TITLE = "turing_tfserving report"
USERNAME = 'zhangkl@knowbox.cn'
PWD = 'xxx!'


class SendMail(object):
    def __init__(self):
        self.content =""
        
    def run(self,logfile):
        self.content = "the day log is not exist!!!\n"
        self.content +=logfile

        print(datetime.datetime.now(),self.content)
        self.send_mail(self.content)

    @staticmethod
    def send_mail(mail_content, title=TITLE):
        message = MIMEText(mail_content, 'plain', 'utf-8')
        message['From'] = Header(SENDER, 'utf-8')  # 发送者
        message['To'] = Header(' '.join(RECEIVERS), 'utf-8')  # 接收者
        message['Subject'] = Header(title, 'utf-8')
        username = USERNAME
        mailpwd = PWD

        try:
            smtpObj = smtplib.SMTP_SSL()  # 465
            smtpObj.connect('smtp.qiye.aliyun.com', 465)  # 465

            smtpObj.login(username, mailpwd)
            smtpObj.sendmail(SENDER, RECEIVERS, message.as_string())
            smtpObj.quit()
        except Exception as e:
            print("send email failed %s" % e)


def log(file: str, s: str):

    record_view_click = os.path.join(PATH,"log/new-view-click-{}.txt".format(s))
    record_view_view = os.path.join(PATH,"log/new-view-view-{}.txt".format(s))

    if os.path.exists(record_view_click) and  os.path.exists(record_view_view):
        return record_view_click
    else:
        regex = re.compile(r'(\\[a-zA-Z0-9]*)+')
        # conf = configparser.ConfigParser()
        # conf.read("reject.ini")

        question_request = {}
        learning_request = {}
        error = 0
        right = 0
        

        for cnt in [1, 2]:
            if not os.path.exists((file + "{}").format(cnt)):
                sem = SendMail()
                sem.run((file + "{}").format(cnt))
                continue

            #with open((file + "{}").format(cnt), "r") as f:
            with open((file + "{}").format(cnt), "r",encoding='utf-8') as f:
                for line, com in enumerate(f.readlines(), 1):
                    if not com.startswith(s):
                        continue
                    if com.find("recommend_question") > -1:
                        position = com.find("request_post")
                        if position > -1:
                            try:
                                dd = eval(com[position + len("request_post") + 1:].strip())
                                if dd["packageId"].startswith(package_prefix):
                                    kk = (dd["courseSection2"], dd["courseSection3"], dd["packageId"])
                                    if kk not in question_request:
                                        question_request[kk] = int(dd["packageId"].split("-")[4])
                            except:
                                continue
                    elif com.find("recommend_learning") > -1:
                        position = com.find("request_post")
                        if position > -1:
                            uid = com.split(" ")[4]
                            try:
                                dd = eval(com[position + len("request_post") + 1:].strip())
                                learning_request[uid] = int(str(dd["sectionId"]).split(",")[0])
                            except:
                                continue

        for cnt in [1, 2]:
            if not os.path.exists((file + "{}").format(cnt)):
                continue

            #with open((file + "{}").format(cnt), "r") as f:
            with open((file + "{}").format(cnt), "r",encoding='utf-8') as f:
                for line, com in enumerate(f.readlines(), 1):
                    if not com.startswith(s):
                        continue
                    if com.find("recommend_learning") > -1:
                        position = com.find("response_data")
                        if position > -1:
                            uid = com.split(" ")[4]

                            a = com[position + len("response_data") + 1:].strip()
                            a = regex.sub(r"*", a)
                            a = a.replace("'", '"').replace("u", "")
                            try:
                                dd = json.loads(a)['data']['homeWorkList']
                                right += 1
                                tt = com[:19].strip()
                                tt = parser.parse(tt)

                                for i in dd:
                                    key = (i["corseSection2"], i["corseSection3"], i["packageId"])
                                    if key in question_request:
                                        label = question_request[key]
                                        cop = sorted(deepcopy(dd), key=lambda x: int(x["rank"]))
                                        nn = [[tt, d["teacherId"], d["classId"], d["corseSection2"],
                                            d["corseSection3"], d["style"], d["rank"],
                                            "click" if (d["corseSection2"], d["corseSection3"],
                                                        d["packageId"]) in question_request else "view",
                                            learning_request[uid], ]
                                            for d in cop if d["rank"] <= label]
                                        with open(record_view_click, "a+") as aim:
                                            for n in nn:
                                                if n[5] == "":
                                                    continue
                                                aim.write(";".join([str(cons) for cons in n]))
                                                aim.write("\r\n")
                                        break
                                else:
                                    if dd[0]["packageId"][:2] == package_prefix:
                                        cop = sorted(deepcopy(dd), key=lambda x: int(x["rank"]))
                                        nn = [[tt, d["teacherId"], d["classId"], d["corseSection2"],
                                            d["corseSection3"], d["style"], d["rank"], "view",
                                            learning_request[uid]] for d in cop]
                                        with open(record_view_view, "a+") as aim:
                                            for n in nn:
                                                if n[5] == "":
                                                    continue
                                                aim.write(";".join([str(cons) for cons in n]))
                                                aim.write("\r\n")
                            except Exception as e:
                                # print(e)
                                error += 1
        return record_view_click
    # print(right, error)


def log_raw(file: str, s: str):
    regex = re.compile(r'(\\[a-zA-Z0-9]*)+')
    # conf = configparser.ConfigParser()
    # conf.read("reject.ini")

    question_request = {}
    learning_request = {}
    error = 0
    right = 0

    for cnt in [1, 2]:
        if not os.path.exists((file + "{}").format(cnt)):
            continue

        with open((file + "{}").format(cnt), "r") as f:
            for line, com in enumerate(f.readlines(), 1):
                if not com.startswith(s):
                    continue
                if com.find("recommend_question") > -1:
                    position = com.find("request_post")
                    if position > -1:
                        try:
                            dd = eval(com[position + len("request_post") + 1:].strip())
                            if dd["packageId"].startswith(package_prefix):
                                kk = (dd["courseSection2"], dd["courseSection3"], dd["packageId"])
                                if kk not in question_request:
                                    question_request[kk] = int(dd["packageId"].split("-")[4])
                        except:
                            continue
                elif com.find("recommend_learning") > -1:
                    position = com.find("request_post")
                    if position > -1:
                        uid = com.split(" ")[4]
                        try:
                            dd = eval(com[position + len("request_post") + 1:].strip())
                            learning_request[uid] = int(str(dd["sectionId"]).split(",")[0])
                        except:
                            continue

    for cnt in [1, 2]:
        if not os.path.exists((file + "{}").format(cnt)):
            continue

        with open((file + "{}").format(cnt), "r") as f:
            for line, com in enumerate(f.readlines(), 1):
                if not com.startswith(s):
                    continue
                if com.find("recommend_learning") > -1:
                    position = com.find("response_data")
                    if position > -1:
                        uid = com.split(" ")[4]

                        a = com[position + len("response_data") + 1:].strip()
                        a = regex.sub(r"*", a)
                        a = a.replace("'", '"').replace("u", "")
                        try:
                            dd = json.loads(a)['data']['homeWorkList']
                            right += 1
                            tt = com[:19].strip()
                            tt = parser.parse(tt)

                            for i in dd:
                                key = (i["corseSection2"], i["corseSection3"], i["packageId"])
                                if key in question_request:
                                    label = question_request[key]
                                    cop = sorted(deepcopy(dd), key=lambda x: int(x["rank"]))
                                    nn = [[tt, d["teacherId"], d["classId"], d["corseSection2"],
                                           d["corseSection3"], d["style"], d["rank"],
                                           "click" if (d["corseSection2"], d["corseSection3"],
                                                       d["packageId"]) in question_request else "view",
                                           learning_request[uid], ]
                                          for d in cop if d["rank"] <= label]
                                    with open(record_view_click, "a+") as aim:
                                        for n in nn:
                                            if n[5] == "":
                                                continue
                                            aim.write(";".join([str(cons) for cons in n]))
                                            aim.write("\r\n")
                                    break
                            else:
                                if dd[0]["packageId"][:2] == package_prefix:
                                    cop = sorted(deepcopy(dd), key=lambda x: int(x["rank"]))
                                    nn = [[tt, d["teacherId"], d["classId"], d["corseSection2"],
                                           d["corseSection3"], d["style"], d["rank"], "view",
                                           learning_request[uid]] for d in cop]
                                    with open(record_view_view, "a+") as aim:
                                        for n in nn:
                                            if n[5] == "":
                                                continue
                                            aim.write(";".join([str(cons) for cons in n]))
                                            aim.write("\r\n")
                        except Exception as e:
                            # print(e)
                            error += 1
    # print(right, error)



if __name__ == "__main__":
    log("/data/lishuang/handle-log/log/flaskapp.log.2019-08-01.", "2019-08-01")
