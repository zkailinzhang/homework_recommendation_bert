#-*- coding:utf-8 -*-
import sys, time, os
import tensorflow as tf

import numpy as np 
import pandas as pd
import utils
import json
import csv
import logging
import os 
import datetime
import time
import handle_log
import feature
import handle_train
import model_sum_34
import traceback
import copy

PATH = os.path.dirname(os.path.abspath(__file__))
flags = tf.app.flags

flags.DEFINE_string("logfile",'logs.sum.34.out',"log file to save")
flags.DEFINE_string("csvfile","test_metric_day.sum.34.csv",'csv file to save test metric')


FLAGS = flags.FLAGS

logging.basicConfig(filename=os.path.join(PATH,FLAGS.logfile),filemode='a',
format='%(asctime)s %(name)s:%(levelname)s:%(message)s',datefmt="%d-%m-%Y %H:%M:%S",
level=logging.DEBUG)


cha_file = os.path.join(PATH,"dic_base_cha_final.json")
sec_file = os.path.join(PATH,"dic_base_sec_final.json")
ref_file = os.path.join(PATH,"dic_reflect.json")

STYLE_MAP = {}
STYLE = []
ccccc = 1
for fir in ["1", "2", "3", "4"]:
    for sec in ["100", "010", "001", "110", "101", "011", "111"]:
        key = fir + "0" + sec
        STYLE.append(key)
        STYLE_MAP[int(key)] = ccccc
        ccccc += 1
del ccccc, fir, sec

RST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_sum_34")
BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save_sum_34")
MODLE_PATH = os.path.join(BASE_PATH, "model")
SERVE_PATH = os.path.join(BASE_PATH, "serving")
os.makedirs(MODLE_PATH, exist_ok=True)
os.makedirs(SERVE_PATH, exist_ok=True)
os.makedirs(RST_PATH, exist_ok=True)

HIS_DAYS_NUM = 14



def repeat(ww):

    vv = copy.deepcopy(ww)
    #看前面
    for i  in range(len(ww)):
        one = ww[i]
        one_np= np.array(one)
        len1 = len(one)
        len2 =0
        #若某一天 全为零 
        if not ((one_np ==0).all()):
            #j =i+1 
            for j in range(i+1,len(ww)):
                tmp = ww[j]
                len2 = len(tmp)
                if not ((np.array(tmp) ==0).all()):
                    break    
                else:
                    new = []
                    if len2 <=len1:                    
                        for k in range(len2):
                            new.append(ww[i][k])                      
                    else:           
                        for k in range(len2):
                            if k <= (len1-1):
                                new.append(ww[i][k])
                            else:
                                new.append(ww[i][len1-1]) 
                    vv[j] = new
    #看后面
    for i  in range(len(vv)):
        one_np= np.array(vv[i])
        len1 = len(vv[i])
        len2 = 0
        #若某一天 全为零 
        if not ((one_np ==0).all()):
            for j in range(i):
                len2 = len(vv[j])
                new = []
                if len2 <=len1:                    
                    for k in range(len2):
                        new.append(vv[i][k])                      
                else:           
                    for k in range(len2):
                        if k <= (len1-1):
                            new.append(vv[i][k])
                        else:
                            new.append(vv[i][len1-1]) 
                vv[j] = new
            break

    return vv


def prepare_data(data):
    def to_int(l):
        return [int(i) if i!='' else 0 for i in l]

    resp = {}
    keep = ["teacher_id_ph", "student_count_ph",
            "province_id_ph", "city_id_ph", "core_type_ph",

            "class_id_ph", "edition_id_ph", "grade_id_ph",
            "class_student_ph", "cap_avg_ph", "cap_max_ph", "cap_min_ph",

            "gap_days_ph",

            "month_submit_rate_ph",

            "region_capacity_ph",

            "prefer_assign_time_avg_ph", "prefer_assign_time_var_ph",
            "prefer_assign_rank_avg_ph", "prefer_assign_rank_var_ph",

            "register_diff_ph",

            "homework_count_ph",

            "week_count_ph",

            "lastday_count_ph"
            ]
    for key in keep:
        resp[key] = [i[key] for i in data]
    resp["study_vector_ph"] = [eval(i["study_vector_ph"]) for i in data]

    tchap_ = ["today_chapters_ph",  ]
    with open(cha_file,'r') as jf:
        cha = json.loads(jf.read())

        for item in tchap_:

            aim = [to_int(str(i[item]).strip().split(",")) for i in data]
            length = max([len(i) for i in aim])     
            rst = copy.deepcopy(aim)

            for i in range(len(aim)):  
                for j in range(len(aim[i])): 
                        rst[i][j]=cha[str(aim[i][j])]

            for k, v in enumerate(rst):
                rst[k] = v + [v[-1]] * (length - len(v))
            resp[item] = rst 

 
    tsec_ = [ "today_sections_ph"]
    with open(sec_file,'r') as jf:
        sec = json.loads(jf.read())

        for item in tsec_:

            aim = [to_int(str(i[item]).strip().split(",")) for i in data]
            length = max([len(i) for i in aim])
            
            rst = copy.deepcopy(aim)

            for i in range(len(aim)):
                for j in range(len(aim[i])): 
                        rst[i][j]=sec[str(aim[i][j])]

            for k, v in enumerate(rst):
                rst[k] = v + [v[-1]] * (length - len(v))
            resp[item] = rst

 
    his_days =['one', 'two', 'three', 'four','five','six','seven','eight','nine',
    'ten','eleven','twelve','thirteen','fourteen']
    chap_ = list(map(lambda x: "history_" + x + "_chap_ph", his_days))

    with open(cha_file,'r') as jf:
        cha = json.loads(jf.read())
        for item in chap_:
            #aaaaa = [str(i[item]) for i in data]
            #一个批次里面的所有one ，保证长度一样
            aim = [to_int(str(i[item]).strip().split(",")) for i in data]

            #aim = repeat(aim_)
            rst = copy.deepcopy(aim)

            length = max([len(i) for i in aim])

            for i in range(len(aim)):  
                for j in range(len(aim[i])): 
                        # if aim[i][j]==126:
                        #     print(aim[i])
                    rst[i][j]=cha[str(aim[i][j])]

            for k, v in enumerate(rst):
                
                rst[k] = v + [v[-1]] * (length - len(v))
            resp[item] = rst

    sec_ = list(map(lambda x: "history_" + x + "_sec_ph", his_days))

    
    with open(sec_file,'r') as jf:
        sec = json.loads(jf.read())
        for item in sec_:

            aim = [to_int(str(i[item]).strip().split(",")) for i in data]
            
            #aim = repeat(aim)
            rst = copy.deepcopy(aim)
            length = max([len(i) for i in aim])

            # with open(sec_file,'r') as jf:
            #     sec = json.loads(jf.read())
            for i in range(len(aim)):
                # if len(aim[i])== 0: 
                #     continue   
                # else:
                for j in range(len(aim[i])): 
                    rst[i][j]=sec[str(aim[i][j])]

            for k, v in enumerate(rst):
                rst[k] = v + [v[-1]] * (length - len(v))
            resp[item] = rst
    
    #先保证列等长 占位符，  所有样本的第一天， 用最后一位 填充，同时 做映射
    # 在 一个样本的所有天，，中间缺失天的  repeat  在重新赋值
    
    
    sample_nums =len(resp["history_one_chap_ph"])
    for i in range(sample_nums):
        cha_days=[]
        for item in chap_[::-1]:
            tmp = resp[item][i]
            cha_days.append(tmp)
        cha_days_aim = repeat(cha_days)
        
        num=0
        for item in chap_[::-1]:
            resp[item][i] = cha_days_aim[num]
            num+=1

    for i in range(sample_nums):
        sec_days=[]
        for item in sec_[::-1]:
            tmp = resp[item][i]
            sec_days.append(tmp)
        sec_days_aim = repeat(sec_days)
        
        num=0
        for item in sec_[::-1]:
            resp[item][i] = sec_days_aim[num]
            num+=1



    ref_ = ["reflect_value_ph"]
    with open(ref_file,'r') as jf:
        ref = json.loads(jf.read())
        for item in ref_:
            aim = [to_int(str(i[item]).strip().split(",")) for i in data]
            length = max([len(i) for i in aim])
            rst = copy.deepcopy(aim)

            for i in range(len(aim)):
                for j in range(len(aim[i])): 
                        rst[i][j]=ref[str(aim[i][j])]

            for k, v in enumerate(rst):
                rst[k] = v + [v[-1]] * (length - len(v))
            resp[item] = rst
        
     #连续值 
    today_style_ph = ["today_style_ph"]
    for item in today_style_ph:
        resp[item] = [STYLE_MAP.get(i[item], 0) for i in data]
    
    #
    style_ph = ["style_ph"]
    for item in style_ph:
        # if item != "style_ph":
        #     raise RuntimeError("style_ph!")
        for s in STYLE:
            resp["style_" + s + "_ph"] = [i[item][s] for i in data]

    # item = "label"
    # resp["target_ph"] = [[0, 1] if i[item] == 1 else [1, 0] for i in data]

    return resp

def find_LCIS(cc):
    cc_ = [int(str(i).strip().split(",")[0])  for i in cc]
    
    ee = np.where(np.asarray(cc_)<=0) 
    ww= ee[0]  
    if ww.size == 0 :
        return 0
    len_max=1
    len_=1

    for i in range(len(ww)-1):
        if ww[i+1]==ww[i]+1:
            len_ +=1 
            len_max = max(len_,len_max)
        else:
            len_=1

    return len_max


def get_lastyear_data(cursor,cursor2, teacher_id,class_id):

        #multi values  limit 1  desc
    # ignore school_id  "result_data10.txt",,add school_id "result_data13.txt"
    # ignore school_id  "result_data16.txt",,add school_id "result_data15.txt"
    sql = """SELECT province_id,city_id,county_id FROM r_teacher_info where teacher_id={}"""
    cursor2.execute(sql.format(teacher_id))
    res = cursor2.fetchall()
    if res == ():
        return 0 
    base_ = res[0]

    sql2 = """SELECT teacher_id FROM r_teacher_info WHERE province_id={} AND city_id={} AND county_id={}"""
    cursor2.execute(sql2.format(base_["province_id"],base_["city_id"],base_["county_id"]))
    res = cursor2.fetchall()
    if res == ():
        return 0 
    base_2 = res

    #term
    #sql3 = """SELECT edition_id,grade FROM r_teacher_class WHERE class_id={} """
    sql3 = """SELECT edition_id,grade FROM r_class_info WHERE class_id={}"""
    cursor2.execute(sql3.format(class_id))
    res = cursor2.fetchall()
    if res == ():
        return 0 
    base_3 = res[0]
    
    from dateutil.relativedelta import relativedelta
    lastyear = (datetime.datetime.strptime(day,"%Y-%m-%d") - relativedelta(months=12)).strftime("%Y-%m-%d")
    
    #term ignore?
    sql3 = """SELECT base_chapters,base_sections FROM daily_homework_info WHERE `day`='{}' AND edition_id={} AND grade={} AND teacher_id in {} GROUP BY teacher_id"""
    base_22 = [str(t["teacher_id"]) for t in base_2]
    base_22 = "(" +",".join(base_22)+ ")"

    cursor.execute(sql3.format(lastyear,base_3["edition_id"],base_3["grade"],base_22))
    res_f = cursor.fetchall()
    #mybe not one
    base_4 = res_f
    if res_f == ():
        return 0 
    #  (1064, "You have an error in your SQL syntax; check the manual that corresponds
    #   to your MySQL server version for the right syntax to use near '2' at line 1")
    return base_4
    
def simplity_cha(last_year_cha,aa):
    
    flag =0
    for num,ii in enumerate(last_year_cha):
        if (aa["base_chapters"] == ii["base_chapters"]) and \
         (aa["base_sections"] == ii["base_sections"]) :
            flag +=1
            return flag

        
    return flag






if __name__ == "__main__":
            
    # result_fin = [13, 12, 14, 5, 14, 14, 12, 9, 14, 14, 14, 14, 7, 1, 14, 9, 12, 12, 2, 12, 10, 14, 9, 14, 14, 14, 14, 9, 2, 14, 14, 14, 1, 13, 14, 13, 9, 10, 14, 14, 12, 14, 13, 11, 8, 10, 10, 11, 14, 11, 12, 12, 12, 12, 12, 11, 3, 11, 3, 11, 12, 11, 14, 10, 11, 14, 11, 11, 9, 11, 12, 6, 10, 10, 11, 4, 11, 12, 10, 14, 10, 12, 13, 11, 12, 11, 10, 13, 12, 11, 11, 11, 10, 12, 10, 13, 14, 10, 13, 9, 13, 12, 14, 10, 10, 11, 12, 14, 12, 7, 11, 11, 13, 12, 10, 10, 9, 13, 11, 11, 14, 11, 11, 1, 9, 14, 11, 9, 13, 11, 10, 8, 14, 11, 14, 11, 12, 11, 13, 10, 14, 14, 14, 14, 11, 13, 11, 5, 14, 14, 14, 12, 11, 9, 10, 12, 7, 4, 12, 11, 13, 12, 10, 14, 5, 13, 14, 12, 10, 10, 14, 11, 14, 14, 14, 4, 9, 4, 12, 7, 10, 14, 11, 14, 14, 14, 10, 10, 11, 10, 12, 11, 7, 9, 10, 11, 6, 14, 12, 14, 14, 13, 14, 7, 11, 14, 14, 11, 14, 8, 14, 14, 13, 6, 8, 13, 9, 11, 10, 13, 12, 12, 12, 13, 11, 11, 10, 13, 11, 12, 11, 13, 5, 6, 10, 9, 14, 14, 11, 10, 14, 14, 13, 5, 11, 14, 13, 13, 14, 10, 11, 9, 7, 14, 10, 13, 12, 11, 4, 14, 12, 3, 10, 14, 10, 11, 12, 5, 13, 14, 12, 14, 14, 11, 11, 5, 12, 13, 12, 14, 7, 11, 14, 6, 11, 10, 14, 10, 11, 13, 12, 14, 13, 14, 14, 11, 12, 9, 13, 12, 11, 11, 13, 14, 11, 6, 11, 14, 14, 12, 14, 14, 12, 11, 11, 7, 7, 5, 11, 14, 12, 12, 11, 11, 14, 12, 14, 11, 13, 14, 12, 2, 13, 11, 13, 11, 3, 8, 14, 12, 6, 13, 13, 9, 11, 11, 13, 10, 14, 14, 11, 10, 10, 12, 11, 10, 10, 11, 11, 11, 11, 12, 11, 14, 11, 12, 12, 12, 14, 13, 13, 12, 12, 13, 13, 14, 14, 14, 14, 14, 14, 13, 13, 13, 14, 14, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 7, 12, 13, 7, 7, 8, 3, 13, 4, 12, 5, 12, 6, 6, 9, 8, 7, 5, 8, 7, 7, 13, 6, 14, 9, 6, 8, 14, 9, 14, 6, 6, 5, 7, 14, 8, 5, 6, 3, 6, 7, 5, 12, 6, 10, 9, 3, 5, 6, 9, 2, 14, 9, 2, 2, 9, 4, 5, 8, 12, 1, 4, 7, 8, 14, 4, 8, 6, 8, 8, 4, 11, 6, 9, 6, 8, 6, 13, 9, 14, 7, 8, 12, 5, 10, 9, 6, 8, 6, 6, 13, 6, 6, 4, 5, 14, 7, 6, 8, 8, 2, 8, 6, 4, 12, 5, 8, 5, 9, 13, 7, 6, 1, 7, 5, 3, 6, 7, 14, 5, 7, 14, 6, 14, 12, 9, 3, 6, 6, 9, 7, 9, 6, 5, 9, 10, 7, 6, 5, 6, 14, 5, 7, 5, 4, 4, 8, 7, 6, 9, 7, 6, 7, 13, 7, 6, 14, 11, 6, 6, 8, 6, 6, 8, 8, 10, 6, 6, 9, 6, 6, 10, 1, 5, 9, 12, 14, 14, 3, 14, 14, 5, 5, 5, 13, 12, 8, 5, 9, 10, 14, 14, 14, 6, 7, 7, 10, 10, 14, 6, 5, 5, 9, 5, 12, 8, 4, 3, 13, 2, 14, 14, 13, 6, 11, 6, 5, 14, 13, 13, 7, 9, 9, 7, 5, 8, 2, 1, 7, 1, 8, 6, 9, 12, 7, 5, 3, 14, 14, 14, 13, 14, 11, 7, 14, 14, 7, 7, 8, 14, 14, 1, 5, 6, 4, 6, 5, 14, 6, 14, 9, 7, 7, 8, 8, 8, 8, 9, 9, 13, 8, 8, 9, 10, 9, 11, 14, 10, 12, 11, 14, 13, 12, 12, 14, 12, 14, 13, 14, 12, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 3, 2, 2, 8, 2, 14, 6, 14, 10, 3, 14, 14, 5, 14, 2, 8, 2, 3, 4, 2, 6, 4, 14, 3, 8, 8, 3, 2, 3, 7, 3, 5, 11, 14, 3, 7, 1, 3, 1, 3, 5, 4, 2, 14, 4, 2, 11, 3, 8, 4, 10, 4, 14, 2, 7, 4, 2, 3, 11, 2, 14, 4, 3, 4, 1, 1, 4, 1, 3, 2, 3, 11, 3, 3, 5, 3, 14, 10, 4, 3, 8, 2, 3, 2, 14, 2, 3, 10, 5, 3, 1, 5, 14, 13, 8, 11, 4, 4, 4, 5, 6, 5, 7, 13, 12, 10, 4, 1, 14, 4, 4, 14, 2, 7, 1, 3, 3, 2, 9, 1, 3, 10, 2, 3, 2, 9, 7, 4, 8, 9, 5, 6, 1, 14, 9, 3, 10, 3, 14, 3, 13, 14, 2, 10, 14, 5, 4, 3, 1, 1, 4, 1, 3, 2, 14, 4, 1, 4, 2, 9, 11, 8, 3, 3, 3, 3, 4, 9, 7, 9, 14, 9, 3, 3, 2, 14, 3, 3, 7, 3, 4, 3, 4, 14, 3, 2, 6, 7, 5, 6, 2, 5, 9, 4, 14, 3, 10, 7, 14, 11, 5, 3, 3, 3, 3, 3, 3, 3, 9, 1, 4, 7, 1, 6, 2, 2, 3, 6, 8, 4, 2, 3, 3, 3, 7, 11, 3, 2, 11, 14, 4, 3, 7, 1, 2, 9, 2, 8, 14, 14, 14, 14, 14, 6, 4, 4, 2, 13, 8, 5, 3, 14, 14, 14, 3, 1, 3, 4, 10, 3, 3, 2, 3, 2, 2, 3, 3, 3, 4, 4, 12, 13, 7, 14, 7, 8, 9, 9, 10, 8, 9, 14, 14, 9, 9, 9, 9, 9, 14, 14, 14, 10, 10, 14, 14, 12, 14, 12, 13, 14, 13, 13, 14, 14, 14, 14, 14, 2, 11, 5, 1, 3, 10, 14, 14, 9, 3, 4, 12, 1, 11, 11, 4, 14, 3, 2, 6, 14, 1, 14, 14, 10, 14, 14, 2, 1, 14, 14, 11, 7, 3, 2, 8, 9, 14, 7, 14, 4, 2, 8, 12, 4, 1, 14, 2, 2, 1, 3, 10, 1, 5, 14, 2, 5, 2, 2, 1, 14, 5, 4, 4, 7, 5, 3, 7, 3, 1, 4, 5, 14, 14, 2, 3, 5, 4, 14, 8, 3, 4, 3, 11, 3, 1, 1, 13, 2, 2, 4, 14, 2, 1, 10, 9, 8, 4, 5, 2, 2, 3, 4, 8, 14, 1, 14, 1, 7, 4, 3, 1, 8, 12, 3, 1, 6, 7, 4, 12, 3, 1, 3, 6, 13, 13, 1, 1, 2, 12, 14, 12, 2, 12, 1, 14, 14, 12, 2, 14, 7, 4, 6, 14, 14, 14, 9, 5, 1, 11, 11, 2, 5, 14, 10, 6, 6, 4, 3, 3, 2, 5, 9, 14, 1, 14, 10, 9, 5, 7, 1, 14, 8, 1, 7, 14, 3, 2, 13, 5, 3, 14, 12, 14, 2, 4, 14, 1, 5, 4, 3, 3, 1, 7, 9, 14, 6, 3, 8, 5, 8, 6, 1, 14, 10, 1, 2, 5, 3, 2, 3, 5, 2, 9, 4, 12, 3, 4, 5, 8, 8, 6, 12, 14, 9, 9, 9, 13, 12, 12, 14, 13, 14, 14, 14, 14, 14, 14, 14, 14, 3, 11, 7, 2, 6, 2, 14, 6, 1, 11, 3, 4, 3, 7, 7, 4, 4, 3, 4, 6, 4, 5, 2, 3, 13, 14, 4, 3, 1, 7, 4, 1, 5, 5, 4, 4, 4, 14, 4, 4, 2, 5, 14, 3, 14, 14, 3, 4, 9, 5, 4, 3, 6, 4, 5, 9, 13, 1, 4, 2, 1, 1, 2, 6, 1, 7, 14, 3, 5, 4, 5, 10, 7, 3, 1, 4, 4, 12, 5, 14, 7, 5, 3, 13, 1, 4, 2, 3, 14, 1, 14, 13, 6, 3, 2, 8, 3, 5, 3, 1, 6, 3, 13, 11, 3, 3, 11, 4, 1, 3, 6, 1, 9, 8, 4, 2, 7, 14, 3, 12, 3, 4, 11, 2, 11, 2, 2, 9, 4, 14, 9, 2, 3, 7, 6, 2, 14, 12, 3, 3, 3, 7, 4, 1, 8, 3, 7, 4, 3, 5, 3, 2, 2, 3, 3, 5, 10, 1, 4, 4, 14, 3, 12, 12, 7, 8, 3, 7, 4, 3, 6, 2, 1, 3, 2, 5, 6, 3, 4, 3, 1, 3, 3, 2, 1, 5, 4, 7, 12, 3, 2, 13, 8, 1, 14, 13, 2, 4, 7, 4, 5, 14, 3, 14, 4, 5, 13, 10, 14, 1, 14, 4, 4, 3, 5, 13, 1, 6, 8, 14, 5, 3, 3, 2, 14, 1, 2, 7, 8, 3, 14, 7, 7, 11, 2, 2, 13, 7, 7, 1, 8, 3, 14, 14, 2, 2, 2, 1, 14, 3, 3, 6, 4, 4, 7, 5, 2, 3, 2, 2, 13, 3, 4, 3, 2, 6, 3, 4, 14, 2, 14, 5, 1, 1, 2, 14, 4, 14, 2, 4, 4, 1, 6, 5, 6, 6, 6, 6, 9, 7, 10, 9, 7, 8, 14, 6, 10, 8, 9, 9, 8, 7, 7, 7, 14, 8, 14, 8, 8, 9, 10, 10, 10, 11, 11, 11, 11, 13, 12, 13, 14, 12, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 6, 7, 2, 14, 9, 1, 2, 13, 13, 2, 4, 9, 3, 3, 5, 1, 3, 11, 2, 9, 2, 8, 3, 2, 6, 1, 2, 1, 3, 5, 7, 3, 6, 2, 14, 4, 5, 1, 1, 5, 1, 1, 4, 8, 7, 9, 2, 11, 1, 1, 1, 6, 6, 1, 1, 2, 4, 1, 5, 10, 3, 2, 1, 3, 1, 4, 1, 4, 4, 1, 1, 1, 5, 1, 1, 1, 1, 2, 1, 10, 4, 3, 2, 6, 3, 6, 1, 1, 2, 1, 6, 2, 3, 3, 6, 1, 14, 9, 2, 1, 3, 6, 8, 5, 5, 4, 3, 14, 14, 8, 1, 3, 7, 3, 7, 5, 8, 4, 3, 2, 1, 13, 3, 12, 14, 5, 3, 4, 5, 2, 5, 1, 5, 14, 5, 1, 1, 2, 3, 4, 4, 7, 5, 9, 1, 11, 14, 2, 2, 1, 7, 12, 8, 1, 1, 1, 1, 2, 1, 10, 5, 1, 6, 13, 13, 1, 8, 8, 4, 2, 2, 6, 1, 1, 1, 2, 3, 2, 3, 5, 5, 11, 10, 4, 14, 3, 6, 12, 5, 4, 14, 5, 14, 11, 9, 3, 2, 14, 9, 3, 7, 2, 4, 3, 8, 11, 6, 1, 6, 1, 5, 5, 1, 2, 5, 11, 1, 4, 13, 4, 7, 1, 6, 1, 1, 3, 4, 1, 3, 5, 1, 4, 5, 5, 2, 9, 4, 5, 5, 7, 2, 4, 5, 6, 2, 1, 2, 3, 4, 8, 4, 5, 5, 5, 8, 8, 9, 11, 12, 13, 13, 14, 14, 14, 14, 14, 14, 14, 1, 8, 8, 9, 8, 4, 7, 14, 5, 1, 11, 10, 3, 1, 5, 3, 6, 1, 8, 14, 3, 5, 4, 7, 14, 2, 1, 7, 10, 5, 11, 9, 5, 7, 7, 1, 10, 1, 1, 7, 7, 5, 7, 8, 3, 7, 1, 11, 13, 1, 2, 11, 8, 5, 8, 8, 14, 4, 13, 1, 6, 11, 1, 1, 3, 2, 1, 13, 7, 7, 6, 10, 12, 7, 12, 7, 1, 8, 3, 7, 7, 7, 1, 4, 2, 3, 5, 8, 10, 7, 2, 1, 7, 8, 5, 11, 4, 1, 7, 10, 3, 6, 6, 8, 1, 5, 5, 13, 4, 1, 2, 14, 11, 7, 4, 6, 9, 4, 12, 8, 14, 4, 5, 3, 10, 3, 11, 11, 14, 5, 10, 10, 10, 4, 7, 8, 7, 7, 3, 14, 7, 6, 6, 3, 5, 1, 8, 6, 1, 1, 8, 8, 11, 6, 13, 7, 5, 1, 8, 10, 5, 9, 1, 2, 7, 6, 7, 11, 14, 7, 1, 8, 6, 13, 14, 11, 8, 7, 11, 2, 14, 3, 10, 10, 9, 3, 1, 8, 9, 1, 7, 2, 1, 14, 14, 7, 13, 1, 9, 2, 1, 2, 3, 1, 10, 8, 1, 3, 7, 14, 7, 4, 2, 4, 2, 7, 9, 9, 4, 1, 2, 13, 12, 7, 6, 4, 2, 2, 7, 14, 12, 10, 5, 10, 8, 8, 2, 14, 14, 14, 1, 4, 1, 7, 7, 11, 13, 2, 8, 6, 2, 8, 2, 5, 2, 7, 6, 11, 4, 6, 13, 8, 11, 12, 7, 4, 8, 3, 3, 1, 2, 4, 4, 14, 14, 2, 8, 2, 14, 12, 14, 7, 12, 9, 1, 5, 2, 2, 5, 8, 11, 12, 2, 2, 1, 14, 2, 1, 1, 12, 1, 9, 6, 1, 1, 2, 8, 7, 2, 3, 14, 3, 11, 11, 12, 11, 12, 14, 13, 13, 12, 14, 13, 13, 14, 14, 14, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 6, 1, 1, 14, 6, 6, 6, 1, 1, 1, 7, 14, 14, 5, 3, 8, 6, 8, 6, 3, 7, 4, 6, 6, 13, 6, 6, 6, 14, 14, 6, 4, 4, 6, 14, 1, 6, 5, 10, 6, 3, 6, 6, 6, 6, 6, 9, 5, 5, 6, 6, 6, 1, 5, 6, 6, 1, 6, 8, 1, 6, 4, 3, 7, 6, 8, 1, 2, 3, 5, 1, 6, 7, 2, 6, 11, 1, 6, 7, 4, 12, 14, 14, 6, 1, 10, 2, 1, 6, 6, 5, 5, 6, 3, 6, 6, 5, 6, 1, 6, 6, 6, 1, 5, 6, 5, 4, 2, 5, 6, 1, 3, 7, 6, 1, 14, 12, 12, 4, 2, 6, 4, 11, 14, 6, 4, 2, 1, 6, 6, 9, 8, 3, 5, 1, 3, 2, 6, 12, 14, 7, 6, 8, 14, 6, 6, 10, 1, 14, 6, 6, 4, 4, 4, 6, 13, 14, 6, 14, 9, 13, 1, 6, 11, 6, 14, 2, 2, 6, 8, 1, 1, 1, 8, 7, 1, 6, 1, 6, 14, 6, 1, 13, 5, 14, 5, 14, 8, 1, 6, 5, 1, 10, 6, 6, 2, 3, 1, 5, 1, 1, 7, 6, 6, 1, 2, 2, 6, 6, 2, 1, 1, 10, 5, 14, 6, 4, 6, 4, 12, 10, 7, 7, 1, 2, 6, 6, 1, 1, 8, 6, 2, 6, 12, 12, 6, 6, 6, 6, 7, 13, 2, 1, 7, 11, 7, 6, 2, 6, 1, 8, 6, 7, 8, 14, 3, 9, 1, 3, 6, 8, 2, 1, 4, 1, 8, 8, 2, 6, 4, 3, 2, 6, 14, 6, 2, 9, 7, 2, 6, 2, 6, 6, 9, 4, 6, 7, 1, 6, 5, 2, 14, 6, 6, 9, 12, 1, 13, 3, 6, 14, 2, 2, 6, 6, 1, 1, 1, 1, 1, 13, 6, 8, 1, 2, 1, 14, 8, 1, 3, 5, 8, 7, 7, 8, 8, 8, 9, 9, 14, 10, 12, 6, 14, 14, 13, 13, 14, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 1, 9, 14, 2, 12, 2, 2, 2, 9, 9, 4, 10, 5, 2, 3, 14, 4, 4, 3, 5, 1, 2, 1, 2, 2, 1, 4, 9, 3, 2, 3, 5, 1, 1, 2, 4, 9, 2, 2, 11, 2, 1, 2, 2, 4, 1, 7, 1, 5, 5, 2, 1, 1, 14, 11, 14, 2, 14, 1, 2, 5, 5, 14, 1, 2, 2, 2, 1, 2, 2, 14, 10, 5, 2, 6, 1, 14, 2, 1, 3, 2, 2, 2, 14, 2, 1, 5, 1, 1, 5, 2, 1, 2, 2, 2, 2, 14, 2, 1, 5, 3, 1, 2, 2, 2, 8, 1, 4, 5, 14, 3, 3, 1, 9, 1, 4, 1, 1, 1, 14, 5, 7, 14, 3, 14, 14, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 3, 3, 13, 7, 3, 1, 6, 4, 1, 2, 1, 2, 14, 1, 1, 1, 2, 1, 7, 2, 12, 9, 1, 5, 1, 1, 8, 1, 4, 2, 6, 7, 6, 1, 14, 2, 1, 2, 14, 1, 1, 2, 1, 2, 5, 1, 6, 1, 1, 1, 7, 5, 6, 10, 1, 13, 1, 7, 9, 2, 11, 5, 7, 5, 7, 3, 8, 1, 7, 1, 8, 3, 2, 2, 6, 1, 1, 1, 2, 2, 4, 1, 6, 7, 2, 1, 1, 14, 1, 7, 7, 1, 6, 3, 3, 1, 4, 1, 13, 1, 1, 1, 1, 3, 1, 2, 14, 2, 14, 3, 2, 1, 1, 6, 1, 2, 12, 5, 4, 14, 14, 14, 3, 4, 14, 3, 1, 14, 2, 14, 14, 14, 14, 14, 6, 6, 3, 1, 2, 6, 2, 1, 14, 2, 4, 2, 1, 9, 1, 2, 6, 1, 5, 4, 7, 1, 1, 2, 5, 6, 8, 1, 2, 7, 7, 3, 3, 5, 6, 6, 6, 6, 14, 7, 8, 9, 9, 10, 11, 13, 13, 12, 13, 14, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 2, 7, 1, 2, 1, 3, 6, 1, 6, 2, 14, 7, 2, 8, 4, 3, 1, 4, 3, 1, 9, 1, 14, 2, 2, 4, 2, 4, 6, 2, 3, 2, 1, 2, 6, 2, 4, 3, 14, 7, 1, 1, 3, 10, 12, 1, 1, 6, 1, 7, 4, 2, 3, 6, 1, 1, 1, 2, 2, 2, 6, 3, 1, 2, 1, 2, 14, 6, 2, 7, 14, 2, 1, 2,
    #  3, 3, 5, 2, 4, 2, 1, 1, 3, 14, 2, 1, 5, 2, 3, 3, 2, 2, 1, 2, 3, 4, 2, 1, 3, 2, 2, 3, 14, 2, 14, 1, 6, 3, 2, 8, 14, 14, 4, 4, 2, 8, 5, 1, 3, 1, 10, 13, 3, 14, 2, 3, 6, 9, 1, 12, 1, 2, 13, 14, 14, 4, 5, 1, 2, 2, 3, 3, 3, 3, 2, 4, 3, 5, 1, 7, 1, 13, 6, 6, 2, 7, 7, 1, 1, 2, 1, 14, 2, 9, 4, 5, 14, 1, 10, 4, 5, 2, 2, 1, 3, 1, 1, 14, 10, 2, 2, 3, 4, 2, 1, 4, 3, 3, 3, 3, 5, 3, 5, 2, 7, 14, 1, 5, 3, 5, 4, 4, 5, 4, 3, 3, 2, 3, 5, 2, 3, 5, 2, 3, 8, 1, 3, 1, 5, 14, 14, 14, 14, 14, 6, 3, 6, 6, 2, 10, 5, 3, 5, 5, 2, 14, 4, 6, 3, 3, 3, 3, 6, 1, 1, 2, 4, 13, 3, 2, 3, 1, 3, 13, 4, 14, 12, 12, 6, 11, 6, 11, 4, 3, 14, 6, 4, 6, 2, 1, 5, 14, 14, 2, 1, 8, 9, 11, 1, 2, 14, 3, 6, 1, 3, 2, 9, 14, 12, 3, 4, 9, 5, 5, 14, 5, 6, 6, 6, 8, 7, 9, 9, 11, 13, 12, 12, 12, 12, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 3, 6, 2, 8, 3, 7, 1, 2, 6, 7, 7, 2, 5, 1, 10, 2, 6, 2, 14, 2, 7, 3, 14, 1, 3, 3, 14, 1, 9, 5, 3, 5, 1, 5, 3, 1, 13, 2, 2, 2, 7, 3, 2, 6, 4, 1, 11, 2, 6, 5, 1, 7, 8, 1, 2, 3, 4, 3, 3, 4, 4, 10, 2, 2, 10, 2, 2, 14, 5, 3, 3, 1, 3, 2, 4, 14, 1, 5, 14, 7, 6, 1, 2, 13, 3, 6, 3, 2, 2, 7, 9, 3, 1, 4, 2, 5, 4, 13, 3, 3, 2, 2, 1, 1, 7, 2, 14, 1, 5, 2, 2, 5, 2, 3, 3, 10, 10, 5, 7, 14, 1, 1, 3, 14, 14, 2, 2, 2, 7, 5, 3, 5, 14, 4, 1, 4, 14, 4, 2, 2, 2, 11, 3, 2, 3, 1, 1, 1, 1, 1, 4, 14, 1, 1, 9, 8, 11, 1, 14, 4, 3, 4, 2, 3, 12, 8, 3, 6, 10, 6, 12, 12, 9, 14, 14, 10, 1, 2, 3, 4, 1, 14, 6, 2, 14, 11, 14, 1, 4, 2, 10, 9, 4, 11, 5, 11, 3, 3, 1, 4, 1, 3, 3, 14, 12, 1, 1, 5, 6, 4, 7, 1, 1, 4, 2, 1, 1, 2, 1, 6, 1, 14, 1, 14, 1, 1, 2, 2, 2, 2, 4, 3, 11, 2, 6, 14, 9, 14, 14, 5, 14, 7, 10, 13, 14, 14, 14, 14, 14, 14, 14]
    # with open(os.path.join(PATH,"result_data.txt"),'a') as f:
    #     f.write(str(result_fin))   

    # result_data = np.loadtxt("result_data.txt")  
    # print(result_data) 


    Day_start = 'Sep 1, 2019'
    start_day = "2019-09-01"
    logfile = "handle-log/flaskapp.log.{}."
    #logfile = os.path.join(PATH,"handle-log/flaskapp.log.{}.".format(start_day))
    #day = "2019-09-01"

    # file = sys.argv[1]
    # file = file[:len(file) - 1]
    #
    # day = sys.argv[2]
    
    print_iter = 1
    serve_iter = 2000
    save_iter = 6
    decay_iter = 1000
    batch_size =128
    train_break_sum = 8
    epoches =1
    version =0
    curentday = start_day
    nowday = utils.get_now_day_fmt()
    version = utils.get_max_serve_num(SERVE_PATH)

    mol = model_sum_34.SimpleModel()

    with tf.Session(graph=mol.graph) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())


        day_nums = (version-1) 
        
        startday = datetime.datetime.strptime(Day_start, '%b %d, %Y')
        iterday =  startday+ datetime.timedelta(days=day_nums) 

        nowday = utils.get_now_day_fmt()

        result =[]

         
        while int(''.join(curentday.split('-'))) < int(''.join(nowday.split('-'))):
    
            
            if int(''.join(curentday.split('-'))) >= int(''.join("2019-11-06".split('-'))):
                logging.info("****Day_nums-1 == index {}****".format(curentday ))
                break
            if int(''.join(curentday.split('-'))) == int(''.join("2019-10-27".split('-'))):
                curentday = datetime.datetime.strptime(curentday,"%Y-%m-%d") + datetime.timedelta(days=1)
                curentday = curentday.strftime("%Y-%m-%d")
                continue 
           

            cnt = 0
            test_data_dict ={}

            try:
                applog = os.path.join(PATH,logfile.format(curentday))
                viewclickfile = handle_log.log(applog, curentday)
    
                #从两个TXT  提取 csv
                test_base_path = handle_train.generate_train_base(curentday,viewclickfile)
                #train_base = os.path.join(PATH,"train_base/train_base--2019-09-01.csv")

                #  结合 mysql数据中 储存的数据，出  训练数据
                    # maybe we can read ourselves to avoid OOM error!
  
                test_base = pd.read_csv(test_base_path)
                #训练 有drop
                test_base.drop(labels=["time", "rank", "tag", "cnt"], axis=1, inplace=True)
                # print(len(train_base))
                # """to test"""
                # train_base = train_base[:1]

                #def proc_test(train_base):
                assert len(test_base) > 0, "len(train_base) must >0!"
            

                with feature.get_coursor("recom_turing", "dict") as f:
                    with feature.get_coursor("recom", "dict") as ff:
                        
                        data_tmp = pd.DataFrame()
                        #applog = os.path.join(PATH,logfile.format(testday))

                        tmp = test_base.groupby(["teacher_id", "class_id", "day", "chapter_id"])
                        for dd in tmp:
                            #保证这一组里面有正样本,, 肯定也有负样本
                            if dd[1]["label"].sum() > 0:
                                data_tmp = pd.concat([data_tmp, dd[1]], ignore_index=True, sort=False)
                        #print(data_tmp['label'])
                        data_gb = data_tmp.groupby(by=["teacher_id", "class_id", "day", "chapter_id"])
                        #遍历，每组唯一的  
                        
                        for item in data_gb:

                            teacher_id, class_id, day, chapter_id = item[0]
                            #print("chapter_id:" ,chapter_id
                            #
                            aim = item[1]
                            for row in aim.itertuples(index=False):
                                if getattr(row, "label") == 1:
                                    aa = {"base_chapters": getattr(row, "base_chapters"),
                                        "base_sections": getattr(row, "base_sections"),
                                        }
                                    break
                            rst =0
                            last_year_cha = get_lastyear_data( ff,f, teacher_id,class_id)
                            #print(last_year_cha)
                            if last_year_cha == 0:
                                rst = -1
                            else:
                                rst = simplity_cha(last_year_cha,aa)
                            print(rst)
                            result.append(rst)



            except Exception as e:
                print(e,)
                logging.debug("test cnt {}\n,error {}".format(cnt,e))
                logging.debug("execept e :{}".format(traceback.format_exc()))

            curentday = datetime.datetime.strptime(curentday,"%Y-%m-%d") + datetime.timedelta(days=5)
            curentday = curentday.strftime("%Y-%m-%d")
            if int(''.join(nowday.split('-'))) != int(''.join(utils.get_now_day_fmt().split('-'))):
                nowday = utils.get_now_day_fmt()   

        print(result)  
        with open(os.path.join(PATH,"result_data16.txt"),'a') as f:
            f.write(str(result))   

    


'''
一天一个csv  groupby  class_id  拿历史作业cha   
一天很多个样本，每个样本对应 一个14天的数据，
9.1   9.10   9.20  9.30  10.5  10.15  10.25  11.6 
总共多少样本，每个样本[[],[],[],[]]
遍历，每个14天

3个连续0 
5个连续0
10个连续0
14个连续0

直方统计图
'''
                        