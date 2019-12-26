import sys, time, os
import tensorflow as tf

import pandas as pd
import numpy as np 
import utils
import csv
import logging
import os 
import datetime
import time
import handle_log
import feature
import handle_train
import model_bgru_online
import traceback
import json
import copy
from transport_model import trans_model

PATH = os.path.dirname(os.path.abspath(__file__))
flags = tf.app.flags

flags.DEFINE_string("logfile",'logs.bgru.online.out',"log file to save")

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


BASE_PATH = os.path.join(PATH, "update-model-1")
MODLE_PATH = os.path.join(BASE_PATH, "model")
SERVE_PATH = os.path.join(BASE_PATH, "serving")
os.makedirs(MODLE_PATH, exist_ok=True)
os.makedirs(SERVE_PATH, exist_ok=True)



def repeat(ww,gram=5):

    vv = copy.deepcopy(ww)
    #看前面
    for i  in range(len(ww)):
        one = ww[i]
        one_np= np.array(one)
        len1 = len(one)
        len2 =0
        
        if not ((one_np ==0).all()):
            #j =i+1 
            for j in range(i+1,len(ww)):
                tmp = ww[j]
                len2 = len(tmp)
                if not ((np.array(tmp) ==0).all()):
                    break    
                else:
                    if j -i <=gram:
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
       
        if not ((one_np ==0).all()):
            for j in range(i):
                if j >=i-gram:
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
 
            aim = [to_int(str(i[item]).strip().split(",")) for i in data]

            rst = copy.deepcopy(aim)
            length = max([len(i) for i in aim])

            for i in range(len(aim)):  
                for j in range(len(aim[i])): 
                    rst[i][j]=cha[str(aim[i][j])]

            for k, v in enumerate(rst):
                
                rst[k] = v + [v[-1]] * (length - len(v))
            resp[item] = rst

    sec_ = list(map(lambda x: "history_" + x + "_sec_ph", his_days))

    
    with open(sec_file,'r') as jf:
        sec = json.loads(jf.read())
        for item in sec_:

            aim = [to_int(str(i[item]).strip().split(",")) for i in data]

            rst = copy.deepcopy(aim)
            length = max([len(i) for i in aim])

            for i in range(len(aim)):

                for j in range(len(aim[i])): 
                    rst[i][j]=sec[str(aim[i][j])]

            for k, v in enumerate(rst):
                rst[k] = v + [v[-1]] * (length - len(v))
            resp[item] = rst

    
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
        
    today_style_ph = ["today_style_ph"]
    for item in today_style_ph:
        resp[item] = [STYLE_MAP.get(i[item], 0) for i in data]
    
    #
    style_ph = ["style_ph"]
    for item in style_ph:
        for s in STYLE:
            resp["style_" + s + "_ph"] = [i[item][s] for i in data]

    item = "label"
    resp["target_ph"] = [[0, 1] if i[item] == 1 else [1, 0] for i in data]

    return resp






if __name__ == "__main__":
    
              
    #WATCH_PATH = "/data/lishuang/handle-log/log"
    logfile = "/data/lishuang/handle-log/log/flaskapp.log.{}."
    #logfile = "/data/zhangkl/turing_new/handle-log/flaskapp.log.{}."
    
    #day = "2019-09-01"

    # file = sys.argv[1]
    # file = file[:len(file) - 1]
    
    nowday = utils.get_now_day_fmt()
    lastday = datetime.datetime.strptime(nowday,"%Y-%m-%d") + datetime.timedelta(days=-1)
    lastday = lastday.strftime("%Y-%m-%d")

    
    print_iter = 1

    decay_iter = 200
    batch_size =128
    epoches =1
    version =0

    curentday = lastday
   
    version = utils.get_max_serve_num(SERVE_PATH)

    mol = model_bgru_online.SimpleModel()

    with tf.Session(graph=mol.graph) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        mol.restore(sess, os.path.join(MODLE_PATH, "ckpt_") + str(version))
       
        try:
            iiter =0
            #applog = os.path.join(PATH,logfile.format(curentday))
            applog = logfile.format(curentday)

            viewclickfile = handle_log.log(applog, curentday)

            train_base = handle_train.generate_train_base(curentday,viewclickfile)

            train_data = feature.get_data_with_pandas(train_base, batch_size)
                 
            lr = 0.001
            loss_sum = 0.0
            accuracy_sum = 0.0 
            
            if version % decay_iter == 0:
                lr *= 0.8
                if lr < 10e-7:
                    lr = 0.0001

            for i in range(epoches):
                for data_pre in train_data:

                    data_ = prepare_data(data_pre)
                    data_["lr_ph"] = lr

                    loss, acc = mol.train_with_dict(sess, data_)
                                            
                    loss_sum += loss
                    accuracy_sum += acc

                    iiter += 1
                    if iiter % print_iter == 0:
                        #print(curentday,iiter, loss, acc)
                        logging.info("---train--- day:{}, iter: {},loss_average:{}, accuracy_average:{},loss:{},acc:{}".format(
                        curentday,iiter,
                                loss_sum / iiter, accuracy_sum / iiter,loss, acc))
                    

            version += 1
            mol.save(sess, os.path.join(MODLE_PATH, "ckpt_") + str(version))                
            mol.save_serving_model(sess, SERVE_PATH, str(version))

            """trans model !!!"""
            logging.info("--start trans_model--")
            trans_model(version)
      

        except Exception as e:
            #print(e,)
            logging.debug("test cnt {}\n,error {}".format(iiter,e))
            logging.debug("execept e :{}".format(traceback.format_exc()))
