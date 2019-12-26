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
import model_sum_61
import traceback
import copy

PATH = os.path.dirname(os.path.abspath(__file__))
flags = tf.app.flags

flags.DEFINE_string("logfile",'logs.sum.61.out',"log file to save")
flags.DEFINE_string("csvfile","test_metric_day.sum.61.csv",'csv file to save test metric')


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

RST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_sum_60")
BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save_sum_61")
MODLE_PATH = os.path.join(BASE_PATH, "model")
SERVE_PATH = os.path.join(BASE_PATH, "serving")
os.makedirs(MODLE_PATH, exist_ok=True)
os.makedirs(SERVE_PATH, exist_ok=True)
os.makedirs(RST_PATH, exist_ok=True)

HIS_DAYS_NUM = 14


def repeat(ww,gram=7):

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


if __name__ == "__main__":
            

    Day_start = 'Sep 2, 2019'
    start_day = "2019-09-02"
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
    epoches =2
    version =0
    curentday = start_day
    nowday = utils.get_now_day_fmt()
    version = utils.get_max_serve_num(SERVE_PATH)

    mol = model_sum_61.SimpleModel()

    with tf.Session(graph=mol.graph) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())


        day_nums = (version-1) 
        
        startday = datetime.datetime.strptime(Day_start, '%b %d, %Y')
        iterday =  startday+ datetime.timedelta(days=day_nums) 

        nowday = utils.get_now_day_fmt()
        if int(''.join(iterday.strftime("%Y-%m-%d").split('-') )) > int(''.join(start_day.split('-'))):    
            #model.restore(sess, os.path.join(PATH, model_path) + str(version))
            mol.restore(sess, os.path.join(MODLE_PATH, "ckpt_") + str(version))
            curentday = iterday.strftime("%Y-%m-%d")
        
    
        while int(''.join(curentday.split('-'))) < int(''.join(nowday.split('-'))):
    
            
            if int(''.join(curentday.split('-'))) >= int(''.join("2019-11-08".split('-'))):
                logging.info("****Day_nums-1 == index {}****".format(curentday ))
                break
            if int(''.join(curentday.split('-'))) == int(''.join("2019-10-27".split('-'))):
                curentday = datetime.datetime.strptime(curentday,"%Y-%m-%d") + datetime.timedelta(days=1)
                curentday = curentday.strftime("%Y-%m-%d")
                continue 
            
            try:
                iiter =0
                #从flaskapp log日志中，.1 .2 两个文件，提取出  两个 txt
                applog = os.path.join(PATH,logfile.format(curentday))
                viewclickfile = handle_log.log(applog, curentday)

                #从两个TXT  提取 csv
                train_base = handle_train.generate_train_base(curentday,viewclickfile)
                #train_base = os.path.join(PATH,"train_base/train_base--2019-09-01.csv")

                #  结合 mysql数据中 储存的数据，出  训练数据
                train_data = feature.get_data_with_pandas(train_base, batch_size)
              
                
                lr = 0.001
                loss_sum = 0.0
                accuracy_sum = 0.0 
                break_cnt = 1
                
                flag = True
                #一天就93个数据吗，每次取一个batch
                for i in range(epoches):
                    for data_pre in train_data:
                        if len(data_pre)==0:
                            continue
                        data_ = prepare_data(data_pre)
                        data_["lr_ph"] = lr
                        
                        data_["target_ph"] = [[0, 1] if i["label"] == 1 else [1, 0] for i in data_pre]


                        loss, acc = mol.train_with_dict(sess, data_)
                                             
                        loss_sum += loss
                        accuracy_sum += acc


                        iiter += 1
                        if iiter % print_iter == 0:
                            print(curentday,iiter, loss, acc)
                            logging.info("---train--- day:{}, epoches:{} iter: {},loss_average:{}, accuracy_average:{},loss:{},acc:{}".format(
                            curentday,i,iiter,
                                    loss_sum / iiter, accuracy_sum / iiter,loss, acc))
                            
                        if iiter % decay_iter == 0:
                            lr *= 0.8

                    train_data = feature.get_data_with_pandas_lastyear(train_base, batch_size)

            except Exception as e:
                print(e,)
                logging.debug("test cnt {}\n,error {}".format(iiter,e))
                logging.debug("execept e :{}".format(traceback.format_exc()))
                #转到哪里了
                continue

            version += 1
            mol.save(sess, os.path.join(MODLE_PATH, "ckpt_") + str(version))                
            mol.save_serving_model(sess, SERVE_PATH, str(version))

            logging.info('########################### TEST ###########################')
            tmp = datetime.datetime.strptime(curentday,"%Y-%m-%d") + datetime.timedelta(days=1)
            testday = tmp.strftime("%Y-%m-%d")

            if int(''.join(testday.split('-'))) == int(''.join(nowday.split('-'))):
                logging.info("****Day_nums-1 == index {}****".format(testday ))
                break

            cnt = 0
            try:
                applog = os.path.join(PATH,logfile.format(testday))
                viewclickfile = handle_log.log(applog, testday)

                #从两个TXT  提取 csv
                test_base = handle_train.generate_train_base(testday,viewclickfile)
                #train_base = os.path.join(PATH,"train_base/train_base--2019-09-01.csv")

                #  结合 mysql数据中 储存的数据，出  训练数据
                test_data = feature.get_data_with_pandas_test(test_base)
                
                result = []
                for data_pre in test_data:
                    
                    test_batch,condicate_id,keya = data_pre
                    test_data_final = prepare_data(test_batch)
                    
                    #
                    prob2 = mol.predict_with_dict(sess, test_data_final)
                    #prob2,loss2, accuracy2= mol.calculate_with_dict(sess, test_data_final)


                    rank_ = [[i, j] for i, j in zip(prob2[0], condicate_id)]
                    #是x[0][0]  x[0][1]   第二维度排序   tgt [0,1] 为正样本
                    rank = sorted(rank_, key=lambda x: x[0][1], reverse=True)
                    
                    topk = utils.point(rank, keya)
                    result.append(topk)

                    cnt += 1
                
                    
                re = pd.DataFrame(result)
                print("test:rst len ", len(re))
                re.to_csv(os.path.join(RST_PATH,"result-%s.csv" % (testday)), index=False)


            except Exception as e:
                print(e,)
                logging.debug("test cnt {}\n,error {}".format(cnt,e))
                logging.debug("execept e :{}".format(traceback.format_exc()))


            curentday = datetime.datetime.strptime(curentday,"%Y-%m-%d") + datetime.timedelta(days=1)
            curentday = curentday.strftime("%Y-%m-%d")
            if int(''.join(nowday.split('-'))) != int(''.join(utils.get_now_day_fmt().split('-'))):
                nowday = utils.get_now_day_fmt()




                