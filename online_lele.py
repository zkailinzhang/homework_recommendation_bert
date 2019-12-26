
import os 
import handle_log
import handle_train 
import feature 
from utils import *
from feature import *
import copy 
import numpy as np 
import tensorflow as tf 
from tensorflow.python.saved_model import tag_constants

PATH = os.path.dirname(os.path.abspath(__file__))

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


def repeat(ww):

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
def prepare_base_data(data,lens):


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
        resp[key] = [data[key]]*lens 
    resp["study_vector_ph"] = [(eval(data["study_vector_ph"]))]*lens

        
    style_ph = ["style_ph"]
    for item in style_ph:
        for s in STYLE:
            resp["style_" + s + "_ph"] = [(data[item][s]) ]*lens

    return resp


def prepare_update_data(data,lens):
    def to_int(l):
        return [int(i) if i!='' else 0 for i in l]

    resp = {}

 
    his_days =['one', 'two', 'three', 'four','five','six','seven','eight','nine',
    'ten','eleven','twelve','thirteen','fourteen']
    chap_ = list(map(lambda x: "history_" + x + "_chap_ph", his_days))
    #改动
    with open(cha_file,'r') as jf:
        cha = json.loads(jf.read())
        cha_days=[]
        for item in chap_:
            aim = to_int(str(data[item]).strip().split(","))
        
            for j in range(len(aim)): 
                aim[j]=cha[str(aim[j])]

            cha_days.append([aim])

        cha_days_aim = repeat(cha_days)
        num=0
        for item in chap_:
            resp[item] = cha_days_aim[num]*lens
            num+=1

    sec_ = list(map(lambda x: "history_" + x + "_sec_ph", his_days))

    #改动
    with open(sec_file,'r') as jf:
        sec = json.loads(jf.read())
        sec_days=[]
        for item in sec_:

            aim = to_int(str(data[item]).strip().split(",")) 
            
            for j in range(len(aim)): 
                aim[j]=sec[str(aim[j])]
            sec_days.append([aim])

        sec_days_aim = repeat(sec_days)
        num=0
        for item in sec_:
            resp[item] = sec_days_aim[num]*lens
            num+=1
    

    #改动
    ref_ = ["reflect_value_ph"]
    with open(ref_file,'r') as jf:
        ref = json.loads(jf.read())
        for item in ref_:
            aim = to_int(str(data[item]).strip().split(",")) 

            for j in range(len(aim)): 
                aim[j]=ref[str(aim[j])]

            resp[item] = [aim]*lens


    return resp


def prepare_today_data(data):
    def to_int(l):
        return [int(i) if i!='' else 0 for i in l]

    resp = {}

    tchap_ = ["today_chapters_ph",  ]
    #改动
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
                #改进
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
                #改进
                rst[k] = v + [v[-1]] * (length - len(v))
            resp[item] = rst
  
     #连续值 
    today_style_ph = ["today_style_ph"]
    for item in today_style_ph:
        resp[item] = [STYLE_MAP.get(i[item], 0) for i in data]
    
 

    return resp


HIS_DAYS_NUM=14

def online_data(teacher_id, class_id, day, chapter_id ,
     grade,editionId,hisHomeWork,):
    
    
    final_rst ={}

    cand_nums = len(hisHomeWork)

    with get_coursor("recom_turing", "dict") as f:
        with get_coursor("recom", "dict") as ff:
            
            
            teacher_info = teacher(f, teacher_id)
            class_info = clazz(f, class_id)
            study_info = study(f, class_id, chapter_id)

            submit_info = submit(f,class_id, day)
            capacity_info = capacity(f, teacher_id)
            prefer_info = prefer(f, teacher_id, day)           
            register_info = register(f, teacher_id, day)
            homework_info = homework(ff, teacher_id, day)
            style_info = style(ff, teacher_id,day)
            week_info = week(ff, teacher_id, day)
            
            last_day_info = last_day(ff, class_id, day)
            

            base_data_dict= {}
            
            info_off_fea = [teacher_info,class_info,submit_info,capacity_info,prefer_info,
            register_info,homework_info,week_info,last_day_info,study_info,style_info ]
            for i in range(len(info_off_fea)):
                base_data_dict.update(info_off_fea[i])

            base_data =  prepare_base_data(base_data_dict,cand_nums)

            final_rst.update(base_data)
            
            update_data_dict= {}
            reflect_info = reflect(ff, class_id,day)
            update_data_dict.update(reflect_info)

            for i in range(HIS_DAYS_NUM):
                his_info = yesterday(ff, class_id, day, i+1)
                update_data_dict.update(his_info)    
            
            update_data =  prepare_update_data(update_data_dict,cand_nums)
            final_rst.update(update_data)

            today_conda = []
            for i  in  range(cand_nums):
                today_dict= {}
                today_dict["today_chapters_ph"] = hisHomeWork[i]["base_chapters"]
                today_dict["today_sections_ph"] = hisHomeWork[i]["base_sections"]
                today_dict["today_style_ph"] = hisHomeWork[i]["style"]
                
                today_conda.append(today_dict)


            canda = prepare_today_data(today_conda)  
            final_rst.update(canda)            
            

    return final_rst                      


def  load_model(feed ,saved_dir ):


    saved_model_dir = saved_dir
    #signature_key = 'test_signature'
    sign = 'serving'
    output_key = 'outputs'
    
    signature_key = tf.saved_model.tag_constants.SERVING

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:

        meta_graph_def = tf.saved_model.loader.load(sess, [signature_key], saved_model_dir)
        signature = meta_graph_def.signature_def

        y_tensor_name = signature[sign].outputs[output_key].name

        y = sess.graph.get_tensor_by_name(y_tensor_name)
      
        inp ={}
        for k, v in feed.items():
            tensor_name = signature[sign].inputs[k].name
            name = sess.graph.get_tensor_by_name(tensor_name)
            inp[name]=v 


        proc = sess.run(y, feed_dict=inp)

    return proc


if __name__ == "__main__":

    test = {"requestId":"requestId","grade":6,"editionId":75,"teacherId":13267,
    "classId":2502867,"chapterId":56589,"day":"2019-01-18",
    "hisHomeWork":[{"base_chapters": "56609, 56607","style": 30110,
    "base_sections": "56608, 56610, 56611, 92473, 92474, 92475, 92476, 92477"}, 
    {"base_chapters": "56609, 56607","style": 30100,
    "base_sections": "56608, 56610, 92473, 92474, 92475, 92476"}]}

    onlineda = online_data(test["teacherId"],test["classId"],test["day"],test["chapterId"],
    test["grade"],test["editionId"],test["hisHomeWork"])

    print(list(onlineda))
    model_path = os.path.join(PATH,'save_sum_26/102/')
    proc = load_model(onlineda,model_path)
    print(proc)

    # [[9.999957e-01 4.357276e-06]
    #  [9.999943e-01 5.736495e-06]]

