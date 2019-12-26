
import os 
import handle_log
import handle_train 
import feature 
from utils import *
from feature import *
import copy 
import numpy as np 
import tensorflow as tf 
import tensorflow.python.saved_model import tag_constants

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

 
    his_days =['one', 'two', 'three', 'four','five','six','seven','eight','nine',
    'ten','eleven','twelve','thirteen','fourteen']
    chap_ = list(map(lambda x: "history_" + x + "_chap_ph", his_days))
    #改动
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

    #改动
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
    
    #先保证列等长 占位符，  所有样本的第一天， 用最后一位 填充，同时 做映射
    # 在 一个样本的所有天，，中间缺失天的  repeat  在重新赋值
    sample_nums =len(resp["history_one_chap_ph"])
    for i in range(sample_nums):
        cha_days=[]
        for item in chap_:
            tmp = resp[item][i]
            cha_days.append(tmp)
        cha_days_aim = repeat(cha_days)
        
        num=0
        for item in chap_:
            resp[item][i] = cha_days_aim[num]
            num+=1

    for i in range(sample_nums):
        sec_days=[]
        for item in sec_:
            tmp = resp[item][i]
            sec_days.append(tmp)
        sec_days_aim = repeat(sec_days)
        
        num=0
        for item in sec_:
            resp[item][i] = sec_days_aim[num]
            num+=1


    #改动
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
                #改动
                rst[k] = v + [v[-1]] * (length - len(v))
            resp[item] = rst
        
     #连续值 
    today_style_ph = ["today_style_ph"]
    for item in today_style_ph:
        resp[item] = [STYLE_MAP.get(i[item], 0) for i in data]
    
    #
    style_ph = ["style_ph"]
    for item in style_ph:
        for s in STYLE:
            resp["style_" + s + "_ph"] = [i[item][s] for i in data]

    return resp


HIS_DAYS_NUM=14

def online_data(teacher_id, class_id, day, chapter_id ,
     grade,editionId,hisHomeWork,):
    
    
    test_batch = []

    with get_coursor("recom_turing", "dict") as f:
        with get_coursor("recom", "dict") as ff:
            
            
            teacher_info = teacher(f, teacher_id)
            class_info = clazz(f, class_id)
            study_info = study(f, class_id, day)

            submit_info = submit(f,class_id, day)
            capacity_info = capacity(f, teacher_id)
            prefer_info = prefer(f, teacher_id, day)           
            register_info = register(f, teacher_id, day)
            homework_info = homework(ff, teacher_id, day)
            style_info = style(ff, teacher_id,day)
            week_info = week(ff, teacher_id, day)
            reflect_info = reflect(ff, class_id,day)
            last_day_info = last_day(ff, class_id, day)
            

            test_data_dict= {}

            info_off_fea = [teacher_info,class_info,submit_info,capacity_info,prefer_info,
            register_info,homework_info,week_info,reflect_info,last_day_info,study_info,style_info ]

            for i in range(len(info_off_fea)):
                test_data_dict.update(info_off_fea[i])

            for i in range(HIS_DAYS_NUM):
                #his_info = 'yes' + str(i+1) +'_info'
                his_info = yesterday(ff, class_id, day, i+1)
                test_data_dict.update(his_info)
            
            cand_nums = len(hisHomeWork)
            
            
            
            for i  in  range(cand_nums):
                today_dict= {}
                today_dict["today_chapters_ph"] = hisHomeWork[i]["base_chapters"]
                today_dict["today_sections_ph"] = hisHomeWork[i]["base_sections"]
                today_dict["today_style_ph"] = hisHomeWork[i]["style"]
                
                test_data_dict_ = test_data_dict.copy()
                test_data_dict_.update(today_dict)
                
                test_batch.append(test_data_dict_)
                
                        
    test_data_final = prepare_data(test_batch)
           
    feed_dict = {}
    for k, v in test_data_final.items():
        feed_dict[k] = v
  
    return feed_dict                      


def  load_model():


    saved_model_dir = "save_model"
    #signature_key = 'test_signature'
    input_key = 'input_x'
    output_key = 'outputs'
    
    signature_key = tf.saved_model.tag_constants.SERVING

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:

        meta_graph_def = tf.saved_model.loader.load(sess, [signature_key], saved_model_dir)
        signature = meta_graph_def.signature_def
        x_tensor_name = signature[signature_key].inputs[input_key].name
        y_tensor_name = signature[signature_key].outputs[output_key].name
        print(x_tensor_name)
        print(y_tensor_name)
        x = sess.graph.get_tensor_by_name(x_tensor_name)
        y = sess.graph.get_tensor_by_name(y_tensor_name)
        feed_dict = {x: np.ones([1, 10])}
        print(sess.run(y, feed_dict=feed_dict))




if __name__ == "__main__":

    test = {"requestId":"requestId","grade":6,"editionId":75,"teacherId":13267,
    "classId":2502867,"chapterId":56589,"day":"2019-01-18",
    "hisHomeWork":[{"base_chapters": "56609, 56607","style": 30110,
    "base_sections": "56608, 56610, 56611, 92473, 92474, 92475, 92476, 92477"}, 
    {"base_chapters": "56609, 56607","style": 30100,
    "base_sections": "56608, 56610, 92473, 92474, 92475, 92476"}]}

    onlineda = online_data(test["teacherId"],test["classId"],test["day"],test["chapterId"],
    test["grade"],test["editionId"],test["hisHomeWork"])

    print(onlineda)

    {'teacher_id_ph': [13267, 13267], 
    'student_count_ph': [150, 150], 
    'province_id_ph': [25, 25], 
    'city_id_ph': [25, 25], 
    'core_type_ph': [1, 1], 
    'class_id_ph': [2502867, 2502867],
     'edition_id_ph': [137, 137], 
     'grade_id_ph': [3, 3], 
     'class_student_ph': [99, 99], 
     'cap_avg_ph': [0.6, 0.6], 
     'cap_max_ph': [2.42, 2.42], 'cap_min_ph': [-0.83, -0.83], 'gap_days_ph': [15, 15], 
     'month_submit_rate_ph': [0.95, 0.95], 
     'region_capacity_ph': [0.09, 0.09], 
     'prefer_assign_time_avg_ph': [9.0, 9.0], 
     'prefer_assign_time_var_ph': [6.59, 6.59], 
     'prefer_assign_rank_avg_ph': [24.0, 24.0], 
     'prefer_assign_rank_var_ph': [0.0, 0.0], 
     'register_diff_ph': [1186, 1186], 'homework_count_ph': [375, 375], 
     'week_count_ph': [4.75, 4.75], 'lastday_count_ph': [1, 1], 
     'study_vector_ph': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
     'today_chapters_ph': [[174, 173], [174, 173]], 
     'today_sections_ph': [[611, 612, 613, 3753, 3754, 3755, 3756, 3757], [611, 612, 3753, 3754, 3755, 3756, 3756, 3756]], 
     'history_one_chap_ph': [[170], [170]], 'history_two_chap_ph': [[170], [170]], 'history_three_chap_ph': [[170], [170]], 'history_four_chap_ph': [[170], [170]], 
     'history_five_chap_ph': [[170], [170]], 'history_six_chap_ph': [[170, 979], [170, 979]], 'history_seven_chap_ph': [[170, 979, 172], [170, 979, 172]], 
     'history_eight_chap_ph': [[170, 979, 172], [170, 979, 172]], 'history_nine_chap_ph': [[170, 171, 979, 172], [170, 171, 979, 172]], 'history_ten_chap_ph': [[170], [170]], 
     'history_eleven_chap_ph': [[170], [170]], 'history_twelve_chap_ph': [[170, 979], [170, 979]], 'history_thirteen_chap_ph': [[170, 979, 168], [170, 979, 168]], 'history_fourteen_chap_ph': [[170, 979], [170, 979]], 
     'history_one_sec_ph': [[4006], [4006]], 'history_two_sec_ph': [[4006], [4006]], 'history_three_sec_ph': [[4006], [4006]], 'history_four_sec_ph': [[4006], [4006]], 'history_five_sec_ph': [[4006], [4006]], 'history_six_sec_ph': [[4006, 3996, 4004, 3998], [4006, 3996, 4004, 3998]], 'history_seven_sec_ph': [[3725, 3996, 4006], [3725, 3996, 4006]], 'history_eight_sec_ph': [[3724, 3996, 3997, 3998, 3999, 4006], [3724, 3996, 3997, 3998, 3999, 4006]], 'history_nine_sec_ph': [[4007, 4008, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 3724, 3725, 3726, 3727, 3728, 3996, 3997, 3998, 3999, 4003, 4004, 4005, 4006], [4007, 4008, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 3724, 3725, 3726, 3727, 3728, 3996, 3997, 3998, 3999, 4003, 4004, 4005, 4006]], 'history_ten_sec_ph': [[4007], [4007]], 
     'history_eleven_sec_ph': [[4007], [4007]], 'history_twelve_sec_ph': [[606, 3996, 3997, 3998, 4003, 4005, 4006], [606, 3996, 3997, 3998, 4003, 4005, 4006]], 'history_thirteen_sec_ph': [[4006, 606, 4002, 3996, 3998], [4006, 606, 4002, 3996, 3998]], 'history_fourteen_sec_ph': [[4006, 3996, 3998], [4006, 3996, 3998]], 'reflect_value_ph': [[0], [0]], 'today_style_ph': [18, 15], 'style_10100_ph': [0, 0], 'style_10010_ph': [0, 0], 'style_10001_ph': [0, 0], 'style_10110_ph': [1, 1], 'style_10101_ph': [0, 0], 'style_10011_ph': [0, 0], 'style_10111_ph': [0, 0], 'style_20100_ph': [0, 0], 'style_20010_ph': [0, 0], 'style_20001_ph': [0, 0], 'style_20110_ph': [0, 0], 'style_20101_ph': [0, 0], 'style_20011_ph': [0, 0], 'style_20111_ph': [0, 0], 'style_30100_ph': [1, 1], 'style_30010_ph': [6, 6], 'style_30001_ph': [0, 0], 'style_30110_ph': [0, 0], 'style_30101_ph': [0, 0], 'style_30011_ph': [0, 0], 'style_30111_ph': [0, 0], 'style_40100_ph': [1, 1], 'style_40010_ph': [8, 8], 'style_40001_ph': [0, 0], 'style_40110_ph': [1, 1], 'style_40101_ph': [0, 0], 'style_40011_ph': [0, 0], 'style_40111_ph': [0, 0]}