from utils import *
import datetime
from itertools import chain
import os, typing
import json
import traceback
import numpy as np 

LAST_DAYS =7
HIS_DAYS_NUM=14


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


# maybe cause OOM error, better to consider read ourselves
def get_data_with_pandas(train_base_path: str, batch_size: int = 128) -> typing.List:
    # maybe we can read ourselves to avoid OOM error!
        # if os.path.exists(train_base_path % (aim)) :
        # return train_base_path % (aim)
    train_base = pd.read_csv(train_base_path)

    train_base.drop(labels=["time", "rank", "tag", "cnt"], axis=1, inplace=True)
    # print(len(train_base))
    # """to test"""
    # train_base = train_base[:1]

    def proc(train_base):
        assert len(train_base) > 0, "len(train_base) must >0!"
        res = []
        #这是两个mysql表
        #曾经的
        with get_coursor("recom_turing", "dict") as f:
            with get_coursor("recom", "dict") as ff:
                #这是从日志中，解析出的csv
                #实时性的一些操作记录
                for row in train_base.itertuples(index=False):
                    tmp = {}
                    teacher_info = teacher(f, getattr(row, "teacher_id"))
                    tmp.update(teacher_info)

                    class_info = clazz(f, getattr(row, "class_id"))
                    tmp.update(class_info)
                    sty = getattr(row, "style")
                    if str(sty)[0]=='9':
                        continue
                
                    today_info = today(f, getattr(row, "base_chapters", ), getattr(row, "base_sections"),
                                       getattr(row, "style"))
                    tmp.update(today_info)

                    cur_day= getattr(row, "day")
                    
                    his_cha = []
                    for i in range(HIS_DAYS_NUM):
                        his_info = yesterday(ff, getattr(row, "class_id"), cur_day, i+1)
                        his_cha.append(list(his_info.values())[0])

                    max_len = find_LCIS(his_cha)
                    if max_len ==14:
                        from dateutil.relativedelta import relativedelta
                        day_last_year = (datetime.datetime.strptime(cur_day,"%Y-%m-%d") - relativedelta(months=12)).strftime("%Y-%m-%d")
                        cur_day=day_last_year

                    for i in range(HIS_DAYS_NUM):
                        his_info = yesterday_sty(ff, getattr(row, "class_id"), cur_day, i+1)
                        tmp.update(his_info)

                    
                                        
                    study_info = study(f, getattr(row, "class_id"), getattr(row, "chapter_id"))
                    tmp.update(study_info)
                    study_ana_info = study_analysis(ff, getattr(row, "class_id"), getattr(row, "base_sections"))
                    tmp.update(study_ana_info)

                    submit_info = submit(f, getattr(row, "class_id"), getattr(row, "day"))
                    tmp.update(submit_info)

                    capacity_info = capacity(f, getattr(row, "teacher_id"))
                    tmp.update(capacity_info)

                    prefer_info = prefer(f, getattr(row, "teacher_id"), getattr(row, "day"))
                    tmp.update(prefer_info)
                    
                    register_info = register(f, getattr(row, "teacher_id"), getattr(row, "day"))
                    tmp.update(register_info)
                    homework_info = homework(ff, getattr(row, "teacher_id"), getattr(row, "day"))
                    tmp.update(homework_info)
                    #style_info = style(ff, getattr(row, "teacher_id"), getattr(row, "day"))
                    #tmp.update(style_info)
                    week_info = week(ff, getattr(row, "teacher_id"), getattr(row, "day"))
                    tmp.update(week_info)
                    reflect_info = reflect(ff, getattr(row, "class_id"), getattr(row, "day"))
                    tmp.update(reflect_info)

                    last_day_info = last_day(ff, getattr(row, "class_id"), getattr(row, "day"))
                    tmp.update(last_day_info)

                    tmp["label"] = getattr(row, "label")
                    res.append(tmp)
        return res

    try:
        if len(train_base) <= batch_size:
            yield proc(train_base)
        else:
            cnt = 0
            while cnt < len(train_base):
                tmp = train_base[cnt:cnt + batch_size]
                cnt += batch_size
                yield proc(tmp)
    except:
        #print(traceback.format_exc())
        # just to avoid process's done, but this gen has't been gc!
        pass
    finally:
        return


def get_lastyear_data(cursor,cursor2, teacher_id,class_id,day):

        #multi values  limit 1  desc
    # ignore school_id  "result_data10.txt",,add school_id "result_data13.txt"
    # ignore school_id  "result_data16.txt",,add school_id "result_data15.txt"
    sql = """SELECT province_id,city_id,county_id FROM r_teacher_info where teacher_id={}"""
    cursor2.execute(sql.format(teacher_id))
    res = cursor2.fetchall()
    
    base_4 = {}

    if len(res) != 0:
        
        base_ = res[0]

        sql2 = """SELECT teacher_id FROM r_teacher_info WHERE province_id={} AND city_id={} AND county_id={}"""
        cursor2.execute(sql2.format(base_["province_id"],base_["city_id"],base_["county_id"]))
        res = cursor2.fetchall()

        if len(res) != 0:
            base_2 = res

            #term
            #sql3 = """SELECT edition_id,grade FROM r_teacher_class WHERE class_id={} """
            sql3 = """SELECT edition_id,grade FROM r_class_info WHERE class_id={}"""
            cursor2.execute(sql3.format(class_id))
            res = cursor2.fetchall()
            if len(res) != 0:
                base_3 = res[0]
                
                
                #term ignore?
                sql3 = """SELECT base_chapters,base_sections,style FROM daily_homework_info WHERE `day`='{}' AND edition_id={} AND grade={} AND teacher_id in {} GROUP BY teacher_id"""
                #sql3 = """SELECT base_chapters,base_sections,style FROM daily_homework_info WHERE `day`='{}' AND edition_id={} AND grade={} AND teacher_id in {} GROUP BY teacher_id limit 1"""
                
                base_22 = [str(t["teacher_id"]) for t in base_2]
                base_22 = "(" +",".join(base_22)+ ")"

                cursor.execute(sql3.format(day,base_3["edition_id"],base_3["grade"],base_22))
                res_f = cursor.fetchall()
                #mybe not one
                #base_4 = res_f if len(res_f) >0 else {"base_chapters": 0,"base_sections": 0,"style": 0}
                base_4 = res_f 
                #print("base_4",base_4)
                
    #  (1064, "You have an error in your SQL syntax; check the manual that corresponds
    #   to your MySQL server version for the right syntax to use near '2' at line 1")
    return base_4
  


def get_data_with_pandas_lastyear(train_base_path: str, batch_size: int = 128) -> typing.List:
    # maybe we can read ourselves to avoid OOM error!
        # if os.path.exists(train_base_path % (aim)) :
        # return train_base_path % (aim)
    train_base = pd.read_csv(train_base_path)

    train_base.drop(labels=["time", "rank", "tag", "cnt"], axis=1, inplace=True)
    # print(len(train_base))
    # """to test"""
    # train_base = train_base[:1]

    def proc(train_base):
        assert len(train_base) > 0, "len(train_base) must >0!"
        res = []
        #这是两个mysql表
        #曾经的
        with get_coursor("recom_turing", "dict") as f:
            with get_coursor("recom", "dict") as ff:
                #这是从日志中，解析出的csv
                #实时性的一些操作记录
                for row in train_base.itertuples(index=False):
                    tmp = {}


                    day, teacher_id,class_id= getattr(row, "day"),getattr(row, "teacher_id"),getattr(row, "class_id")
                    from dateutil.relativedelta import relativedelta
                    day_last_year = (datetime.datetime.strptime(day,"%Y-%m-%d") - relativedelta(months=12)).strftime("%Y-%m-%d")
    
                                       
                    teacher_info = teacher(f, teacher_id)
                    tmp.update(teacher_info)

                    class_info = clazz(f, class_id)
                    tmp.update(class_info)
                    sty = getattr(row, "style")
                    if str(sty)[0]=='9':
                        continue
                    
                    today_list_ = get_lastyear_data( ff,f, teacher_id,class_id,day_last_year)
                    if len(today_list_) ==0:
                        continue
                    today_list = []
                    for i in today_list_:
                        if i not in today_list:
                            today_list.append(i)


                    cur_day=day_last_year
                    for i in range(14):
                        his_info = yesterday_sty(ff, class_id, cur_day, i+1)
                        tmp.update(his_info)
                                        
                    study_info = study(f, class_id, getattr(row, "chapter_id"))
                    tmp.update(study_info)

                    study_ana_info = study_analysis(ff, class_id, getattr(row, "base_sections"))
                    tmp.update(study_ana_info)

                    submit_info = submit(f, class_id,day_last_year)
                    tmp.update(submit_info)

                    capacity_info = capacity(f, teacher_id)
                    tmp.update(capacity_info)

                    prefer_info = prefer(f, teacher_id, day_last_year)
                    tmp.update(prefer_info)
                    
                    register_info = register(f, teacher_id, day_last_year)
                    tmp.update(register_info)
                    homework_info = homework(ff, teacher_id, day_last_year)
                    tmp.update(homework_info)
                    #style_info = style(ff, teacher_id, day_last_year)
                    #tmp.update(style_info)
                    week_info = week(ff, teacher_id,day_last_year)
                    tmp.update(week_info)
                    reflect_info = reflect(ff, class_id, day_last_year)
                    tmp.update(reflect_info)

                    last_day_info = last_day(ff, class_id, day_last_year)
                    tmp.update(last_day_info)

                    tmp["label"] = 1

                    for num,ii in enumerate(today_list):
                        #去重 
                        today_info = today(f,ii["base_chapters"], ii["base_sections"],
                                       ii["style"])
                        
                     
                        test_data_dict_ = tmp.copy()
                        test_data_dict_.update(today_info)
                        #today_ = dict(test_data_dict.items()+today_dict.items())
                        res.append(test_data_dict_)
                    

                    
        return res

    try:
        if len(train_base) <= batch_size:
            yield proc(train_base)
        else:
            cnt = 0
            while cnt < len(train_base):
                tmp = train_base[cnt:cnt + batch_size]
                cnt += batch_size
                yield proc(tmp)
    except:
        #print(traceback.format_exc())
        # just to avoid process's done, but this gen has't been gc!
        pass
    finally:
        return



# maybe cause OOM error, better to consider read ourselves
def get_data_with_pandas_test(test_base_path: str, batch_size: int = 128) -> typing.List:
    # maybe we can read ourselves to avoid OOM error!
    try:
        test_base = pd.read_csv(test_base_path)
        #训练 有drop
        test_base.drop(labels=["time", "rank", "tag", "cnt"], axis=1, inplace=True)
        # print(len(train_base))
        # """to test"""
        # train_base = train_base[:1]

        #def proc_test(train_base):
        assert len(test_base) > 0, "len(train_base) must >0!"
      

        with get_coursor("recom_turing", "dict") as f:
            with get_coursor("recom", "dict") as ff:
                
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
                    #print("chapter_id:" ,chapter_id)

                    aim = item[1]
                    for row in aim.itertuples(index=False):
                        if getattr(row, "label") == 1:
                            aa = {"base_chapters": getattr(row, "base_chapters"),
                                "base_sections": getattr(row, "base_sections"),
                                }
                            break

                    teacher_info = teacher(f, teacher_id)
                    class_info = clazz(f, class_id)
                    #
                    study_info = study(f, class_id, chapter_id)
                    study_ana_info = study_analysis(ff, class_id, getattr(row, "base_sections"))

                    submit_info = submit(f,class_id, day)
                    capacity_info = capacity(f, teacher_id)
                    prefer_info = prefer(f, teacher_id, day)           
                    register_info = register(f, teacher_id, day)
                    homework_info = homework(ff, teacher_id, day)
                    #style_info = style(ff, teacher_id,day)
                    week_info = week(ff, teacher_id, day)
                    reflect_info = reflect(ff, class_id,day)
                    last_day_info = last_day(ff, class_id, day)
                    #tmp["label"] = getattr(row, "label")
                    
                    test_data_dict= {}
                    info_off_fea = [teacher_info,class_info,submit_info,capacity_info,prefer_info,
                    register_info,homework_info,week_info,reflect_info,last_day_info,study_info ,study_ana_info]
                    for i in range(len(info_off_fea)):
                        test_data_dict.update(info_off_fea[i])

                    for i in range(HIS_DAYS_NUM):
                        #his_info = 'yes' + str(i+1) +'_info'
                       # his_info = yesterday(ff, class_id, day, i+1)
                        his_info = yesterday_sty(ff, class_id, day, i+1)
                        test_data_dict.update(his_info)

                         

                    #拿候选集，依据 chapterid 从csv拿    DESC
                    sql = """SELECT chapters base_chapters,sections base_sections,style FROM chapter_homework_set
                    WHERE chapter_id={} ORDER BY rank ASC""".format(chapter_id)
                    ff.execute(sql)          
                    condicate = ff.fetchall()        
                    
                    condicate = condicate or []

                    if len(condicate) == 0:
                        continue

                    test_batch = []
                    condicate_id = []
                    #condicate_id = condicate_id.tolist() 
                    for i  in  range(len(condicate[:500])):
                        today_dict= {}
                        today_dict["today_chapters_ph"] = condicate[i]["base_chapters"]
                        today_dict["today_sections_ph"] = condicate[i]["base_sections"]
                        today_dict["today_style_ph"] = condicate[i]["style"]
                        
                        test_data_dict_ = test_data_dict.copy()
                        test_data_dict_.update(today_dict)
                        #today_ = dict(test_data_dict.items()+today_dict.items())
                        test_batch.append(test_data_dict_)
                        
                        #后面topk对比  key值要一样啊啊
                        # condicate_id.append({"today_chapters_ph":condicate[i]["base_chapters"],
                        #                 "today_sections_ph": condicate[i]["base_sections"]})
                        condicate_id.append({"base_chapters":condicate[i]["base_chapters"],
                                        "base_sections": condicate[i]["base_sections"]})
                    

                    
                    rst = test_batch,condicate_id,aa

  
                    if len(rst[0]) > 0:
                        yield rst 

    except:
        print(traceback.format_exc())
        # just to avoid process's done, but this gen has't been gc!
        pass
    finally:
        return


def teacher(cursor, teacher_id: int) -> dict:
    table, key, setting, col, trans, alias, sql = CONF.get("teacher")

    cursor.execute(sql.format(teacher_id))
    res = cursor.fetchall()
    base = res[0] if len(res) > 0 else {"teacher_id": teacher_id,
                                        'student_count': 0, 'province_id': 0,
                                        'city_id': 0, 'core_type': 0}

    return modify(base, "teacher", col, trans, alias)


def clazz(cursor, class_id: int) -> dict:
    table, key, setting, col, trans, alias, sql = CONF.get("class")

    sql = """select class_id,edition_id,grade,student_count class_student,cap_avg cap_ave,cap_max,cap_min from r_class_info where class_id = {}"""

    cursor.execute(sql.format(class_id))
    res = cursor.fetchall()

    base = res[0] if len(res) > 0 else {"class_id": class_id, 'edition_id': 0, 'grade': 1, 'class_student': 0,
                                        'cap_ave': 0, 'cap_max': 0,
                                        'cap_min': 0}
    return modify(base, "class", col, trans, alias)


def today(cursor, base_chapters: str, base_sections: str, style: int) -> dict:
    table, key, setting, col, trans, alias, sql = CONF.get("today")
    base = {"base_chapters": base_chapters, "base_sections": base_sections,
            "style": style}
    res = modify(base, "today", col, trans, alias)
    return res
    '''
    user id 
    day

    str - > int 
    int 

    '''

def yesterday(cursor, class_id: int, day: str, cnt: int) -> dict:
    name = "yesterday%s" % (cnt) if cnt > 1 else "yesterday"
   
    table, key, setting, col, trans, alias, sql = CONF.get(name)

    chap_name, sec_name = ("yes%s_chap" % (cnt), "yes%s_sec" % (cnt)) if cnt > 1 else ("yes_chap", "yes_sec")
    
    his_day =datetime.datetime.strptime(day, "%Y-%m-%d") + datetime.timedelta(days=-cnt)
    day__ = his_day.date()

    sql = """select base_chapters {},base_sections {} from daily_homework_info where class_id={} and `day`='{}' order by homework_id desc limit 1"""
    cursor.execute(sql.format(chap_name, sec_name, class_id, day__))
    res = cursor.fetchall()
   
    base={}
    base = res[0] if len(res) > 0 else {chap_name: "0", sec_name: "0"}
    # if len(res) > 0:
    #     base = res[0]
    # else:
    #     base = {chap_name: "0", sec_name: "0"}
        #若当天没有作业，怎往前推7天，直到有跳出  赋值给当天，也可以到 handle那边修改
        # for i in range(7):
        #     tmp = cnt + i+1

        #     #chap_name_, sec_name_ = ("yes%s_chap" % (tmp), "yes%s_sec" % (tmp)) if tmp > 1 else ("yes_chap", "yes_sec")
        #     day_ = (datetime.datetime.strptime(his_day.strftime("%Y-%m-%d"), "%Y-%m-%d") +
        #         datetime.timedelta(days=-tmp)).date()
        #     cursor.execute(sql.format(chap_name, sec_name, class_id, day_))
        #     res = cursor.fetchall()
        #     if len(res) > 0:
        #         base = res[0]
        #         return modify(base, name, col, trans, alias)
        #     if i==6:
        #         base = {chap_name: "0", sec_name: "0"}


    return modify(base, name, col, trans, alias)



def yesterday_sty(cursor, class_id: int, day: str, cnt: int) -> dict:
    name = "yesterday%s" % (cnt) if cnt > 1 else "yesterday"
    conf = load_conf2()
    table, key, setting, col, trans, alias, sql = conf.get(name)

    chap_name, sec_name,sty_name = ("yes%s_chap" % (cnt), "yes%s_sec" % (cnt), "yes%s_sty" % (cnt)) if cnt > 1 else ("yes_chap", "yes_sec", "yes_sty")
    
    his_day =datetime.datetime.strptime(day, "%Y-%m-%d") + datetime.timedelta(days=-cnt)
    day__ = his_day.date()

    sql = """select base_chapters {},base_sections {},style {} from daily_homework_info where class_id={} and `day`='{}' order by homework_id desc limit 1"""
    cursor.execute(sql.format(chap_name, sec_name, sty_name, class_id, day__))
    res = cursor.fetchall()
   
    base={}
    base = res[0] if len(res) > 0 else {chap_name: "0", sec_name: "0", sty_name: "0"}

    return modify(base, name, col, trans, alias)


def study(cursor, class_id, chapter_id) -> dict:
    table, key, setting, col, trans, alias, sql = CONF.get("study")

    sql = """select vector,gap_days from class_learning_status_vector where class_id = {} and chapter_id = {} order by `day` desc limit 1"""

    cursor.execute(sql.format(class_id, chapter_id))
    res = cursor.fetchall()

    base = res[0] if len(res) > 0 else {"vector": "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                                        "gap_days": 15}

    return modify(base, "study", col, trans, alias)

def study_analysis(cursor, class_id, section_id) -> dict:
    conf = load_conf2()
    table, key, setting, col, trans, alias, sql = conf.get("study_analysis")
    #限制1  其实多个
    sql = """select avg_times,avg_rate,avg_exp_score,avg_exp_level from class_section_status where class_id = {} and  section_id in {}"""
    
    base_22 = [t for t in section_id.split(",")]
    base_22 = "(" +",".join(base_22)+ ")"
    
    cursor.execute(sql.format(class_id, base_22))
    res = cursor.fetchall()

    sql = """select avg_times analysis_avg_times,avg_rate analysis_avg_rate,avg_exp_score analysis_avg_exp_score,avg_exp_level analysis_avg_exp_level from class_section_status where class_id = {} and section_id in {} order by `update_time` desc limit 1"""

    cursor.execute(sql.format(class_id, base_22))
    res = cursor.fetchall()

    base = res[0] if len(res) > 0 else {"analysis_avg_times": 0,"analysis_avg_rate": 0,"analysis_avg_exp_score": 0,"analysis_avg_exp_level": 0}

    return modify(base, "study_analysis", col, trans, alias)


def submit(cursor, class_id: int, day: str) -> dict:
    table, key, setting, col, trans, alias, sql = CONF.get("submit")

    cursor.execute(sql.format(class_id, day))
    res = cursor.fetchall()
    base = res[0] if len(res) > 0 else {'month_submit_rate': 0}

    return modify(base, "submit", col, trans, alias)


def capacity(cursor, teacher_id: int) -> dict:
    table, key, setting, col, trans, alias, sql = CONF.get("capacity")
    to_sql = "SELECT school_id,county_id,city_id,province_id FROM r_teacher_info WHERE teacher_id={}"

    cursor.execute(to_sql.format(teacher_id))
    teacher_info = cursor.fetchall()
    base = {"capacity": 0.0}
    if len(teacher_info) != 0:
        regin = ["school", "county", "city", "province"]

        to_sql = """SELECT capacity FROM region_capacity_average_daily WHERE region_id={} AND region_type='{}' """
        teacher_info = teacher_info[0]

        for k in regin:
            cursor.execute(to_sql.format(teacher_info[k + "_id"], k))
            res = cursor.fetchall()
            if len(res) == 0:
                continue
            base = {"capacity": res[0]["capacity"]}
            break

    return modify(base, "capacity", col, trans, alias)


def prefer(cursor, teacher_id: int, day: str) -> dict:
    table, key, setting, col, trans, alias, sql = CONF.get("prefer")

    cursor.execute(sql.format(teacher_id, day))
    res = cursor.fetchall()

    base = res[0] if len(res) > 0 else {'month_assign_time_avg': 0.0,
                                        'month_assign_time_var': 0.0,
                                        'month_assign_rank_avg': 0.0,
                                        'month_assign_rank_var': 0.0}

    return modify(base, "prefer", col, trans, alias)


def register(cursor, teacher_id: int, day: str) -> dict:
    table, key, setting, col, trans, alias, sql = CONF.get("register")

    sql = """SELECT register_day FROM r_teacher_info WHERE teacher_id={} """

    cursor.execute(sql.format(teacher_id))
    res = cursor.fetchall()

    base = {"register": 0}
    if len(res) > 0:
        day = datetime.datetime.strptime(day, "%Y-%m-%d", ).date()
        base = {"register": (day - res[0]["register_day"]).days}
    return modify(base, "register", col, trans, alias)


def homework(cursor, teacher_id: int, day: str) -> dict:
    table, key, setting, col, trans, alias, sql = CONF.get("homework")
    sql = """SELECT COUNT(*) a FROM daily_homework_info WHERE teacher_id={} AND `day`<'{}'"""

    cursor.execute(sql.format(teacher_id, day))
    res = cursor.fetchall()

    base = {"homework": 0}
    if len(res) > 0:
        base = {"homework": res[0]['a']}

    return modify(base, "homework", col, trans, alias)


def style(cursor, teacher_id: int, day: str) -> dict:
    sty = []
    for fir in ["1", "2", "3", "4"]:
        for sec in ["100", "010", "001", "110", "101", "011", "111"]:
            sty.append(fir + "0" + sec)
    ssl = "(" + ",".join(sty) + ")"

    table, key, setting, col, trans, alias, sql = CONF.get("style")

    base = {}
    for s in sty:
        base[s] = 0

    day = datetime.datetime.strptime(day, "%Y-%m-%d", )
    #这是是不是14天
    #his = day + datetime.timedelta(days=-30)
    his = day + datetime.timedelta(days=-HIS_DAYS_NUM)


    sql = """SELECT style,COUNT(1) cnt FROM daily_homework_info WHERE teacher_id={} AND `day`<'{}' AND `day`>='{}' AND style in {} GROUP BY style"""

    cursor.execute(sql.format(teacher_id, day.date(), his.date(), ssl))
    res = cursor.fetchall()

    if len(res) > 0:
        for i in res:
            i = {str(i.get("style")): i.get("cnt")}
            base.update(i)

    base = {"style": base}

    return modify(base, 'style', col, trans, alias)


def week(cursor, teacher_id: int, day: str) -> dict:
    table, key, setting, col, trans, alias, sql = CONF.get("week")

    day = datetime.datetime.strptime(day, "%Y-%m-%d", )
    end = day + datetime.timedelta(days=-day.weekday())
    begin = end + datetime.timedelta(days=-28)

    sql = """SELECT week,COUNT(1) cnt FROM daily_homework_info WHERE teacher_id={} AND `day`<'{}' AND `day`>='{}' GROUP BY `week`"""

    cursor.execute(sql.format(teacher_id, end.date(), begin.date()))
    res = cursor.fetchall()

    base = {"week": 0.0}
    if len(res) > 0:
        ssum = 0.0
        for data in res:
            ssum += data["cnt"]
        base = {"week": ssum / 4}

    return modify(base, 'week', col, trans, alias)


def reflect(cursor, class_id: int, day: str) -> dict:
    table, key, setting, col, trans, alias, sql = CONF.get("reflect")

    day = int(datetime.datetime.strptime(day, "%Y-%m-%d", ).strftime("%m%d"))

    sql = """SELECT edition_id,grade FROM r_teacher_class WHERE class_id={} """

    cursor.execute(sql.format(class_id))
    res = cursor.fetchall()

    base = {"reflect": "0"}
    if len(res) > 0:
        edition_id, grade = res[0].get("edition_id"), res[0].get("grade")

        o_sql = """select section_id from chapter_recommend_turing.process_section where day={} and edition_id={} and grade={}"""

        cursor.execute(o_sql.format(day, edition_id, grade))
        res1 = cursor.fetchall()

        if len(res1) > 0:
            a = [str(i["section_id"]) for i in res1]
            a = ",".join(a)
            a = "(" + a + ")"

            a_sql = """SELECT section_id FROM base_section_info  WHERE section_id in {} ORDER BY section_order ASC"""
            cursor.execute(a_sql.format(a))
            res2 = cursor.fetchall()
            if len(res2) > 0:
                base = {"reflect": ",".join([str(i["section_id"]) for i in res2])}

    return modify(base, "reflect", col, trans, alias)


def reflect_online(cursor, class_id: int, day: str,edition_id,grade) -> dict:
    table, key, setting, col, trans, alias, sql = CONF.get("reflect")

    day = int(datetime.datetime.strptime(day, "%Y-%m-%d", ).strftime("%m%d"))

    base = {"reflect": "0"}

    edition_id, grade = edition_id, grade

    o_sql = """select section_id from chapter_recommend_turing.process_section where day={} and edition_id={} and grade={}"""

    cursor.execute(o_sql.format(day, edition_id, grade))
    res1 = cursor.fetchall()

    if len(res1) > 0:
        a = [str(i["section_id"]) for i in res1]
        a = ",".join(a)
        a = "(" + a + ")"

        a_sql = """SELECT section_id FROM base_section_info  WHERE section_id in {} ORDER BY section_order ASC"""
        cursor.execute(a_sql.format(a))
        res2 = cursor.fetchall()
        if len(res2) > 0:
            base = {"reflect": ",".join([str(i["section_id"]) for i in res2])}

    return modify(base, "reflect", col, trans, alias)

def reflect_online2(cursor, class_id: int, day: str,edition_id,grade) -> dict:
    table, key, setting, col, trans, alias, sql = CONF.get("reflect")

    day = int(datetime.datetime.strptime(day, "%Y-%m-%d", ).strftime("%m%d"))

    sql = """SELECT edition_id,grade FROM r_teacher_class WHERE class_id={} """

    cursor.execute(sql.format(class_id))
    res = cursor.fetchall()

    base = {"reflect": "0"}
    if len(res) > 0:
        edition_id, grade = res[0].get("edition_id"), res[0].get("grade")

        o_sql = """select section_id from chapter_recommend_turing.process_section where day={} and edition_id={} and grade={}"""

        cursor.execute(o_sql.format(day, edition_id, grade))
        res1 = cursor.fetchall()

        if len(res1) > 0:
            a = [str(i["section_id"]) for i in res1]
            a = ",".join(a)
            a = "(" + a + ")"

            a_sql = """SELECT section_id FROM base_section_info  WHERE section_id in {} ORDER BY section_order ASC"""
            cursor.execute(a_sql.format(a))
            res2 = cursor.fetchall()
            if len(res2) > 0:
                base = {"reflect": ",".join([str(i["section_id"]) for i in res2])}

    return modify(base, "reflect", col, trans, alias)


def last_day(cursor, class_id: int, day: str) -> dict:
    table, key, setting, col, trans, alias, sql = CONF.get("last_day")

    day = datetime.datetime.strptime(day, "%Y-%m-%d", )

    day = (day + datetime.timedelta(days=-LAST_DAYS)).date()

    sql = """SELECT COUNT(*) a FROM daily_homework_info WHERE class_id={} AND `day`='{}'"""

    cursor.execute(sql.format(class_id, day))
    res = cursor.fetchall()
    base = {"last_day": 0}
    if len(res) > 0:
        base = {"last_day": 1 if res[0]['a'] > 0 else 0}

    return modify(base, "last_day", col, trans, alias)


def modify(base: dict, name: str, col: typing.List[str], trans: typing.List[str], alias: typing.List[str]) -> dict:
    res = {}
    for old, new in zip(col, alias):
        res[new] = base[old]
    return res


if __name__ == "__main__":
    # get_feature("train_base-2019-03-07.csv", "123")

    pass
    data = None
    import os
    PATH = os.path.dirname(os.path.abspath(__file__))

    for i in get_data_with_pandas(os.path.join(PATH,"train_base/train_base--2019-09-01.csv"),
                                  5):
        data = i
        break
    print(len(data))
