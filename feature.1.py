from utils import *
import datetime
from itertools import chain
import os, typing
import json
import traceback


# maybe cause OOM error, better to consider read ourselves
def get_data_with_pandas(train_base_path: str, batch_size: int = 128) -> typing.List:
    # maybe we can read ourselves to avoid OOM error!
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

                    today_info = today(f, getattr(row, "base_chapters", ), getattr(row, "base_sections"),
                                       getattr(row, "style"))
                    tmp.update(today_info)

                    cur_day= getattr(row, "day")
                    for i in range(14):
                        #his_info = 'yes' + str(i+1) +'_info'
                        
                        his_info = yesterday(ff, getattr(row, "class_id"), cur_day, i+1)
                        tmp.update(his_info)

                        # tmp_day = datetime.datetime.strptime(cur_day,"%Y-%m-%d") + datetime.timedelta(days=-(i+1))
                        # cur_day = tmp_day.strftime("%Y-%m-%d")
                    
                    study_info = study(f, getattr(row, "class_id"), getattr(row, "day"))
                    tmp.update(study_info)

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
                    style_info = style(ff, getattr(row, "teacher_id"), getattr(row, "day"))
                    tmp.update(style_info)
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


# maybe cause OOM error, better to consider read ourselves
def get_data_with_pandas_test(train_base_path: str, batch_size: int = 128) -> typing.List:
    # maybe we can read ourselves to avoid OOM error!
    train_base = pd.read_csv(train_base_path)
    train_base.drop(labels=["time", "rank", "tag", "cnt"], axis=1, inplace=True)
    # print(len(train_base))
    # """to test"""
    # train_base = train_base[:1]

    def proc_test(train_base):
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

                    today_info = today(f, getattr(row, "base_chapters", ), getattr(row, "base_sections"),
                                       getattr(row, "style"))
                    tmp.update(today_info)

                    cur_day= getattr(row, "day")
                    for i in range(14):
                        #his_info = 'yes' + str(i+1) +'_info'
                        
                        his_info = yesterday(ff, getattr(row, "class_id"), cur_day, i+1)
                        tmp.update(his_info)

                        # tmp_day = datetime.datetime.strptime(cur_day,"%Y-%m-%d") + datetime.timedelta(days=-(i+1))
                        # cur_day = tmp_day.strftime("%Y-%m-%d")
                    
                    #study_info = study(f, getattr(row, "class_id"), getattr(row, "day"))
                    study_info = study(f, getattr(row, "class_id"), getattr(row, "chapter_id"))
                    tmp.update(study_info)

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
                    style_info = style(ff, getattr(row, "teacher_id"), getattr(row, "day"))
                    tmp.update(style_info)
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
            yield proc_test(train_base)
        else:
            cnt = 0
            while cnt < len(train_base):
                tmp = train_base[cnt:cnt + batch_size]
                cnt += batch_size
                yield proc_test(tmp)
    except:
        #print(traceback.format_exc())
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


def study(cursor, class_id, chapter_id) -> dict:
    table, key, setting, col, trans, alias, sql = CONF.get("study")

    sql = """select vector,gap_days from class_learning_status_vector where class_id = {} and chapter_id = {} order by `day` desc limit 1"""

    cursor.execute(sql.format(class_id, chapter_id))
    res = cursor.fetchall()

    base = res[0] if len(res) > 0 else {"vector": "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                                        "gap_days": 15}

    return modify(base, "study", col, trans, alias)


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
    his = day + datetime.timedelta(days=-30)

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


def last_day(cursor, class_id: int, day: str) -> dict:
    table, key, setting, col, trans, alias, sql = CONF.get("last_day")

    day = datetime.datetime.strptime(day, "%Y-%m-%d", )
    day = (day + datetime.timedelta(days=-7)).date()

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
