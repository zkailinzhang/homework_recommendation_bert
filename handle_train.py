import datetime, time
import numpy as np

#import handle_log
from utils import *

# record_view_click = handle_log.record_view_click
# record_view_view = handle_log.record_view_view
#train_base_path = "/data/lishuang/turing_new/train_base/train_base-%s.csv"
import os
PATH = os.path.dirname(os.path.abspath(__file__))

train_base_path = os.path.join(PATH,"train_base/train_base-%s.csv")

def all2base():
    with get_coursor("recom", ) as f:
        f.execute("""SELECT section_jichu,section_kousuan,section_yingyong,section_order FROM base_section_map""")
        info = f.fetchall()
        f.execute("""SELECT section_id,parent_id FROM base_section_info WHERE level=3""")
        info3 = f.fetchall()
        f.execute("""SELECT section_id,section_order FROM base_section_info WHERE level=2""")
        info2 = f.fetchall()

    sec2 = {}
    for i in info2:
        sec2[i[0]] = [i[0], i[1]]
    sec3 = {}
    for i in info3:
        try:
            sec3[i[0]] = sec2[i[1]]
        except:
            pass

    kousuan, yingyong, jichu, sec2chap = {}, {}, {}, {}
    for i in info:
        if i[1] != 0:
            kousuan[i[1]] = i[0]
        if i[2] != 0:
            yingyong[i[2]] = i[0]
        jichu[i[0]] = [i[0], i[3]]
        sec2chap[i[0]] = sec3[i[0]]
    return jichu, kousuan, yingyong, sec2chap


def handle_log(begin, end, view_click):
    jichu, kousuan, yingyong, sec2chap = all2base()
    a, b = [], []
    with get_coursor("recom") as qq:
        #with open(view_click, "r",encoding='utf-8') as f:
        with open(view_click, "r") as f:
            for com in f.readlines():
                tmp = com.split(";")
                t = datetime.datetime.strptime(tmp[0], "%Y-%m-%d %H:%M:%S")
                if t < begin:
                    continue
                elif t >= end:
                    break
                else:
                    secs = [int(i.strip()) for i in tmp[4].split(",")]
                    ss, cc = [], []
                    for i in secs:
                        base = kousuan.get(i, None) or yingyong.get(i, None) or jichu.get(i, None)
                        if base:
                            if not isinstance(base, list):
                                base = jichu.get(base, None)
                        if not base:
                            qq.execute(
                                "SELECT section_id,section_order FROM base_section_info WHERE section_id={}".format(i))
                            info = qq.fetchone()
                            try:
                                base = [info[0], info[1]]
                            except:
                                continue
                            ss.append(base)
                            cc.append(base)
                        else:
                            ss.append(base)
                            cc.append(sec2chap[base[0]])
                    ss = sorted(ss, key=lambda x: x[1], )
                    cc = sorted(cc, key=lambda x: x[1], )
                    s, c = [], []
                    for i, j in zip(ss, cc):
                        ii, jj = str(i[0]), str(j[0])
                        if ii not in s:
                            s.append(ii)
                        if jj not in c:
                            c.append(jj)
                    a.append(
                        [t, int(tmp[1]), int(tmp[2]), ",".join(c), ",".join(s), int(tmp[5]), int(tmp[6]),
                         tmp[7].strip()])
                    b.append(int(tmp[8]))

    return a, b


def generate_train_base(aim: str = "2018-10-26",logtxt:str="log/new-view-click.txt") -> str:

    if os.path.exists(train_base_path % (aim)) :
        return train_base_path % (aim)
    else:  
        names = ["time", "teacher_id", "class_id", "base_chapters", "base_sections", "style", "rank", "tag"]

        begin = datetime.datetime.strptime(aim, "%Y-%m-%d")
        end = begin + datetime.timedelta(days=1)

        """
        use pandas, but may cause OOM err。
        # a = pd.read_csv("/data/lishuang/handle-log/view-click.txt", sep=";", names=names)
        # b = a[a["time"] < [aim] * len(a)]
        """

        #a, c = handle_log(begin, end,os.path.join(PATH,logtxt))
        a, c = handle_log(begin, end,logtxt)
        a = pd.DataFrame(a, columns=names)
        # print(len(a))
        a["day"] = [i.date() for i in a["time"]]
        a["chapter_id"] = c

        o = pd.DataFrame()

        with get_coursor("recom", "dict") as f:
            cnt = 0
            while True:
                f.execute("""select distinct teacher_id,class_id,base_chapters,base_sections,1 cnt from daily_homework_info
                  where `day`='{}' """.format(begin.date()))
                data = f.fetchall()
                if len(data) == 0:
                    time.sleep(60)
                    continue
                if cnt == 0:
                    time.sleep(60)
                    cnt = 1
                    continue
                o = get_pandas(data)
                break

        a = a.merge(o, how="left", on=["teacher_id", "class_id", "base_chapters", "base_sections"])
        a.drop_duplicates(subset=["teacher_id", "class_id", "base_chapters", "base_sections", "style"], keep="first",
                        inplace=True)

        # biaoqian
        a["label"] = a[["tag", "cnt"]].apply(lambda x: 1 if (not np.isnan(x[1])) and x[0] == "click" else 0, axis=1)

 
        kousuan2jichu = {}
        yingyong2jichu = {}
        jichu = set()
        section3_2 = {}
        with get_coursor("recom", "dict") as ff:
            ff.execute("SELECT section_jichu,section_kousuan,section_yingyong FROM base_section_map")
            res = ff.fetchall()
            for item in res:
                jichu.add(item["section_jichu"])
                kousuan2jichu[item["section_kousuan"]] = item["section_jichu"]
                yingyong2jichu[item["section_yingyong"]] = item["section_jichu"]
            kousuan2jichu.pop(0)
            yingyong2jichu.pop(0)

            ff.execute("SELECT section_id,parent_id FROM base_section_info where level=3")
            res = ff.fetchall()
            for item in res:
                section3_2[item["section_id"]] = item["parent_id"]

        def modify_sec2chap(section_id, base_chapters):
            section_id = int(section_id)
            if section_id == 0:
                return int(base_chapters.strip().split(",")[0])
            if section_id in jichu:
                return section3_2.get(section_id, -1)
            elif section_id in kousuan2jichu:
                return section3_2.get(kousuan2jichu[section_id], -1)
            elif section_id in yingyong2jichu:
                return section3_2.get(yingyong2jichu[section_id], -1)
            else:
                return -1

       

        a["chapter_id"] = [ modify_sec2chap( 
                          a.iloc[i]["chapter_id"],a.iloc[i]["base_chapters"]) for i in range(len(a)) ]
       
        data = pd.DataFrame()
        oo = a.groupby(["teacher_id", "class_id", "day", "chapter_id"])
        for dd in oo:
            if dd[1]["label"].sum() > 0:
                data = pd.concat([data, dd[1]], ignore_index=True, sort=False)

        data.to_csv(train_base_path % (aim), index=False)
        return train_base_path % (aim)


if __name__ == "__main__":
    pass
    handle_log(datetime.datetime.strptime("2019-08-01", "%Y-%m-%d"),
               datetime.datetime.strptime("2019-08-02", "%Y-%m-%d"), )
