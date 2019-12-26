import os 
import pandas as pd 
import csv
import datetime
PATH = os.path.dirname(os.path.abspath(__file__))

start_day = "2019-09-03"

nums =83 # 66 9.2->11.6   83  9.2 ->11.23
curentday =start_day

#已存在的csv 按行 按列增加
rst = os.path.join(PATH,"topk/result_sum_63.2.csv")
headers =['date','len','top1','top3','top5','top10','top15']
with open(rst, "a") as fo:
    f_csv = csv.writer(fo)
    f_csv.writerow(headers)


for n in range(nums):
    if curentday == '2019-10-28' or curentday == '2019-10-27':
        tmp = datetime.datetime.strptime(curentday,"%Y-%m-%d") + datetime.timedelta(days=1)
        curentday = tmp.strftime("%Y-%m-%d")
        continue
    name = os.path.join(PATH,"result_sum_63/result-{}.csv".format(curentday))
    if not os.path.exists(name):
        continue

    # print("*******************")
    a = pd.read_csv("%s" % (name))
    # print(a.columns)
    # print(len(a))
    a = a[~a["0"].isin([-1])]
    # print(len(a))
    # print(n, len(a), a["0"].sum(), a["1"].sum(), a["2"].sum(),
    #         a["3"].sum(), a["4"].sum(), sep=",")

    with open(rst, "a") as fo:                  
        f_csv = csv.writer(fo) 
        f_csv.writerow([curentday,len(a),a["0"].sum(), a["1"].sum(), a["2"].sum()
        , a["3"].sum(), a["4"].sum()])

    tmp = datetime.datetime.strptime(curentday,"%Y-%m-%d") + datetime.timedelta(days=1)
    curentday = tmp.strftime("%Y-%m-%d")




#     0,117,5,8,10,13,18
# 1,216,17,46,61,76,85
# 2,280,38,78,99,133,158
# 3,292,50,114,142,182,200
# 4,362,58,119,153,207,235
# 5,253,38,70,93,122,140
# 6,187,26,80,98,118,125
# 7,311,48,118,158,190,205
# 8,274,55,139,180,216,226
# 9,305,54,145,181,222,236
# 10,397,88,209,275,331,345
# 11,197,41,106,134,159,164
# 12,206,49,97,134,162,168
# 13,191,37,85,124,145,150
# 14,300,67,141,197,244,250
# 15,307,79,180,206,241,243
# 16,338,92,193,241,273,277
# 17,365,119,212,256,294,308
# 18,392,102,212,257,304,311
# 19,228,63,131,150,184,186
# 20,204,42,99,130,156,157
# 21,313,79,159,193,243,251
# 22,318,90,179,221,253,257
# 23,310,87,173,222,261,265
# 24,331,79,166,228,270,279
# 25,391,74,189,256,309,320
# 26,216,43,92,125,161,166
# 27,325,76,174,213,263,266
# 28,433,93,207,273,341,347
# 29,254,60,123,163,203,206
# 30,247,61,136,168,189,193
# 31,235,50,106,157,188,189
# 32,288,76,158,199,225,225
# 33,226,43,99,140,167,169
# 34,235,50,127,154,180,184
# 35,218,47,106,132,160,166
# 36,304,88,153,185,230,238
# 37,317,77,154,197,249,252
# 38,327,73,171,217,267,270
# 39,344,77,179,222,276,279
# 40,363,78,185,231,286,286
# 41,224,50,120,151,183,184
# 42,321,55,146,202,252,256
# 43,338,73,183,242,282,286
# 44,343,77,171,216,262,265
# 45,333,81,179,234,270,274
# 46,392,91,212,273,311,313
# 47,247,57,117,162,200,201
# 48,188,44,101,130,150,150
# 49,318,77,172,215,257,264
# 50,313,85,172,214,255,258
# 51,332,88,175,241,268,271
# 52,358,69,168,242,290,298
# 53,360,66,166,238,281,288