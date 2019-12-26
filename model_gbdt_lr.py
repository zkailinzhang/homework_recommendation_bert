import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import pandas as pd
import os 
import time 


class Model(object, ):
    """
    模型原型，用于给出基础推荐结果
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def save(self, filepath, *args, **kwargs):
        joblib.dump(self._model, filepath)

    def load(self, filepath, *args, **kwargs):
        self._model = joblib.load(filepath)

    def accuracy(self, data=None, *args, **kwargs):
        if data:
            return self._model.score(data[0], data[1].values.ravel(), )
        else:
            return self._model.score(self._test_data[0], self._test_data[1].values.ravel())

    def set_train_data(self, train_data):
        self._train_data = train_data

    def set_test_data(self, test_data):
        self._test_data = test_data

    def retrain(self, *args, **kwargs):
        raise NotImplementedError

class Lr(Model):

    def __init__(self, **kwargs):
        self._model = LR(**kwargs)

    def train(self, data=None, *args, **kwargs):
        if data:
            self._model.fit(data[0], data[1].values.ravel())
        else:
            import os
            path = os.path.dirname(os.path.dirname(__file__))
            if os.path.exists(os.path.join(path, 'test', '.model')):
                self._model = joblib.load(os.path.join(path, 'test', 'lr.model'))
                return
            self._model.fit(self._train_data[0], self._train_data[1].values.ravel())

    def predict(self, data=None, *args, **kwargs):
        if data is not None:
            return self._model.predict(data, )
        else:
            return self._model.predict(self._test_data[0], )



class Other_Model(object):

    def fit(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass


class Other_Xgboost(Other_Model):

    def __init__(self, *args, **kwargs):
        self._model = xgb.XGBClassifier(**kwargs)

    def train(self, X, y, **kwargs):
        self._model.fit(X, y, **kwargs)

    def apply(self, X, ntree_limit=0):
        a = self._model.apply(X, ntree_limit=ntree_limit)
        return a[:, :]

    def save(self, filepath):
        # self._model.save_model(filepath)
        joblib.dump(self._model, filepath)

    def load(self, filepath):
        # self._model.load_model(filepath)
        self._model = joblib.load(filepath)
        # return joblib.load(filepath)


def other_handle(model_type=None, *args, **kwargs):
    if model_type == 'xgb':
        x = Other_Xgboost(**kwargs)
        return x
    else:
        return None




class Feature(object):
    """
    加工混合特征，形成各个特征，给模型做输入，返回内存数据或者地址。
    """

    def __init__(self, dis=None, *args, **kwargs):
        self.dis = dis if dis else Discover()

    def handle(self, data, has_label, dtype, prefix="dis_", *args, **kwargs):
        fea1 = self._handle(data, has_label, *args, **kwargs)
        fea2 = self._discover(data, has_label, *args, **kwargs)
        return self._join(fea1, fea2, prefix=prefix, dtype=dtype)


    def _handle(self, data, has_label, *args, **kwargs):
        return data


    def _discover(self, data, has_label, *args, **kwargs):
        data = self.dis.transform(data, has_label, *args, **kwargs)
        return data


    def _join(self, fea1, fea2, prefix="dis_", dtype='both', ):
        if dtype == 'left':
            return fea1
        cols = []
        for i in range(len(fea2[0])):
            cols.append(prefix + str(i))
        fea2 = pd.DataFrame(fea2, columns=cols)
        if dtype == 'right':
            if isinstance(fea1, tuple):
                a1, a2 = fea1
                fea2 = (fea2, a2)
            return fea2
        else:
            if isinstance(fea1, tuple):
                a1, a2 = fea1
                a1 = a1.join(fea2)
                fea1 = (a1, a2)
            else:
                fea1 = fea1.join(fea2)
            return fea1
            # return fea1 + fea2


class Discover(object):

    def __init__(self, model_type=None, *args, **kwargs):
        self._model_type = model_type


    def train_save(self, data, model_path, ohe_path, *args, **kwargs):
        self._train(data, )
        self._save(model_path, ohe_path)

    def _train(self, data, type_kwargs=None, model_kwargs=None, ohe_kwargs=None):
        type_kwargs = {} if not type_kwargs else type_kwargs
        model_kwargs = {} if not model_kwargs else model_kwargs
        ohe_kwargs = {} if not ohe_kwargs else ohe_kwargs

        if self._model_type:
            self._model = other_handle(model_type=self._model_type, **type_kwargs)
            import os
            if os.path.exists('./test/xx.model'):
                self._model = joblib.load('./qq.model')
            else:
                self._model.train(data[0], data[1].values.ravel(), **model_kwargs)
            self._ohe = OneHotEncoder(**ohe_kwargs)
            a = self._model.apply(data[0])
            self._ohe.fit(a)
        else:
            pass

    def _save(self, model_path, ohe_path):
        if self._model_type:
            # try:
            #     self._model.save(model_path)
            # except:
            self._model.save(model_path)
            # joblib.dump(self._model, model_path)
            joblib.dump(self._ohe, ohe_path)
        else:
            pass


    def load(self, model_path, ohe_path, *args, **kwargs):
        self._load(model_path, ohe_path, *args, **kwargs)

    def _load(self, model_path, ohe_path, *args, **kwargs):
        if self._model_type:
            # assert hasattr(self, '_model')
            self._model = other_handle(model_type=self._model_type, )
            self._model.load(model_path)
            self._ohe = joblib.load(ohe_path)
        else:
            pass


    def transform(self, data, has_label, *args, **kwargs):
        if self._model_type:
            if has_label:
                mid = self._model.apply(data[0])
            else:
                mid = self._model.apply(data)
            return self._ohe.transform(mid).toarray()
        else:
            return data

def log_data(day, add=0, filename="file"):
    assert add <= 0
    data = pd.DataFrame()
    if add == 0:
        tmp = pd.read_csv("./train_test/train_base/train_base-%s.csv" % (day.strftime("%Y-%m-%d")))
        #分组，以这几个为索引，，这个老师，这个班级，这天，布置这个章节，
        tmp = tmp.groupby(["teacher_id", "class_id", "day", "chapter_id"])
        for dd in tmp:
            #有正样本  也有负样本
            if dd[1]["label"].sum() > 0:
                data = pd.concat([data, dd[1]], ignore_index=True, sort=False)
        return data
    else:
        if os.path.exists(filename.format(day.strftime("%Y-%m-%d"))):
            return pd.read_csv(filename.format(day.strftime("%Y-%m-%d")))

        middle = day + datetime.timedelta(days=add - 1)
        start, stop = middle, day
        for i in range((stop - start).days):
            date = start + datetime.timedelta(days=i)
            dd = pd.read_csv("./train_test/train_base/train_base-%s.csv" % (date.strftime("%Y-%m-%d")))
            oo = dd.groupby(["teacher_id", "class_id", "day", "chapter_id"])
            for dd in oo:
                if dd[1]["label"].sum() > 0:
                    data = pd.concat([data, dd[1]], ignore_index=True, sort=False)
        data.to_csv(filename.format(day.strftime("%Y-%m-%d")), index=False)
        return data


def generate(base_data):
    conf = load_conf("config")

    # handler
    from handler import Handler_Local, Handler_Remote, Handler_Call
    handler = {
        "teacher": Handler_Remote("recom"),
        "class": Handler_Local("../table/class.csv"),
        "today": Handler_Local(base_data),
        "study": Handler_Remote("recom"),
        "yesterday": Handler_Call(base_data),
        "yesterday2": Handler_Call(base_data),
        "yesterday3": Handler_Call(base_data),

        "yesterday4": Handler_Call(base_data),
        "yesterday5": Handler_Call(base_data),
        "yesterday6": Handler_Call(base_data),
        "yesterday7": Handler_Call(base_data),
        "yesterday8": Handler_Call(base_data),
        "yesterday9": Handler_Call(base_data),
        "yesterday10": Handler_Call(base_data),

        "submit": Handler_Remote("recom"),
        "capacity": Handler_Call(base_data),
        "prefer": Handler_Remote("recom"),
        "register": Handler_Call(base_data),
        "homework": Handler_Call(base_data),
        "style": Handler_Call(base_data),
        "week": Handler_Call(base_data),
        "reflect": Handler_Call(base_data),
        "last_day": Handler_Call(base_data),
    }

    # saver
    from save import Save_Local
    save = {
        "teacher": Save_Local(".", "aaa.csv", False),
        "class": Save_Local(".", "bbb.csv", False),
        "today": Save_Local(".", "ccc.csv", False),
        "study": Save_Local(".", "ddd.csv", False),
        "yesterday": Save_Local(".", "eee.csv", False),
        "yesterday2": Save_Local(".", "fff.csv", False),
        "yesterday3": Save_Local(".", "ggg.csv", False),

        "yesterday4": Save_Local(".", "y4.csv", False),
        "yesterday5": Save_Local(".", "y5.csv", False),
        "yesterday6": Save_Local(".", "y6.csv", False),
        "yesterday7": Save_Local(".", "y7.csv", False),
        "yesterday8": Save_Local(".", "y8.csv", False),
        "yesterday9": Save_Local(".", "y9.csv", False),
        "yesterday10": Save_Local(".", "y10.csv", False),

        "submit": Save_Local(".", "hhh.csv", False),
        "capacity": Save_Local(".", "iii.csv", False),
        "prefer": Save_Local(".", "jjj.csv", False),
        "register": Save_Local(".", "kkk.csv", False),
        "homework": Save_Local(".", "lll.csv", False),
        "style": Save_Local(".", "mmm.csv", False),
        "week": Save_Local(".", "nnn.csv", False),
        "reflect": Save_Local(".", "ooo.csv", False),
        "last_day": Save_Local(".", "ppp.csv", False),
    }

    from prepare import Base

    class Test(Base):
        config_class = False
        config_study = False
        config_teacher = False
        config_today = False
        config_yesterday = False
        config_yesterday2 = False
        config_yesterday3 = False

        config_yesterday4 = True
        config_yesterday5 = True
        config_yesterday6 = True
        config_yesterday7 = True
        config_yesterday8 = True
        config_yesterday9 = True
        config_yesterday10 = True

        config_submit = True
        config_capacity = True
        config_prefer = True
        config_register = True
        config_homework = True
        config_style = True
        config_week = True
        config_reflect = True
        config_last_day = True

    tt = Test(conf, base_data, handler, save, True)

    from reflect.province import province, city
    from encode import OneHot, Label, Eval
    p = OneHot(province())
    c = OneHot(city())
    e = Eval()
    with open("../reflect/encode_v2.pkl", "rb") as f:
        v2 = pickle.load(f)
    with open("../reflect/encode_v3.pkl", "rb") as f:
        v3 = pickle.load(f)
    chap = Label({str(k): v2[k] for k in v2}, slot=3, style="default")
    sec = Label({str(k): v3[k] for k in v3}, slot=5, style="default")

    encode = {"teacher": {"province": p, "city": c},
              "class": {},
              "today": {"chap": chap, "sec": sec},
              "study": {"v": e},
              "yesterday": {"yes_chap": chap, "yes_sec": sec},
              "yesterday2": {"yes2_chap": chap, "yes2_sec": sec},
              "yesterday3": {"yes3_chap": chap, "yes3_sec": sec},

              "yesterday4": {"yes4_chap": chap, "yes4_sec": sec},
              "yesterday5": {"yes5_chap": chap, "yes5_sec": sec},
              "yesterday6": {"yes6_chap": chap, "yes6_sec": sec},
              "yesterday7": {"yes7_chap": chap, "yes7_sec": sec},
              "yesterday8": {"yes8_chap": chap, "yes8_sec": sec},
              "yesterday9": {"yes9_chap": chap, "yes9_sec": sec},
              "yesterday10": {"yes10_chap": chap, "yes10_sec": sec},

              "style": {"style_pre": e},
              "reflect": {"ref": sec}
              }

    from generater import Generater

    conf = load_conf("generate")
    gen = Generater(conf, encode, None, None)
    return tt, gen



class PP(Proc):
    def __init__(self, base_data, pre, gen, eva):
        super().__init__(base_data, pre, gen)
        self.eva = eva

    def handle_train(self, base_data):
        base_data.drop(["tag", "time", "cnt", "rank"], axis=1, inplace=True)
        return base_data

    def model_fit(self, data):

        # 分割训练集和测试集
        data = train_local(data, test_size=0.1)
        train_data, test_data = (data[0], data[2]), (data[1], data[3])
        self.train_col = data[4]

        self.load_dis(day, train_data)

        self.fea = Feature(self.dis)
        fea_data = self.fea.handle(train_data, has_label=True, dtype="both")

        self.load_model(day, fea_data)

        test = self.fea.handle(test_data, has_label=True, dtype="both")
        test = [test[0].fillna(0.0), test[1]]
        try:
            print(day, "model accuracy: ", self.model.accuracy(test))
        except:
            pass

    def load_dis(self, day, train_data):

        if os.path.exists("dis-%s.dis" % (day.strftime("%Y-%m-%d"))):
            self.dis = Discover("xgb")
            self.dis.load("dis-%s.dis" % (day.strftime("%Y-%m-%d")),
                        "ohe-%s.dis" % (day.strftime("%Y-%m-%d")))
        else:
            self.dis = Discover("xgb")

            # Model kwargs
            type_kwargs = {"booster": "gbtree",  # 每次迭代的模型
                        "slient": 0,  # 可视训练过程
                        "scale_pos_weight": 5,  # 正样本的权重,解决失衡问题
                        "n_estimatores": 100,  # 树个数
                        "early_stopping_rounds": 15,  # 提前停止
                        "max_depth": 6,  # 树最大深度
                        "min_child_weight": 2,  # 最小叶子节点样本权重和。值越大，越容易欠拟合；值越小，越容易过拟合
                        "subsample": 0.75,  # 树样本采样
                        "colsample_bytree": 0.75,  # 树特征采样
                        "learning_rate": 0.11,  # 步长
                        "objective": "binary:logitraw",
                        # reg:linear/logistic,binary:logistic/logitraw,multi：softmax  num_class=n/softprob   num_class=n
                        "gamma": 0.1,  # 节点分裂所需的最小损失函数下降值
                        "alpha": 1,  # L1正则化项
                        "lambda": 1,  # L2正则化项
                        }
            # Fit kwargs
            model_kwargs = None
            # One-Hot kwargs
            ohe_kwargs = {"categories": "auto",  # 类别设置
                        "handle_unknown": "error",  # 类别出现异常值时的处理方式
                        "n_values": "auto",  # 每个特征的值的数量
                        }

            self.dis._train(train_data, type_kwargs, model_kwargs, ohe_kwargs)
            self.dis._save("dis-%s.dis" % (day.strftime("%Y-%m-%d")),
                        "ohe-%s.dis" % (day.strftime("%Y-%m-%d")))

    def load_model(self, day, fea_data):
        model_kwargs = {
            "penalty": "l2",  # 正则化
            "dual": False,  # 是否要转成对偶问题求解
            "C": 8.0,  # 正则化系数
            "fit_intercept": True,  # 截距是否加入模型
            "class_weight": {0: 0.9, 1: 0.1},  # 调节正负样本权重
            "random_state": 100,  # 随机种子
            "solver": "sag",  # 损失函数
            "max_iter": 10000,  # 最大迭代
            "multi_class": "ovr",  # 分类方法参数选择，ovr和multinomial，对应二分类多分类
            "warm_start": False,  # 上次模型作为初始化
        }

        if os.path.exists("model-%s.mdl" % (day.strftime("%Y-%m-%d"))):
            self.model = Lr()
            self.model.load("model-%s.mdl" % (day.strftime("%Y-%m-%d")))
        else:
            self.model = Lr(**model_kwargs)
            begin = time.time()
            fea_data = [fea_data[0].fillna(0.0), fea_data[1]]
            self.model.train(fea_data)
            print(day, "train costs: ", time.time() - begin)
            self.model.save("model-%s.mdl" % (day.strftime("%Y-%m-%d")))

    def aggregate(self, data):
        #把data 分组   给evaluate
        return data.groupby(by=["teacher_id", "class_id", "day", "chapter_id"])

    def evaluate(self, data, ):
        result = []
        with db.Handle_Sql("recom") as f:
            #这个data ？？   len为groupby之后的 长度，
            #离线表的 数据
            ssum = len(data)
            cnt = 0
            #遍历，每组唯一的  
            for item in data:
                teacher_id, class_id, day, chapter_id = item[0]

                yesterday = datetime.datetime.strptime(day, "%Y-%m-%d") + datetime.timedelta(days=-1)
                #aim 分组后元素
                aim = item[1]
                
                #拿候选集，依据 chapter id
                his = f.get_all("""SELECT chapters,sections,style FROM chapter_homework_set
                WHERE chapter_id={} ORDER BY rank ASC""".format(chapter_id))

                d = pd.DataFrame([item[0]], columns=["teacher_id", "class_id", "day", "chapter_id"])
                d["cnt"] = 1
                his = his or []
                #候选集没有 就过
                if len(his) == 0:
                    continue
                #his很多吗，只取前500
                his = pd.DataFrame(list(his[:500]), columns=["base_chapters", "base_sections", "style"])
                #加一列 方便拼接，完了删除
                #所有行都为1？
                his["cnt"] = 1
                #d就一行， his可能多行啊
                d = pd.merge(d, his, how="left", on="cnt")
                d.drop(["cnt"], axis=1, inplace=True)


                p = self.pre.new_instance(d, None, is_save=False)
                begin = time.time()
                p.prepare()
                print("prepare", time.time() - begin)

                default = {"study": {"encode-v": str([0] * 20), "raw-gap_days": 15},
                        "today": {"encode-chap": "key-base_chapters",
                                    "encode-sec": "key-base_sections"},
                        "yesterday": {}
                        }

                g = self.gen.new_instance(p, None, default)
                #dd？？ 把d 变成 字典，
                dd = read_local(d)
                begin = time.time()
                #？？？  side_output？
                r, side_output = g.trans(dd, True)
                print("generate", time.time() - begin)
                if len(r) == 0:
                    continue

                r = r[self.train_col]

                begin = time.time()
                test = self.fea.handle(r, has_label=False, dtype="both")
                test.fillna(0.0, inplace=True)
                prob = self.model._model.predict_proba(test)
                print("predict", time.time() - begin)

                begin = time.time()
                rank = [[i, j] for i, j in zip(prob, side_output)]
                rank = sorted(rank, key=lambda x: x[0][1], reverse=True)

                print("sort", time.time() - begin)

                for row in aim.itertuples(index=False):
                    if getattr(row, "label") == 1:
                        aa = {"base_chapters": getattr(row, "base_chapters"),
                            "base_sections": getattr(row, "base_sections"),
                            }
                        break
                begin = time.time()
                #aa 
                result.append(self.eva.point(rank, aa))
                print("point", time.time() - begin)
                cnt += 1
                print(__file__, "processing", cnt, "/", ssum)

        re = pd.DataFrame(result)
        re.to_csv("result-%s.csv" % (day), index=False)



base_data = log_data(day, 0, "base-{}.csv")

pre, gen = generate(base_data)
eva = Eva(point)

t = PP(base_data, pre, gen, eva)

# t.run_out_prepare()
t.run()
#训练的 之前的天数  30天
#测试  当天的
dd = log_data(day, 0)
t.predict(dd)

if __name__ == "__main__":
    
    # 用每天之前的30天数据训练模型，并进行那天的预测验证
    # 11.26-12.10 done。
    # 12.11-12.24
    begin = datetime.datetime(2019, 9, 1)

    # 12.15
    for add in range(15):
        day = begin + datetime.timedelta(days=add)
        print(day)
        exec(day)