import os
import tensorflow as tf
import contextlib
import MySQLdb
import pandas as pd

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import *
#from tensorflow.python.ops.rnn_cell_impl import  _Linear
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear
from tensorflow import keras
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from keras import backend as K


class QAAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
        Args:
        num_units: int, The number of units in the GRU cell.
        activation: Nonlinearity to use.  Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
        kernel_initializer: (optional) The initializer to use for the weight and
        projection matrices.
        bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(QAAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        new_h = (1. - att_score) * state + att_score * c
        return new_h, new_h


class VecAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h


def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """
    #raw_arr  [[y_pred,y_true],[y_pred,y_true],[y_pred,y_true],[y_pred,y_true]]
    #按d[0]排序  就是 按y_pred 排序，  
    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)
    #arr[:][1] 真值   arr[:][0] 预测值

    def recall_precise_f1(arr):
        row = 0
        col = 0
        aim = 0
        for record in arr:
            if record[1] == 1:
                row += 1

            if record[0] >= 0.5:
                col += 1
            if (record[1] == 1 and record[0] >= 0.5):
                aim += 1
        
        try: 
            r = aim / row
        except:
            r = 0.0           
            pass            
        try: 
            p = aim / col
        except: 
            p = 0.0          
            pass
    
        if p == 0 and r ==0:
            return 0.0,0.0,0.0
        else:
            return r, p, (2 * r * p) / (r + p)


    #record[1] 是真值   
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            #真值中的正例
            pos += 1
        else:
            #真值中的负例
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp / neg, tp / pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y
    r, p, f1 = recall_precise_f1(arr)
    return auc, r, p, f1
    

def point(rank, aa):
    #为啥会返回 0 1 2 3 4
    tag = -1

    for num, ii in enumerate(rank, 1):
        ii = ii[1]
        for k, v in aa.items():
            if ii.get(k, None) == v:
                continue
            else:
                break
        else:
            #和基准一样的排第几，
            tag = num
        if tag != -1:
            break

    evaa = [1, 3, 5, 10, 15]

    def create(tag):
        if tag == -1:
            return -1
        else:
            if tag > max(evaa):
                return 0
            else:
                for k, v in enumerate(evaa, 1):
                    if tag > v:
                        continue
                    else:
                        return k

    tag = create(tag)

    if tag == -1:
        return [-1] * len(evaa)
    elif tag == 0:
        tt = [0] * (len(evaa))
    else:
        # print("********************")
        # print(rank[0], )
        # print(aa)
        # print("********************")
        tt = [0] * (tag - 1) + [1] * (len(evaa) - tag + 1)
    return tt


def din_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                  return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print("querry_size mismatch")
        query = tf.concat(values=[
            query,
            query,
        ], axis=1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    return output


def din_fcn_attention(query, facts, attention_size, mask, scope='null',stag='null', mode='SUM', softmax_stag=1, time_major=False,
                      return_alphas=False, forCnn=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
    query = prelu(query,scope)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    if not forCnn:
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    if return_alphas:
        return output, scores
    return output






def dice(_x, axis=-1, epsilon=0.000000001, name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha' + name, _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        input_shape = list(_x.get_shape())

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]

    # case: train mode (uses stats of the current batch)
    mean = tf.reduce_mean(_x, axis=reduction_axes)
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape)
    x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    # x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)
    x_p = tf.sigmoid(x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x


# prelu
def parametric_relu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


# prelu
def prelu(_x, scope='null'):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_" + scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(tf.convert_to_tensor(0.0, _x.dtype), _x) + _alpha * tf.minimum(
            tf.convert_to_tensor(0.0, _x.dtype), _x)


# belows, use for mysql

#本地做客端口映射 
#ssh -N -L 7777:10.9.98.136:3306 zhangkl@106.75.22.248

db_map = {"recom": {"host": "127.0.0.1",  #10.9.98.136
                    "user": "dba_user",
                    "passwd": "dba_user_!@#",
                    "db": "chapter_recommend_offline",
                    "port": 7777, },    #3306
          "recom_turing": {"host": "127.0.0.1",   #10.9.98.136
                           "user": "dba_user",
                           "passwd": "dba_user_!@#",
                           "db": "chapter_recommend_turing",
                           "port": 7777, }, #3306
          "wechat": {"host": "10.215.48.135",
                     "user": "lishuang",
                     "passwd": "lishuang18",
                     "db": "wechat_service",
                     "port": 3314,
                     "charset":"utf8"},
          }


@contextlib.contextmanager
def get_coursor(db: str, cursor_type=None):
    if db not in db_map:
        raise LookupError("no db-name %s!" % (db))
    conn = MySQLdb.connect(**db_map[db])

    cursor = conn.cursor(MySQLdb.cursors.DictCursor) if cursor_type == "dict" else conn.cursor()

    yield cursor

    cursor.close()
    conn.close()


def get_pandas(data) -> pd.DataFrame:
    columns = data[0].keys()
    mid = {}
    for col in columns:
        mid[col] = []
        for elem in data:
            mid[col].append(elem.get(col))
    return pd.DataFrame(mid)


def load_conf(section: str = "config") -> dict:
    import configparser
    PATH = os.path.dirname(os.path.abspath(__file__))

    conf = configparser.ConfigParser()
    conf.read(os.path.join(PATH,"config.ini")) or conf.read(os.path.join(PATH,"config.ini.example"))

    base = {}
    for k, v in conf.items(section):
        base[k] = split_config(v)
    return base


def load_conf2(section: str = "config") -> dict:
    import configparser
    PATH = os.path.dirname(os.path.abspath(__file__))

    conf = configparser.ConfigParser()
    conf.read(os.path.join(PATH,"config2.ini")) or conf.read(os.path.join(PATH,"config2.ini.example"))

    base = {}
    for k, v in conf.items(section):
        base[k] = split_config(v)
    return base


def split_config(info):
    table, key, setting = info.split(";")
    col, trans, alias = [], [], []
    for i in setting.split(","):
        a, b, c = i.split("-")
        col.append(a)
        trans.append(b)
        alias.append(c)
    return table, key, setting, col, trans, alias, \
           "SELECT " + ",".join(col) + " FROM " + table + " WHERE " + \
           " and ".join([i + "=\'{}\'" for i in key.split(",")])


CONF = load_conf()


def get_max_model_num(path: str):
    num = -1

    for i in os.listdir(path):
        o = i.split(".")[0]
        try:
            a = int(o.split("_")[1])
            if a > num:
                num = a
        except:
            pass

    return num


def get_max_serve_num(path: str):
    num = -1

    for i in os.listdir(path):
        o = i.split(".")[0]
        try:
            a = int(o)
            if a > num:
                num = a
        except:
            pass

    return num

def get_last_day_fmt():
    import datetime
    day = datetime.datetime.today()
    yes = day + datetime.timedelta(days=-1)
    #'2019-07-23'
    return yes.strftime("%Y-%m-%d")

def get_now_day_fmt():
    import datetime
    day = datetime.datetime.today()
    
    return day.strftime("%Y-%m-%d")

def get_some_day_fmt(start,num):
    '''
    'Feb 28, 2018'
    number 包括当天
    '''

    import datetime
  
    start = datetime.datetime.strptime(start, '%b %d, %Y')
    
    dates = []

    for i in range(num):
        tmp = start + datetime.timedelta(days=i)
        dates.append(tmp.strftime("%Y-%m-%d"))

    index = [i for i in range(num)]
    rst = collections.OrderedDict(zip(index,dates))


    return rst



if __name__ == "__main__":
    print(load_conf())


'''

'cap_avg_ph':-0.62
'cap_max_ph':0.65
'cap_min_ph':-1.85
'city_id_ph':142
'class_id_ph':1306480
'class_student_ph':82
'core_type_ph':1
'edition_id_ph':1
'gap_days_ph':15
'grade_id_ph':5
'history_one_chap_ph':'0'list
'history_one_sec_ph':'0'list
'history_three_chap_ph':'0'list
'history_three_sec_ph':'0'list
'history_two_chap_ph':'0'
'history_two_sec_ph':'0'   list
'history_two_chap_ph':'86874,86867'   list
'history_two_sec_ph':'86873,86875,86876'  list


'homework_count_ph':23
'label':1
'lastday_count_ph':0
'month_submit_rate_ph':0
'prefer_assign_rank_avg_ph':0.0
'prefer_assign_rank_var_ph':0.0
'prefer_assign_time_avg_ph':0.0
'prefer_assign_time_var_ph':0.0
'province_id_ph':24
'reflect_value_ph':'0'  list
'region_capacity_ph':-0.18
'register_diff_ph':593
'student_count_ph':100
'study_vector_ph':'[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]'
'style_ph':{'10001': 0, '10010': 2, '10011': 0, '10100': 0, '10101': 0, '10110': 0, '10111': 0, '20001': 0, '20010': 0, '20011': 0, '20100': 0, '20101': 0, '20110': 0, '20111': 0, ...}
'teacher_id_ph':2861982
'today_chapters_ph':'87097' list
'today_sections_ph':'87101' list 
'today_style_ph':10010 映射为实值 连续
'week_count_ph':0.0  
__len__:36



'lastday_count_ph':0
'month_submit_rate_ph':0
'prefer_assign_rank_avg_ph':0.0
'prefer_assign_rank_var_ph':0.0
'prefer_assign_time_avg_ph':6.0
'prefer_assign_time_var_ph':0.0
'province_id_ph':32
'reflect_value_ph':'0'
'region_capacity_ph':-0.32
'register_diff_ph':348
'student_count_ph':49
'study_vector_ph':'[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]'
'style_ph':{'10001': 0, '10010': 0, '10011': 0, '10100': 0, '10101': 0, '10110': 0, '10111': 0, '20001': 0, '20010': 0, '20011': 0, '20100': 0, '20101': 0, '20110': 0, '20111': 0, ...}
'teacher_id_ph':2766065
'today_chapters_ph':'86939,86947,86956,86976'
'today_sections_ph':'86943,86955,86957,86979'
'today_style_ph':40100   

单值的
多值的 
连续值的
list的

'''