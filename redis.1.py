

from redis import ConnectionPool
from redis import StrictRedis 
import  datetime
import json 
import time 
import redis 

import redis
import os 

PATH = os.path.dirname(os.path.abspath(__file__))

cha_file = os.path.join(PATH,"dic_base_cha_final.json")
sec_file = os.path.join(PATH,"dic_base_sec_final.json")
ref_file = os.path.join(PATH,"dic_reflect.json")




conn = redis.Redis(host='127.0.0.1', port=6379)
# 可以使用url方式连接到数据库
# conn = Redis.from_url('redis://@localhost:6379/1')
conn.set('name', 'LinWOW')
print(conn.get('name'))

cha_len =0
cha_mod_0,cha_mod_1 ,cha_mod_2,cha_mod_3,cha_mod_4,cha_mod_5={},{},{},{},{},{}

with open(cha_file,'r') as jf:
    cha = json.loads(jf.read())
    cha_len = len(cha)
    for k,v in cha.items():
        mod = int(k)%6 

        if mod ==0:
            cha_mod_0[k] =v
        elif mod ==1:
            cha_mod_1[k] =v
        elif mod ==2:
            cha_mod_2[k] =v
        elif mod ==3:
            cha_mod_3[k] =v
        elif mod ==4:
            cha_mod_4[k] =v
        elif mod ==5:
            cha_mod_5[k] =v


conn.hset("chapter", "cha_mod_0",json.dumps(cha_mod_0))
conn.hset("chapter", "cha_mod_1",json.dumps(cha_mod_1))
conn.hset("chapter", "cha_mod_2",json.dumps(cha_mod_2))
conn.hset("chapter", "cha_mod_3",json.dumps(cha_mod_3))
conn.hset("chapter", "cha_mod_4",json.dumps(cha_mod_4))
conn.hset("chapter", "cha_mod_5",json.dumps(cha_mod_5))
conn.hset("chapter", "cha_len",json.dumps(cha_len))

cha_mod_00= conn.hget("chapter", "cha_mod_0")
cha_len_= conn.hget("chapter", "cha_len")
print()
cha_mod_0_new = json.loads(cha_mod_00)
if 0 == int(k)%6: 
    cha_mod_0_new[k]= json.loads(cha_len_)host

r.hdel("hash1", "k1")    # 删除一个键值对