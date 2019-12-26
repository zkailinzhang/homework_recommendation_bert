

from redis import ConnectionPool
from redis import StrictRedis 
import  datetime
import json 
import time 
import os 

PATH = os.path.dirname(os.path.abspath(__file__))

cha_file = os.path.join(PATH,"dic_base_cha_final.json")
sec_file = os.path.join(PATH,"dic_base_sec_final.json")
ref_file = os.path.join(PATH,"dic_reflect.json")

redis_conf={"homework": {"host": "10.19.188.126", 
                    "db": 5,
                    "port": 6379,
                     "password":'Rum5vOGhbQVQ', 
                     }}


class MYRedis(object):
    def __init__(self, env):
        self.r = StrictRedis(**redis_conf[env])

        #pool = ConnectionPool(**redis_conf[env])
        #self.r = StrictRedis(connection_pool=pool,decode_responses=True)
        self.pipe = self.r.pipeline()

    def incr(self, k):
        if self.r.exists(k):
            return self.r.incr(k)
        return self.r.set(k, 1, ex=3600 * 24 * 7)

    def do_incr(self,key):
        with self.r.pipeline() as pipe:
            pipe.multi()
            access_n = pipe.incr(key + ':access')
            success_n = pipe.incr(key + ':success')
            pipe.execute()

    def rpush(self, k, v):
        _time = self.get_time()
        if not self.r.exists(k):
            self.r.rpush(k, v)
            self.r.expire(k, _time)
            return
        return self.r.rpush(k, v)

    def lpush(self, k, v):
        _time = self.get_time()
        if not self.r.exists(k):
            self.r.lpush(k, v)
            self.r.expire(k, _time)
            return
        return self.r.lpush(k, v)

    def lrange(self,key):
        return self.r.lrange(key,0,-1)

    @staticmethod
    def get_time():
        today = datetime.date.today()
        tomorrow = today + datetime.timedelta(days=1)
        tomorrow_int = int(time.mktime(tomorrow.timetuple()))
        second = tomorrow_int - int(time.time()) + 300
        return second

    def lpop(self, k):
        return self.r.lpop(k)

    def llen(self, k):
        return self.r.llen(k)

    def get(self, k):
        # print("redis_get","###",k,self.r.get(k))
        return self.r.get(k)

    def get_json(self, k):
        res = self.get(k)
        if res is None:
            return {}
        try:
            return json.loads(res.decode('utf-8'))
        except Exception:
            return {}

    def get_int(self, k):
        res = self.r.get(k)
        try:
            return int(res)
        except Exception:
            return 0

    def set(self, k, v, ex=3600 * 24 * 7):
        return self.r.set(k, v,ex=ex)

    def set_json(self, k, data):
        assert isinstance(data, dict)
        return self.set(k, json.dumps(data))

    def del_key(self, k):
        return self.r.delete(k)

    def hset(self,name, key,value):
        return self.r.hset(name, key,value)

    def hget(self,name,key):
        return self.r.hget(name,key)

    def hexists(self,name, key):
        return self.r.hexists(name, key)

    def __del__(self):
        self.r.connection_pool.disconnect()





def update(conn,table="chapter",key = "cha_mod_0", value=0):
    # if 0 == int(k)%6:
    cha_mod_0= conn.hget(table, key)
    cha_len_ = conn.hget("lens_dic", "lens")

    cha_mod_0_new = json.loads(cha_mod_0)
    new_len_dict = json.loads(cha_len_)

    cha_mod_0_new[str(value)]= new_len_dict[table]

    new_len_dict[table] = new_len_dict[table]+1

    conn.hset(table, key,json.dumps(cha_mod_0_new))
    conn.hset("lens_dic", "lens",json.dumps(new_len_dict))

if __name__ == "__main__":

    conn = MYRedis("homework")

    update(conn,table="chapter",key="cha_mod_0",value=6001)

