import os

from utils import tf, prelu, dice

# don't that large
N_CHAPTER = 5000   #1000000
N_SECTION = 12000    #1000000

N_STYLE = 10000

N_TEACHER = 100000
N_PROVINCE = 100
N_CITY = 100000
N_CLASS = 100000
N_GRADE = 13
N_EDITION = 1000
N_REFLECT = 15000 #100000

EMBEDDING_DIM = 128
HIS_DAYS=14

def get_self_or_expand_dims(aim):
    return tf.cast(tf.expand_dims(aim, -1), tf.float32)

def get_mask_zero_embedded(var_em,var_ph):
    mask = tf.equal(var_ph,0)

    mask2 = tf.concat([tf.expand_dims(~mask,-1) for i in range(EMBEDDING_DIM) ],-1)

    rst = tf.where(mask2,
            tf.nn.embedding_lookup(var_em,var_ph),
                tf.zeros([tf.shape(var_ph)[0],
                tf.shape(var_ph)[1],
                EMBEDDING_DIM])) 
    return rst

def get_history_sum_embedded(self):
    # TODO: add mask info for this operation
    his_days =['one', 'two', 'three', 'four','five','six','seven','eight','nine',
    'ten','eleven','twelve','thirteen','fourteen']

    for fir in his_days:
        key = "history_" + fir + "_chap_ph"
        embed_key = "history_" + fir + "_chap_embedded"
        setattr(self, embed_key,get_mask_zero_embedded(self.chapters_embeddings_var,
                                    getattr(self, key)))
    for fir in his_days:
        key = "history_" + fir + "_sec_ph"
        embed_key = "history_" + fir + "_sec_embedded"
        setattr(self, embed_key,get_mask_zero_embedded(self.sections_embeddings_var,
                                    getattr(self, key)))



    chap = tf.reduce_mean(self.history_one_chap_embedded, axis=-2)
    #b*x*128 b*128    b*(128*14)
    sec = tf.reduce_mean(self.history_one_sec_embedded, axis=-2)

    for fir in his_days[:0:-1]:
        key_c = "history_" + fir + "_chap_embedded"
        chap = tf.concat([chap, tf.reduce_mean(getattr(self, key_c), axis=-2)], axis=-1)
        key_s = "history_" + fir + "_sec_embedded"
        sec = tf.concat([sec, tf.reduce_mean(getattr(self, key_s), axis=-2)], axis=-1)
    
    history_chap_emb = tf.reshape(chap, [-1, HIS_DAYS, EMBEDDING_DIM])
    history_sec_emb = tf.reshape(sec, [-1, HIS_DAYS, EMBEDDING_DIM])


    chap_mean = tf.reduce_mean(history_chap_emb, axis=-2)
    sec_mean = tf.reduce_mean(history_sec_emb, axis=-2)
    return chap_mean, sec_mean

    #return tf.concat([chap, sec], axis=-1)

    

  


class Model(object):
    def __init__(self, *, use_dice=False):
        self.graph = tf.Graph()
        self.tensor_info = {}
        self.use_dice = use_dice

        with self.graph.as_default():
            # Main Inputs
            with tf.name_scope('Main_Inputs'):

                self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
                self.lr_ph = tf.placeholder(tf.float32, [], name="lr_ph")

                # with tf.name_scope("Teacher_Info"):
                self.teacher_id_ph = tf.placeholder(tf.int32, [None, ], name="teacher_id_ph")
                self.student_count_ph = tf.placeholder(tf.int32, [None, ], name="student_count_ph")
                self.province_id_ph = tf.placeholder(tf.int32, shape=[None, ], name="province_id_ph")
                self.city_id_ph = tf.placeholder(tf.int32, shape=[None, ], name="city_id_ph")
                # TODO: binary 是否金牌讲师
                self.core_type_ph = tf.placeholder(tf.int32, shape=[None, ], name="core_type_ph")

                # with tf.name_scope("Class_Info"):
                #今天 教的班级id
                self.class_id_ph = tf.placeholder(tf.int32, [None, ], name="class_id_ph")
                #课本版本
                self.edition_id_ph = tf.placeholder(tf.int32, [None, ], name="edition_id_ph")
                self.grade_id_ph = tf.placeholder(tf.int32, [None, ], name="grade_id_ph")
                #老师教的所有班级 学生总人数， int连续特征
                self.class_student_ph = tf.placeholder(tf.int32, [None, ], name="class_student_ph")
                #kefei
                #浮点连续特征
                self.cap_avg_ph = tf.placeholder(tf.float32, [None, ], name="cap_avg_ph")
                self.cap_max_ph = tf.placeholder(tf.float32, [None, ], name="cap_max_ph")
                self.cap_min_ph = tf.placeholder(tf.float32, [None, ], name="cap_min_ph")


                # with tf.name_scope("Homework_Info"):
                #候选， 召回集  首先天宇会给一个初步刷选的作业集，好几组，每组进去很多题目，暂且不管
                #粒度暂且放在一组，上，看做一个  作业，  特征属性两个chapters  sections
                #这一组中，今天这个老师 id，，布置了某一组，则吧这个 label 为1   其他布置的  为0   这样就构造了样本
                #另外chapters  sections分别可能是几个数字®️的，  类似于多lable吧， 为了保持统一长度， 所以补零
                self.today_chapters_ph = tf.placeholder(tf.int32, [None, None], name="today_chapters_ph")
                self.today_sections_ph = tf.placeholder(tf.int32, [None, None], name="today_sections_ph")
                
                #没用
                self.today_chap_mask_ph = tf.placeholder(tf.float32, [None, None], name='today_chap_mask_ph')
                self.today_chap_len_ph = tf.placeholder(tf.int32, [None, ], name='today_chap_len_ph')
                self.today_sec_mask_ph = tf.placeholder(tf.float32, [None, None], name='today_sec_mask_ph')
                self.today_sec_len_ph = tf.placeholder(tf.int32, [None, ], name='today_sec_len_ph')
                #作业的风格  是这道题的 风格， 什么预习啊 什么深度啊，，
                self.today_style_ph = tf.placeholder(tf.int32, [None, ], name='today_style_ph')

                # TODO: use three dims to capture more history info
                #这个是 这个班级前三天 给布置的作业， 比如，昨天的，仍是两个特征来表征，chap sec 每个都是多个数字，
                #所以N N 
                for fir in ['one', 'two', 'three', 'four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen']:
                    key = "history_" + fir + "_chap_ph"
                    setattr(self, key,
                                tf.placeholder(tf.int32, [None, None], name=key))
                for fir in ['one', 'two', 'three', 'four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen']:
                    key = "history_" + fir + "_sec_ph"
                    setattr(self, key,
                                tf.placeholder(tf.int32, [None, None], name=key))

                # TODO: All belows should consider the type and input
                # with tf.name_scope("Study_Info"):
                # TODO: study_vector_ph's type can change?
                #kefei  这个班级的学习能力  类似于期中考试，  这个班级  表征为 20维的向量  int
                self.study_vector_ph = tf.placeholder(tf.float32, [None, 20], name="study_vector_ph")
                #上面的结果  什么时候评测的    连续值，隔的天数
                self.gap_days_ph = tf.placeholder(tf.int32, [None, ], name="gap_days_ph")

                # with tf.name_scope("Submit_Info"):  这个班级 一个月的app内 作业提交率 ，连续float特征
                self.month_submit_rate_ph = tf.placeholder(tf.float32, [None, ], name="month_submit_rate_ph")

                # with tf.name_scope("Capacity_Info"):  地区区域整体能力   也是  float连续特征
                self.region_capacity_ph = tf.placeholder(tf.float32, [None, ], name="region_capacity_ph")

                # with tf.name_scope("Prefer_Info"):
                #老师 在这个班级 上，，喜欢布置作业的 难度 和时间   float连续值特征
                self.prefer_assign_time_avg_ph = tf.placeholder(tf.float32, [None, ],
                                                                name="prefer_assign_time_avg_ph")
                self.prefer_assign_time_var_ph = tf.placeholder(tf.float32, [None, ],
                                                                name="prefer_assign_time_var_ph")
                self.prefer_assign_rank_avg_ph = tf.placeholder(tf.float32, [None, ],
                                                                name="prefer_assign_rank_avg_ph")
                self.prefer_assign_rank_var_ph = tf.placeholder(tf.float32, [None, ],
                                                                name="prefer_assign_rank_var_ph")

                # with tf.name_scope("Register_Info"):  老师 注册app的 时间，int连续值特征
                self.register_diff_ph = tf.placeholder(tf.int32, [None, ], name="register_diff_ph")

                # with tf.name_scope("HomeworkCount_Info"):  老师 布置了多少题目   int连续值特征
                #是总共布置的吗  从注册app？？？
                self.homework_count_ph = tf.placeholder(tf.int32, [None, ], name="homework_count_ph")

                # with tf.name_scope("Style_Info"):
                # TODO: use 3 dims  老师 作业 的风格  ？  一个特征域
                for fir in ["1", "2", "3", "4"]:
                    for sec in ["100", "010", "001", "110", "101", "011", "111"]:
                        key = "style_" + fir + "0" + sec + "_ph"
                        setattr(self, key,
                                tf.placeholder(tf.int32, [None, ], name=key))

                # with tf.name_scope("WeekHomeworkCount_Info"):
                #这周 老师布置作业，，作业率，，怎么float ？？   连续值特征
                self.week_count_ph = tf.placeholder(tf.float32, [None, ], name="week_count_ph")

                # with tf.name_scope("Reflect_Info"):
                # TODO: explore more graceful  映射 作业类目
                self.reflect_value_ph = tf.placeholder(tf.int32, [None, None], name="reflect_value_ph")
                self.reflect_mask_ph = tf.placeholder(tf.float32, [None, None], name="reflect_mask_ph")
                self.reflect_len_ph = tf.placeholder(tf.int32, [None, ], name="reflect_len_ph")

                # with tf.name_scope("Lastdat_Info"):  昨天布置的 个数  int连续实值
                self.lastday_count_ph = tf.placeholder(tf.int32, [None, ], name="lastday_count_ph")

            # Embedding layer
            with tf.name_scope('Main_Embedding_layer'):
                # almost done
                with tf.name_scope("Others"):
                    # teacher
                    with tf.name_scope("Teacher"):
                        self.teacher_id_embeddings_var = tf.get_variable("teacher_id_embeddings_var",
                                                                         [N_TEACHER, EMBEDDING_DIM], )
                        # tf.summary.histogram('teacher_id_embeddings_var', self.teacher_id_embeddings_var)
                        self.teacher_id_embedded = tf.nn.embedding_lookup(self.teacher_id_embeddings_var,
                                                                          self.teacher_id_ph, )

                        self.province_id_embeddings_var = tf.get_variable("province_id_embeddings_var",
                                                                          [N_PROVINCE, EMBEDDING_DIM])
                        # tf.summary.histogram('province_id_embeddings_var', self.province_id_embeddings_var)
                        self.province_id_embedded = tf.nn.embedding_lookup(self.province_id_embeddings_var,
                                                                           self.province_id_ph)

                        self.city_id_embeddings_var = tf.get_variable("city_id_embeddings_var",
                                                                      [N_CITY, EMBEDDING_DIM])
                        # tf.summary.histogram('city_id_embeddings_var', self.city_id_embeddings_var)
                        self.city_id_embedded = tf.nn.embedding_lookup(self.city_id_embeddings_var,
                                                                       self.city_id_ph)

                        self.core_type_embeddings_var = tf.get_variable("core_type_embeddings_var",
                                                                        [2, EMBEDDING_DIM])
                        # tf.summary.histogram('core_type_embeddings_var', self.core_type_embeddings_var)
                        self.core_type_embedded = tf.nn.embedding_lookup(self.core_type_embeddings_var,
                                                                         self.core_type_ph)
                        # just to use embedded for var,maybe tf.identify?
                        self.student_count_embedded = get_self_or_expand_dims(self.student_count_ph)

                    with tf.name_scope("Class"):
                        self.class_id_embeddings_var = tf.get_variable("class_id_embeddings_var",
                                                                       [N_CLASS, EMBEDDING_DIM])
                        # tf.summary.histogram('class_id_embeddings_var', self.class_id_embeddings_var)
                        self.class_id_embedded = tf.nn.embedding_lookup(self.class_id_embeddings_var,
                                                                        self.class_id_ph)

                        self.edition_id_embeddings_var = tf.get_variable("edition_id_embeddings_var",
                                                                         [N_EDITION, EMBEDDING_DIM])
                        # tf.summary.histogram('edition_id_embeddings_var', self.edition_id_embeddings_var)
                        self.edition_id_embedded = tf.nn.embedding_lookup(self.edition_id_embeddings_var,
                                                                          self.edition_id_ph)

                        self.grade_id_embeddings_var = tf.get_variable("grade_id_embeddings_var",
                                                                       [N_GRADE, EMBEDDING_DIM])
                        # tf.summary.histogram('grade_id_embeddings_var', self.grade_id_embeddings_var)
                        self.grade_id_embedded = tf.nn.embedding_lookup(self.grade_id_embeddings_var,
                                                                        self.grade_id_ph)
                        # just to use embedded for var,maybe tf.identify?
                        #连续值 dense 本身有意义的直接喂入
                        self.class_student_embedded = get_self_or_expand_dims(self.class_student_ph)
                        self.cap_avg_embedded = get_self_or_expand_dims(self.cap_avg_ph)
                        self.cap_max_embedded = get_self_or_expand_dims(self.cap_max_ph)
                        self.cap_min_embedded = get_self_or_expand_dims(self.cap_min_ph)

                    with tf.name_scope("Study"):
                        # just to use embedded for var,maybe tf.identify?
                        self.study_vector_embedded = tf.cast(self.study_vector_ph, tf.float32)
                        self.gap_days_embedded = get_self_or_expand_dims(self.gap_days_ph)

                    with tf.name_scope("Submit"):
                        # just to use embedded for var,maybe tf.identify?
                        self.month_submit_rate_embedded = get_self_or_expand_dims(self.month_submit_rate_ph)

                    with tf.name_scope("Capacity"):
                        # just to use embedded for var,maybe tf.identify?
                        self.region_capacity_embedded = get_self_or_expand_dims(self.region_capacity_ph)

                    with tf.name_scope("Prefer"):
                        # just to use embedded for var,maybe tf.identify?
                        self.prefer_assign_time_avg_embedded = get_self_or_expand_dims(
                            self.prefer_assign_time_avg_ph)
                        self.prefer_assign_time_var_embedded = get_self_or_expand_dims(
                            self.prefer_assign_time_var_ph)
                        self.prefer_assign_rank_avg_embedded = get_self_or_expand_dims(
                            self.prefer_assign_rank_avg_ph)
                        self.prefer_assign_rank_var_embedded = get_self_or_expand_dims(
                            self.prefer_assign_rank_var_ph)

                    with tf.name_scope("Register"):
                        self.register_diff_embedded = get_self_or_expand_dims(self.register_diff_ph)

                    with tf.name_scope("HomeworkCount"):
                        self.homework_count_embedded = get_self_or_expand_dims(self.homework_count_ph)

                    with tf.name_scope("WeekHomeworkCount"):
                        self.week_count_embedded = get_self_or_expand_dims(self.week_count_ph)

                    with tf.name_scope("Lastday"):
                        self.lastday_count_embedded = get_self_or_expand_dims(self.lastday_count_ph)

                # TODO: homework and reflect and style
                with tf.name_scope("Style"):
                    for fir in ["1", "2", "3", "4"]:
                        for sec in ["100", "010", "001", "110", "101", "011", "111"]:
                            key = "style_" + fir + "0" + sec + "_ph"
                            embed_key = "style_" + fir + "0" + sec + "_embedded"
                            setattr(self, embed_key,
                                    get_self_or_expand_dims(getattr(self, key)))

                # homework
                with tf.name_scope("Homework"):
                    self.style_embeddings_var = tf.get_variable("style_embeddings_var",
                                                                [N_STYLE, EMBEDDING_DIM])
                    self.chapters_embeddings_var = tf.get_variable("chapters_embeddings_var",
                                                                   [N_CHAPTER, EMBEDDING_DIM])
                    self.sections_embeddings_var = tf.get_variable("sections_embeddings_var",
                                                                   [N_SECTION, EMBEDDING_DIM])
                    # tf.summary.histogram('homework_embeddings_var', self.homework_embeddings_var)
                    

                    self.today_chapters_embedded = get_mask_zero_embedded(self.chapters_embeddings_var,
                                                                          self.today_chapters_ph)
                    self.today_sections_embedded = get_mask_zero_embedded(self.sections_embeddings_var,
                                                                          self.today_sections_ph)

                    self.history_chap_embedded,self.history_sec_embedded = get_history_sum_embedded(self)

                    self.today_style_embedded = tf.nn.embedding_lookup(self.style_embeddings_var,
                                                                       self.today_style_ph)
                # reflect
                with tf.name_scope("Reflect"):
                    self.reflect_embeddings_var = tf.get_variable("reflect_embeddings_var",
                                                                  [N_REFLECT, EMBEDDING_DIM])
                    # tf.summary.histogram('reflect_embeddings_var', self.reflect_embeddings_var)
                    self.reflect_value_embedded = tf.nn.embedding_lookup(self.reflect_embeddings_var,
                                                                         self.reflect_value_ph)

    def other_inputs(self):
        """
        if use this method,must to rewrite train and test methods.
        :return: list of Var
        """
        self.use_others = False
        return []

    def build_fcn_net(self, inp, use_dice=False):
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=1)


            with tf.name_scope("Out"):
                bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
                dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
                if use_dice:
                    dnn1 = dice(dnn1, name='dice_1')
                else:
                    dnn1 = prelu(dnn1, 'prelu1')

                dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
                if use_dice:
                    dnn2 = dice(dnn2, name='dice_2')
                else:
                    dnn2 = prelu(dnn2, 'prelu2')
                dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
                self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

            with tf.name_scope('Metrics'):
                # Cross-entropy loss and optimizer initialization
                # 'core_type_ph': [1, 1, 0,..],

                ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
                self.loss = ctr_loss
                # tf.summary.scalar('loss', self.loss)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_ph).minimize(self.loss)
                # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr_ph).minimize(self.loss)
                # Accuracy metric
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
                # tf.summary.scalar('accuracy', self.accuracy)

            self.merged = tf.summary.merge_all()

    def train(self, sess, inps):
        loss, accuracy, _ = sess.run(
            [self.loss, self.accuracy, self.optimizer],
            feed_dict=self.get_train_feed_dict(inps)
        )
        return loss, accuracy

    def train_with_dict(self, sess, inps):
        assert isinstance(inps, dict), "inps must be dict!"
        loss, accuracy, _ = sess.run(
            [self.loss, self.accuracy, self.optimizer],
            feed_dict=self.get_train_feed_dict_with_dict(inps)
        )
        return loss, accuracy

    def test_with_dict(self, sess, inps):
        assert isinstance(inps, dict), "inps must be dict!"
        probs,loss, accuracy = sess.run(
            [self.y_hat ,self.accuracy, self.optimizer],
            feed_dict=self.get_test_feed_dict_with_dict(inps)
        )
        return probs,loss, accuracy

    def predict_with_dict(self, sess, inps):
        assert isinstance(inps, dict), "inps must be dict!"
        probs = sess.run(
            [self.y_hat],
            feed_dict=self.get_predict_feed_dict_with_dict(inps)
        )
        return probs

    def calculate(self, sess, inps):
        probs, loss, accuracy = sess.run(
            [self.y_hat, self.loss, self.accuracy],
            feed_dict=self.get_calculate_feed_dict(inps)
        )
        return probs, loss, accuracy

    def calculate_with_dict(self, sess, inps):
        probs, loss, accuracy = sess.run(
            [self.y_hat, self.loss, self.accuracy],
            feed_dict=self.get_calculate_feed_dict_with_dict(inps)
        )
        return probs, loss, accuracy

    def calculate_with_dict2(self, sess, inps):
        probs, loss = sess.run(
            [self.y_hat, self.loss],
            feed_dict=self.get_calculate_feed_dict_with_dict(inps)
        )
        return probs, loss

    def get_train_feed_dict(self, inps):
        raise NotImplementedError()

    def get_train_feed_dict_with_dict(self, inps):
        raise NotImplementedError()
    def get_test_feed_dict_with_dict(self, inps):
        raise NotImplementedError()
    def get_predict_feed_dict_with_dict(self, inps):
        raise NotImplementedError()
    def get_calculate_feed_dict(self, inps):
        raise NotImplementedError()

    def get_calculate_feed_dict_with_dict(self, inps):
        raise NotImplementedError()

    def save(self, sess, path):
        #saver = tf.train.Saver()

        pos = path.rfind("_")
        pre = path[:pos+1]
        num = int(path[pos+1:])
        for i in range(num):
            mdoelpath = pre+ str(i)+ ".meta"
            pth = pre+ str(i)+ "*"
            if os.path.exists(mdoelpath):
                os.system("rm {}".format(pth))

        self.saver.save(sess, save_path=path)

    def restore(self, sess, path):
        
        self.saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

    def build_tensor_info(self):
        """
        Use var.endwith("_ph") to collect!
        Must call __init__() and other_inputs(), because we would use
        vars(self) to build it, if don't, vars(self) will not collect the vars!

        :return:
        """
        if len(self.tensor_info) > 0:
            print("will clear items in tensor_info")
            self.tensor_info.clear()

        to_add = []
        no_need = ["target_ph", "lr_ph"]
        for k in vars(self).keys():
            if k in no_need:
                continue
            if k.endswith("_ph"):
                to_add.append(k)
        self._tf_build_tensor_info(to_add)

    def _tf_build_tensor_info(self, ph: list):
        assert isinstance(ph, (list, tuple)), "ph must be list or tuple!"
        for i in ph:
            self.tensor_info[i] = tf.saved_model.build_tensor_info(getattr(self, i))

    def save_serving_model(self, sess, dir_path=None, version: int = 1):
        if dir_path is None:
            print("using the /current_path/model-serving for dir_path")
            dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model-serving")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        self.build_tensor_info()
        assert len(self.tensor_info) > 0, "when saving model for serving, tensor_info can't empty!"

        prediction_signature = (
            tf.saved_model.build_signature_def(
                inputs=self.tensor_info.copy(),
                outputs={'outputs': tf.saved_model.build_tensor_info(
                    self.y_hat)},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        )

        export_path = os.path.join(dir_path, str(version))
        
        #删除之前保存的，只保留最新的
        for i in range(int(version)):
            pth = os.path.join(dir_path,str(i))
            if os.path.exists(pth):
                os.system("rm -rf {}".format(pth))

        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                "serving": prediction_signature,
            },
            strip_default_attrs=True
        )
        builder.save()


class SimpleModel(Model):
    def __init__(self, *, use_dice=False):
        super().__init__(use_dice=use_dice)
        self.other_inputs()

        teacher = [
            # self.teacher_id_embedded,
            self.province_id_embedded,
            self.city_id_embedded,
            self.core_type_embedded,
            self.student_count_embedded, ]
        # 0-4
        clazz = [
            # self.class_id_embedded,
            self.edition_id_embedded,
            self.grade_id_embedded,
            self.class_student_embedded,
            self.cap_avg_embedded,
            self.cap_max_embedded,
            self.cap_min_embedded, ]
        # 5-11
        study = [
            self.study_vector_embedded,
            self.gap_days_embedded, ]
        # 12-13
        submit = [self.month_submit_rate_embedded,
                  ]
        # 14
        capacity = [self.region_capacity_embedded,
                    ]
        # 15
        prefer = [self.prefer_assign_time_avg_embedded,
                  self.prefer_assign_time_var_embedded,
                  self.prefer_assign_rank_avg_embedded,
                  self.prefer_assign_rank_var_embedded, ]
        # 16-19
        register = [self.register_diff_embedded,
                    ]
        # 20
        homeworkcount = [self.homework_count_embedded,
                         ]
        # 21
        weekcount = [self.week_count_embedded,
                     ]
        # 22
        lastday = [self.lastday_count_embedded,
                   ]
        # 23

        o = teacher + clazz + study + submit + \
            capacity + prefer + register + homeworkcount + weekcount + lastday

        # here, we do like this, not tf.concat(o,axis=-1),
        # because, issue:https://github.com/tensorflow/tensorflow/issues/24816
        # ps: style,homework,reflect don't need to do like this, proved.
        others = o[0]
        for i in o[1:]:
            others = tf.concat([others, i], axis=-1)
        others = [others]

        style = []
        for fir in ["1", "2", "3", "4"]:
            for sec in ["100", "010", "001", "110", "101", "011", "111"]:
                embed_key = "style_" + fir + "0" + sec + "_embedded"
                style.append(getattr(self, embed_key))

        homework = []
        homework.append(self.today_style_embedded)
        homework.append(tf.reduce_mean(self.today_chapters_embedded, -2))
        homework.append(tf.reduce_mean(self.today_sections_embedded, -2))
        homework.append(self.history_chap_embedded)
        homework.append(self.history_sec_embedded)
        homework.append(self.history_chap_embedded * tf.reduce_mean(self.today_chapters_embedded, -2))
        homework.append(self.history_sec_embedded * tf.reduce_mean(self.today_sections_embedded, -2))


        reflect = []
        #reflect 也是均值
        reflect.append(tf.reduce_mean(self.reflect_value_embedded, -2))

        with self.graph.as_default():
            with tf.name_scope("Concat"):
                inps = tf.concat(others + style + homework + reflect, -1)
        self.build_fcn_net(inps, self.use_dice)

    def get_train_feed_dict_with_dict(self, inps):
        feed_dict = {}
        for k, v in inps.items():
            feed_dict[getattr(self, k)] = v
        return feed_dict

    def get_test_feed_dict_with_dict(self, inps):
        feed_dict = {}
        for k, v in inps.items():
            feed_dict[getattr(self, k)] = v
        return feed_dict

    def get_calculate_feed_dict_with_dict(self, inps):
        to_avoid = ["target_ph", "lr_ph"]
        for i in to_avoid:
            if i in inps:
                inps.pop(i)
        feed_dict = {}
        for k, v in inps.items():
            feed_dict[getattr(self, k)] = v
        return feed_dict

    def get_predict_feed_dict_with_dict(self, inps):

        feed_dict = {}
        for k, v in inps.items():
            feed_dict[getattr(self, k)] = v
        return feed_dict


