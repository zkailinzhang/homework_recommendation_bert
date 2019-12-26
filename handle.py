import sys, time, os
import tensorflow as tf

import utils

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

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save")
MODLE_PATH = os.path.join(BASE_PATH, "model")
SERVE_PATH = os.path.join(BASE_PATH, "serving")
os.makedirs(MODLE_PATH, exist_ok=True)
os.makedirs(SERVE_PATH, exist_ok=True)


def prepare_data(data):
    def to_int(l):
        return [int(i) for i in l]

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

    chap_or_sec = ["today_chapters_ph", "today_sections_ph",
                   "history_one_chap_ph", "history_one_sec_ph",
                   "history_two_chap_ph", "history_two_sec_ph",
                   "history_three_chap_ph", "history_three_sec_ph",
                   "reflect_value_ph"
                   ]
    for item in chap_or_sec:
        aim = [to_int(str(i[item]).strip().split(",")) for i in data]
        length = max([len(i) for i in aim])
        for k, v in enumerate(aim):
            aim[k] = v + [0] * (length - len(v))
        resp[item] = aim

    today_style_ph = ["today_style_ph"]
    for item in today_style_ph:
        resp[item] = [STYLE_MAP.get(i[item], 0) for i in data]

    style_ph = ["style_ph"]
    for item in style_ph:
        if item != "style_ph":
            raise RuntimeError("style_ph!")
        for s in STYLE:
            resp["style_" + s + "_ph"] = [i[item][s] for i in data]

    item = "label"
    resp["target_ph"] = [[0, 1] if i[item] == 1 else [1, 0] for i in data]

    return resp


if __name__ == "__main__":

    file = "/data/lishuang/handle-log/log/flaskapp.log.2019-09-01."
    day = "2019-09-01"

    # file = sys.argv[1]
    # file = file[:len(file) - 1]
    #
    # day = sys.argv[2]

    print_iter = 1
    serve_iter = 1
    save_iter = 1
    decay_iter = 1

    print("*" * 40)
    try:
        import handle_log

        begin = time.time()
        handle_log.log(file, day)
        print(day, "handle log costs: ", time.time() - begin)

        import handle_train

        begin = time.time()
        train_base = handle_train.generate_train_base(day)
        print(day, "handle train_base costs: ", time.time() - begin)

        import feature

        begin = time.time()
        train_data = feature.get_data_with_pandas(train_base, 128)
        print(day, "handle feature costs: ", time.time() - begin)

        import model

        begin = time.time()
        mol = model.SimpleModel()
        with tf.Session(graph=mol.graph) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            iiter = utils.get_max_model_num(MODLE_PATH)
            if iiter != -1:
                mol.restore(sess, os.path.join(MODLE_PATH, "ckpt_") + str(iiter))
            if iiter == -1:
                iiter = 0

            version = utils.get_max_serve_num(SERVE_PATH)
            if version == -1:
                version = 0
            version += 1
            lr = 0.001

            for data in train_data:
                data = prepare_data(data)
                data["lr_ph"] = lr
                loss, accuracy = mol.train_with_dict(sess, data)

                iiter += 1
                if iiter % print_iter == 0:
                    print(iiter, loss, accuracy)
                if iiter % save_iter == 0:
                    mol.save(sess, os.path.join(MODLE_PATH, "ckpt_") + str(iiter))
                if iiter % serve_iter == 0:
                    mol.save_serving_model(sess, SERVE_PATH, str(version))
                if iiter % decay_iter == 0:
                    lr *= 0.8

        print(day, "handle train_data costs: ", time.time() - begin)
    # except Exception as e:
    #     print(e,)
    finally:
        print("*" * 40)
