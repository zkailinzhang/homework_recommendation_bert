import pyinotify
import os
import sys 

PATH = os.path.dirname(os.path.abspath(__file__))
#WATCH_PATH = "/data/lishuang/handle-log/log"
WATCH_PATH = "/data/zhangkl/turing_new/handle-log"


# has_observed = []
prefix = len("flaskapp.log.")
first_log =""

class EventHandler(pyinotify.ProcessEvent):

    def __init__(self,):
        super(EventHandler, self).__init__()

    def process_IN_CREATE(self, event):
        file_name = event.name
        print(file_name)
        #

        first_log_tmp = file_name[prefix:len(file_name) - 2]
        global first_log
        while 1:
            if file_name[len(file_name) - 2:] =="gz":
                break
            if first_log == "":             
                first_log = first_log_tmp
                break

            if first_log == first_log_tmp:
                
                """do handle log and train"""
                print("do handle log and train")
                os.system("""python3 {} {} &""".format(os.path.join(PATH, "handle.bgru_online.py"), first_log_tmp))
                first_log =""
                break

    def process_IN_DELETE(self, event):
        pass



def main():
    wm = pyinotify.WatchManager()
    wm.add_watch(WATCH_PATH, pyinotify.ALL_EVENTS, rec=False)

    eh = EventHandler()

    notifier = pyinotify.Notifier(watch_manager=wm, default_proc_fun=eh)

    notifier.loop()


if __name__ == "__main__":
    main()
