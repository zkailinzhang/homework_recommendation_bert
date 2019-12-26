import pyinotify
import os

PATH = os.path.dirname(os.path.abspath(__file__))
WATCH_PATH = "/data/lishuang/handle-log/log"

has_observed = ["2019-08-22", "2019-08-25", "2019-08-26", "2019-08-27",
                "2019-08-28",
                ]

# has_observed = []

prefix = len("flaskapp.log.")


class EventHandler(pyinotify.ProcessEvent):

    def __init__(self, has_observed=None):
        super(EventHandler, self).__init__()
        self.has_observed = has_observed or []

    def process_IN_CREATE(self, event):
        file_name = event.name
        to_test = file_name[prefix:len(file_name) - 2]

        if to_test in self.has_observed:
            pass
        else:
            self.has_observed.append(to_test)
            self.has_observed = self.has_observed[-30:]

            """do handle log and train"""

            os.system("""python3 {} {} {} &""".format(os.path.join(PATH, "handle.py"), event.pathname, to_test))

    def process_IN_DELETE(self, event):
        pass


def main():
    wm = pyinotify.WatchManager()
    wm.add_watch(WATCH_PATH, pyinotify.ALL_EVENTS, rec=False)

    eh = EventHandler(has_observed=has_observed)

    notifier = pyinotify.Notifier(watch_manager=wm, default_proc_fun=eh)

    notifier.loop()


if __name__ == "__main__":
    main()
