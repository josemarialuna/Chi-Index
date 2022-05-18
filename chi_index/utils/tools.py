from datetime import datetime
import time


class Tools:

    def __init__(self, time_format):
        self.time_format = time_format

    def whatTimeIsIt(self):
        return datetime.datetime.fromtimestamp(time.time()).strftime(self.time_format)
