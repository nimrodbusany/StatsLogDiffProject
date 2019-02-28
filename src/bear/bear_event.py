class BEAREvent:

    def __init__(self, ip, time, req_type, url, label, user_class, l):
        self.ip = ip
        self.time = time
        self.req_type = req_type
        self.url = url
        self.label = label
        self.user_class = user_class
        self.line = l

    def __repr__(self):
        return  str(self.time) + "  " + self.req_type + ":" + self.ip + "  " + self.label \
        + "  " + self.url + "  " + self.user_class

    def __str__(self):
        return self.__repr__()