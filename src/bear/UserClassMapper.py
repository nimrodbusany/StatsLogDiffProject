import re

def extract_user_class(line):
    return re.findall(r'"(.*?)"', line, re.DOTALL)[-1]