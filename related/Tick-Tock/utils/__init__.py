from datetime import datetime


def pretty_time():
    return datetime.now().strftime('%H:%M:%S')
