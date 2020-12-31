
import random
import time


def str_time_prop(start, end, format, prop):
    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))
    ptime = stime + prop * (etime - stime)
    return time.strftime(format, time.localtime(ptime))


def get_random_date():
    return str_time_prop("1/1/2020 1:30 PM", "12/31/2020 4:50 AM" , '%m/%d/%Y %I:%M %p',  random.random())



class Article():
	def __init__(self,title = None, publisher = None , date = None , author = None , key_words = None):
		self.title = title
		self.publisher = publisher if publisher else "NY times"
		self.date = date if date else get_random_date()
		self.author = author if author else "Random Person"
		self.key_words = key_words 
	
	def get_response(self):
		return {'title' : self.title , 
				'publisher' : self.publisher,
				'date' : self.date,
				'author' : self.author,
				'key_words' : self.key_words
				}



