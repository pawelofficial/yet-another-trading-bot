import logging 
import time 
from functools import wraps 
def return_on_failure(value1):
  def decorate(f):
    def applicator(*args, **kwargs):
      try:
         return f(*args,**kwargs)
      except Exception as er:
         print('Error')
         logging.info(str(er))
         return value1,str(er)
    return applicator
  return decorate



def retry_on_index_failure(value):
  def decorate(f):
    def applicator(*args, **kwargs):
      i=0
      while i<5:
        i+=1
        time.sleep(2)
        logging.info(' retrying on failure ')
        try:
           return f(*args,**kwargs)
        except IndexError as er:
           print('Error')
           return value
    return applicator
  return decorate



def retry_on_any_error(value):
  def decorate(f):
    def applicator(*args, **kwargs):
      i=0
      while i<5:
        i+=1
        time.sleep(2)
        logging.info(' retrying on error ')
        try:
           return f(*args,**kwargs)
        except Exception as er:
           logging.info(er)
           print('Error')
           return value
    return applicator
  return decorate


if __name__=='__main__':
  f()