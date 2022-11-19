import logging 
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






if __name__=='__main__':
    pass