import tensorflow.experimental.numpy as znp  # this is the name in zfit as well, keep it
from zfit import z
import tensorflow as tf
import functools
from tensorflow.python.util.object_identity import Reference
from collections import OrderedDict
import dill

jit = tf.function(autograph=False)

cache = OrderedDict()

class InvalidateCacheGlobal:

    def invalidate_cache(self):
        cache.clear()


class ZfitParameteter(InvalidateCacheGlobal):

    def __init__(self, x):
        self.x = x

    def get_value(self):
        return self.x

    def set_value(self, value):
        self.invalidate_cache()  # can be moved to a decorator, but not important now
        self.x = value


class FunctionArguments():
  """Class-wraper for input function arguments so we can use them as dictionary key"""

  def __init__(self, args):
    self.input_args = args
  
  def __hash__(self):
        #print("inside function __hash__")
        #print(self.input_args)
        #print(type(self.input_args[0]))
        return hash(dill.dumps(self.input_args[0]))

  def __eq__(self, other):
        return isinstance(other, FunctionArguments) and self._compare_args(other)

  def _compare_args(self, other):
    for arg1, arg2 in zip(self.input_args, other.input_args):
      if not arg1.__eq__(arg2):
        return False
    return True

  def __repr__(self):
    result = ''
    for arg in self.input_args:
        result += str(arg.numpy())
        result += ' '
    return result


class ZfitFunc():
    
    def cache(func):
        """Caching decorator"""

        @functools.wraps(func)
        def wraper(self, *args):
          func_args = FunctionArguments(args)
          #print("input args of function = " + str(func.__name__) + ":")
          #for arg in func_args.input_args:
            #print(arg)
            #print(type(arg))
          #print('\n')
          if str(func.__name__) in cache:
            if func_args in cache[str(func.__name__)]:
              return cache[str(func.__name__)][func_args]
            else:
              cache[str(func.__name__)][func_args] = func(self, *args)
          else:
            func_dict = OrderedDict()
            func_dict[func_args] = func(self, *args)
            cache[str(func.__name__)] = func_dict
          return cache[str(func.__name__)][func_args]
        return wraper

    def __init__(self, param) -> None:
        self.param = param

    @cache
    @jit
    def func(self, data):
        return data * self.param.get_value()

    @cache
    @jit
    def integral(self, lower, upper):
        #print("inside integral")
        #print(upper)
        #print(type(upper))
        #print(self.func(upper))
        #print(type(self.func(upper)))
        #print('\n')
        return upper * self.func(upper) - lower * self.func(lower)



param = ZfitParameteter(znp.array(22.))
func = ZfitFunc(param)
func.func(znp.array(15.))
print(cache)
func.func(znp.array(21.))
print(cache)
print(func.integral(znp.array(4.), znp.array(5.)))
print(cache)
param.set_value(znp.array(2.))
print(cache)
#func = ZfitFunc(param)
print(func.func(znp.array(10.)))
print(cache)