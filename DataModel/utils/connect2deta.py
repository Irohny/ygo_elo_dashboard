from deta import Deta

def connect2deta(name:str, key:str):
    deta = Deta(key)
    return deta.Base(name)