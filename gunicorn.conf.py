import multiprocessing

bind = "0.0.0.0:10000"
worker_class = "gevent"
workers = 1
threads = 4
timeout = 300
keepalive = 2

# Aumentar el límite de recursión
import sys
sys.setrecursionlimit(10000)