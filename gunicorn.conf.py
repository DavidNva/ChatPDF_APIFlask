bind = "0.0.0.0:10000"
worker_class = "gevent"
workers = 1
threads = 2  # Reducido de 4 a 2
timeout = 300
keepalive = 2
max_requests = 1
max_requests_jitter = 5
preload_app = True  # Añadido

import sys
sys.setrecursionlimit(10000)