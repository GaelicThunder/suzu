"""ANSI primitives: SGR helpers, regexes, escape constants."""
import json
import os
import re
import select
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque



ESC = "\033["



RST = ESC + "0m"



def fg(n):
    return ESC + "38;5;%dm" % n



def bg(n):
    return ESC + "48;5;%dm" % n



BOLD = ESC + "1m"



DIM = ESC + "2m"



ANSI_RE = re.compile(r"\033\[[0-9;?]*[a-zA-Z]")



SGR_RE = re.compile(r"\033\[[0-9;]*m")

