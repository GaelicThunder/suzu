"""Runtime configuration set by the CLI (paths/URLs)."""
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



TF_REQLOG = os.path.expanduser("~/models/glm53-tf/sessions/requests.jsonl")



HERMES_DB = os.path.expanduser("~/.hermes/state.db")   # --tokens-db overrides



LANES_URL = None        # optional gateway lanes endpoint (--lanes-url)



RANK1_URL = None        # optional rank-1 metrics endpoint (--rank1-url)

