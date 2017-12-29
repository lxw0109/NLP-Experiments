#!/usr/bin/env python3
# coding: utf-8
# File: public_utils.py
# Author: lxw
# Date: 12/29/17 5:17 PM

import logging
import os

logging.basicConfig(level=logging.DEBUG, filemode="w")

def generate_logger(logger_name):
    my_logger = logging.getLogger(logger_name)
    file_name = os.path.join(os.getcwd(), logger_name+".log")
    if os.path.isfile(file_name):
        with open(file_name, "w"):
            pass
    fh = logging.FileHandler(file_name)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    my_logger.addHandler(fh)
    return my_logger


