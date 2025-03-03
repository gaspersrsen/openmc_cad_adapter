import sys
import cubit

#cubit.init()

def exec_cubit(command):
    return cubit.cmd(command)

def body_id():
    cubit.get_last_id("body")

def body_next():
    cubit.get_next_block_id()