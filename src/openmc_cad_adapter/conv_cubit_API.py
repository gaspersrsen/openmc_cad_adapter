import sys
sys.path.append('/opt/cubit/bin')
sys.path.append('/opt/Coreform-Cubit-2025.1/bin/')
# import cubit

# #cubit.init([])
# cubit.reset()

def exec_cubit(command):
    print(command)
    #return cubit.cmd(command)

def body_id():
    return 1
    #cubit.get_last_id("body")

def body_next():
    return 2
    #cubit.get_next_block_id()