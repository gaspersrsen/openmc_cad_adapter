import sys
sys.path.append('/opt/cubit/bin')
sys.path.append('/opt/Coreform-Cubit-2025.1/bin/')
sys.path.append('/home/sharef8/Documents/Coreform-Cubit-2025.3/bin/')
import cubit

#cubit.init([])
cubit.init(['cubit','-nojournal','-nographics'])#,'-nogui'])
#cubit.reset()

def exec_cubit(command): # Pass the command to cubit
    #print(command)
    return cubit.cmd(command)

def volume_id(): # Returns volume id of last selected or created volumes, single or multiple
    return cubit.get_last_id("volume")

def mul_body_id(): # Returns volume id of last selected or created multiple volume bodies, single or multiple
    return cubit.get_last_id("body")

def block_next(): # Returns the id of next available block
    return cubit.get_next_block_id()

def mat_id(): # Returns materials id of last selected or created volumes, single or multiple
    return cubit.get_last_id("material")