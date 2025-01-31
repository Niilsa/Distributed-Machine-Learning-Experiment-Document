import os
import torch
import torch.distributed as dist 


def dist_init(world_size, rank, master_addr='localhost', master_port='12355'):
    # change it to the corresponding ip addr
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    assert dist.is_initialized(), "Error! The distributed env is not initialized!"

    return True


def get_local_rank():
    # get the local rank (devices id)
    if not dist.is_initialized():
        return 0
    else:
        return dist.get_rank()


def get_world_size():
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()


def init_parameters(model):
    # Boradcast the initial gradients of the model parameters
    if get_world_size() > 1:
        for param in model.parameters():
            dist.broadcast(param.data,0)


def allreduce_average_gradients(model):
    size = dist.get_world_size()
    for param in model.parameters(): ## -- HERE
        # Initialize a tensor to store the summed gradients
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        
        # Average the gradients by dividing the summed gradients by the world size
        param.grad.data /= size

def allgather_average_gradients(model):
    size = dist.get_world_size()
    for param in model.parameters():
        # Initialize a tensor to store the gathered gradients
        gathered_gradients = [torch.zeros_like(param.grad.data) for _ in range(size)]
        dist.all_gather(gathered_gradients, param.grad.data)
        
        # Average the gathered gradients
        averaged_gradients = torch.mean(torch.stack(gathered_gradients), dim=0)
        param.grad.data = averaged_gradients