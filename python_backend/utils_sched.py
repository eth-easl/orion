""" Utility functions """
import torch
import time

def wrap_function(obj, func_name, queue):

    func_ptr = getattr(obj, func_name)

    def wrapper(*args): # TODO: what about kwargs?
        queue.enqueue(func_ptr, 0, args)
        print("hello from wrapper! ", obj.__class__.__name__)
        #return func_ptr(*args)

    setattr(obj, func_name, wrapper)

def overwrite_fwd(obj, queue):

    mod_dict = obj._modules
    for mod_name, mod in mod_dict.items():
        if not is_super_module(mod):
            wrap_function(mod, 'forward', queue)
        else:
            overwrite_fwd(mod, queue)
    

def profile_run(obj_name, obj, func_name, module_stack, module_tensors, func_ptr_dict):

    func_ptr = getattr(obj, func_name)

    #print("obj: ", obj._get_name(), ", module stack size is: ", len(module_stack))
    #print(func_ptr)
    if is_super_module(obj):
        mod_dict = obj._modules
        if obj._get_name() != 'BahdanauAttention': #and obj._get_name() != 'RelPartialLearnableMultiHeadAttn':
            for mod_name, mod in mod_dict.items():
                profile_run(mod_name, mod, 'forward', module_stack, module_tensors, func_ptr_dict)


    def wrapper(*args, **kwargs):
        
        stack_name = len(module_stack)
        module_stack.append(stack_name)
        #print(func_ptr, stack_name)
        func_ptr_dict[stack_name] = func_ptr

        with torch.no_grad():
            res = func_ptr(*args, **kwargs)
            #print(type(res))
            # placeholder - will be set to correct value when op runs
            if isinstance(res, tuple):
                res_new = tuple([torch.empty((1)).cuda()] * len(res))
                #print("assign a tuple: ", res_new, len(res))
            else:
                res_new = torch.empty((1)).cuda()
        
        module_tensors[stack_name] = res_new
        return res

    setattr(obj, func_name, wrapper)


def wrap_forward(obj_name, obj, func_name, queue, module_stack, module_tensors, func_ptr_dict):
    

    if is_super_module(obj):
        mod_dict = obj._modules
        if obj._get_name() != 'BahdanauAttention': #and obj._get_name() != 'RelPartialLearnableMultiHeadAttn':
            for mod_name, mod in mod_dict.items():
                wrap_forward(mod_name, mod, 'forward', queue, module_stack, module_tensors, func_ptr_dict)
    
    def wrapper(*args, **kwargs):

        stack_name = len(module_stack)
        module_stack.append(stack_name)
        func_ptr = func_ptr_dict[stack_name]

        #print("---------------- ", obj, stack_name)
        if is_super_module(obj) and (obj._get_name() != 'BahdanauAttention'): #and (obj._get_name() != 'RelPartialLearnableMultiHeadAttn'):
            
            with torch.no_grad():
                res = func_ptr(*args, **kwargs)
            return res
        else:
            
            res = module_tensors[stack_name]
            
            if obj._get_name() == 'Conv2d':
                profile=0
            else:
                profile=1

            #print(f"Enqueue operator with ouput shape {res.shape} and profile {profile}")
            queue.enqueue(func_ptr, profile, res, *args, **kwargs)
            
            #print(res.shape)
            return res

    setattr(obj, func_name, wrapper)
    


def is_super_module(obj):

    mod_dict = obj._modules
    if bool(mod_dict):
        return True
    else:
        return False
