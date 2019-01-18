import os

INIT_DISPLAY_OK = False
DISPLAY_AVAILABLE = True


def hide_gpu():
    """
    Hides visible CUDA devices so that computations will be run on the CPU only.
    """
    print('Hiding CUDA devices.')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def n_gpus():
    """
    Returns the number of available GPUs in the system (requires Tensorflow).
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


def init_display():
    """
    Initializes Matplotlib's backend based on the available display.
    """
    global INIT_DISPLAY_OK, DISPLAY_AVAILABLE
    if 'DISPLAY' not in os.environ or os.environ['DISPLAY'].startswith('localhost'):
        import matplotlib
        matplotlib.use('Agg')
        DISPLAY_AVAILABLE = False
    INIT_DISPLAY_OK = True
