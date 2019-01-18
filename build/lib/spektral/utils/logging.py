import os
import time

LOGFILE = None
TIME_STACK = []


def init_logging(name=None):
    """
    Creates a log directory with an empty log.txt file. 
    :param name: custom name for the log directory (default \"%Y-%m-%d-%H-%M-%S\")
    :return: string, the relative path to the log directory
    """
    global LOGFILE
    if name is None:
        name = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = './logs/%s/' % name
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    LOGFILE = log_dir + 'log.txt'
    return log_dir


def log(message, print_string=True):
    """
    Prints a message to stdout and writes it to the logfile (requires user to
    call init_logging() at least once in order to save to file).
    :param message: the string to log;
    :param print_string: whether to print the string to stdout;
    """
    global LOGFILE
    message = str(message)
    if print_string:
        print(message)
    if not message.endswith('\n'):
        message += '\n'
    if LOGFILE:
        with open(LOGFILE, 'a') as f:
            f.write(message)


def tic(message=None, print_string=True):
    """
    Start counting.
    :param message: additional message to print;
    :param print_string: whether to print the string to stdout;
    """
    TIME_STACK.append(time.time())
    if message:
        log(str(message), print_string=print_string)


def toc(message=None, print_string=True):
    """
    Stop counting-
    :param message: additional message to print;
    :param print_string: whether to print the string to stdout;
    """
    fmt = 'Elapsed: {:.2f}s'
    try:
        output = fmt.format(time.time() - TIME_STACK.pop())
        if message:
            output = str(message) + '\n' + output
        log(output, print_string=print_string)
    except IndexError:
        print("You have to tic() before you toc()\n")


def model_to_str(model):
    """
    Converts a Keras model to a string.
    :param model: a Keras model;
    :return: the output of `model.summary()` as a string;
    """
    def to_str(line):
        model_to_str.output += str(line) + '\n'
    model_to_str.output = ''
    model.summary(print_fn=lambda x: to_str(x))
    return model_to_str.output
