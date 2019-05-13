import sys


output_normal = sys.stdout
output_error = sys.stderr


def log(string, type="info"):
    if type == "info":
        print("\033[92m[i]\033[0m    {}".format(string), file=output_normal)
    elif type == "detail":
        print("       -> {}".format(string), file=output_normal)
    elif type == "subdetail":
        print("          {}".format(string), file=output_normal)
    elif type == "warn" or type == "warning":
        print("\033[93m[!]\033[0m    {}".format(string), file=output_error)
    elif type == "error":
        print("\033[91m[!]\033[0m    {}".format(string), file=output_error)


def handle(exception):
    log(str(exception), "detail")
    tb = exception.__traceback__.tb_next
    while tb is not None:
        log(
            "* {}:{}".format(str(tb.tb_frame.f_code.co_filename), str(tb.tb_frame.f_lineno)),
            "subdetail"
        )
        tb = tb.tb_next


def set_output(normal, error):
    global output_normal, output_error

    output_normal = normal
    output_error = error


def set_silent():
    global output_normal, output_error

    output_normal = open("/dev/null", "w")
    output_error = open("/dev/null", "w")
