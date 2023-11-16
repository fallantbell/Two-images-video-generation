class bcolors:
    GREEN = "\033[92m"  # GREEN
    YELLOW = "\033[93m"  # YELLOW
    RED = "\033[91m"  # RED
    BLUE = "\033[94m" # BLUE
    END = "\033[0m"  # RESET COLOR

def print_b(text):
    print(f"{bcolors.BLUE}{text}{bcolors.END}")

def print_y(text):
    print(f"{bcolors.YELLOW}{text}{bcolors.END}")

def print_r(text):
    print(f"{bcolors.RED}{text}{bcolors.END}")

def print_g(text):
    print(f"{bcolors.GREEN}{text}{bcolors.END}")
