import optparse
import argparse
"""
Helper script to get command line arguements
"""

""" All keys must be of the same type"""
def sorted_dict2str(dictionary):
    s = "Score, Games\n"
    sc = 0
    for k in sorted(dictionary.keys()):
        while sc < k:
            s += "{:5}, {:5}\n".format(sc, 0)
            sc+=1
        v = dictionary[k]
        s += "{:5}, {:5}\n".format(k, v)
    return s


def str2bool(v):
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
    parser = optparse.OptionParser()

    parser.add_option('-t', '--training',
        action="store", dest="training",
        help="Training flag", default="True")

    parser.add_option('-e', '--env',
        action="store", dest="env",
        help="Training Environment", default="snake")

    parser.add_option('-v', '--verbose',
        action="store_true", dest="v",
        help="Verbose mode", default=False)

    options, args = parser.parse_args()
    return (options.training, options.env, options.v)
