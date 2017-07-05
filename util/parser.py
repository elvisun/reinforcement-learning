import optparse
"""
Helper script to get command line arguements
"""

""" All keys must be o the same type"""
def sorted_dict2str(dictionary):
    s = ""
    for k in sorted(dictionary.keys()):
        v = dictionary[k]
        s += "Scored {:3}: {:5} game(s).\n".format(k, v)
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
        help="Training flag", default=True)

    options, args = parser.parse_args()
    return (options, args)
