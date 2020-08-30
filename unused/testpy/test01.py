import math
import argparse


parser = argparse.ArgumentParser(description='Calculate area of a rectangle')
parser.add_argument('-w', '--width', type=int, metavar='', required=True, help='Width of Rectangle')
parser.add_argument('-H', '--height', type=int, metavar='', required=True, help='Height of Rectangle')
parser.add_argument('-z', type=bool, metavar='', help='test quiet')
parser.add_argument('-x', type=bool, metavar='', help='test verbose')

group = parser.add_mutually_exclusive_group()
group.add_argument('-q', '--quiet', action='store_true', help='print quiet')
group.add_argument('-v', '--verbose', action='store_true', help='print verbose')

args = parser.parse_args()


def rect_area(radius, height):
    area = radius * height
    return area


if __name__ == '__main__':
    area = rect_area(args.width, args.height, args)

    if args.quiet:
        print(area)
    elif args.verbose:
        print('Area: %f | Width: %f | Height: %f' % (area, args.width, args.height))
    else:
        print('Area: %f' % area)
