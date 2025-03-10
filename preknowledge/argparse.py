import argparse

def get_cli_args():
    parser = argparse.ArgumentParser(description="解析命令行参数")
    parser.add_argument(
            '--epoch',
            type=int,
            default=10,
            help="训练轮数"
    )
    parser.add_argument(
            '--lr',
            type=float,
            default=0.01,
            help='学习率'
    )
    args = parser.parse_args()
    print(args)
    return args

if __name__ == '__main__':
    args = get_cli_args()
    print(args.epoch)
    print(args.lr)
