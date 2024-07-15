import argparse
from model.sgl import SGL

def parse_args():
    parser = argparse.ArgumentParser()

    # 模型参数
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--layer_num', default=3, type=int)
    parser.add_argument('--cl_reg', default=0.1, type=float)
    parser.add_argument('--reg', default=0.0001, type=float)

    # 图构建
    parser.add_argument('--dropout_ratio', default=0.1, type=float)

    # 训练参数
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--epoch_num', default=500, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    # 评价指标
    parser.add_argument('--k', default=20, type=int)
    parser.add_argument('--tao', default=0.2, type=float)

    parser.add_argument('--train_data_path', default='./dataset/yelp2018/yelp2018.train', type=str)
    parser.add_argument('--test_data_path', default='./dataset/yelp2018/yelp2018.test', type=str)
    parser.add_argument('--datadir', default='yelp2018', type=str)

    return parser.parse_args()


args = parse_args()


if __name__ == "__main__":
    model = SGL(args)
    model.run()
