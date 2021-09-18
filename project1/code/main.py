import argparse
import ord_lstsq, plot
import numpy as np

parser = argparse.ArgumentParser(
    description='Explore different regression methods' \
                +'and evaluate which one is best.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('-m', '--method',
    type=str,
    choices=['LeastSquare', 'Ridge', 'Lasso'],
    help='Choose which method you want to use.'
)

parser.add_argument('-p', '--polynomial',
    type=int,
    default=3,
    choices=[3, 4, 5],
    help='Choose polynomial degree, max 5.'
)

parser.add_argument('-n', '--num_points',
    type=int,
    default=100,
    help='Choose number of gridpoints'    
)

parser.add_argument('-s', '--scaling',
    type=str,
    default=None,
    choices=['M', 'S', 'N'],
    help='Provide what scaling you want.'
)

parser.add_argument('-r', '--resampling',
    type=str,
    default=None,
    choices=['M', 'S', 'N'],
    help='Provide what scaling you want.'
)

parser.add_argument('-lam', '--lambda',
    type=float,
    default=0,
    help='Choose a lambda'    
)



args = parser.parse_args()

n = args.num_points
p = args.polynomial

x = np.sort(np.random.uniform(size=n))
y = np.sort(np.random.uniform(size=n))
Model = ord_lstsq.Regression((x, y), ord_lstsq.FrankeFunction, P=p, eps0=0, scaling="S")
Model.fit()


if __name__ == "__main__":
    main()
