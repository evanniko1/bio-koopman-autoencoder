import argparse    

def get_args(): 
    parser = argparse.ArgumentParser(description='PyTorch Example')
    #
    parser.add_argument('--model', type=str, default='koopmanAE', help='model to train (multilayer perceptron or KAN)')
    #
    parser.add_argument('--train_size', type=float, default=0.8, help='size of the training set')
    #
    parser.add_argument('--alpha', type=int, default=1,  help='model width')
    #
    parser.add_argument('--hidden', type=int, default=2,  help='number of hidden layers')
    #
    parser.add_argument('--dataset', type=str, default='flow_noisy', metavar='N', help='dataset')
    #
    parser.add_argument('--orthogonal_projection', type=bool, default=False, help='orthogonal projection of data to higher dimensions')
    #
    parser.add_argument('--theta', type=float, default=2.4,  metavar='N', help='angular displacement')
    #
    parser.add_argument('--noise', type=float, default=0.0,  metavar='N', help='noise level')
    #
    parser.add_argument('--lr', type=float, default=1e-2, metavar='N', help='learning rate (default: 0.01)')
    #
    parser.add_argument('--wd', type=float, default=0.0, metavar='N', help='weight_decay (default: 1e-5)')
    #
    parser.add_argument('--epochs', type=int, default=600, metavar='N', help='number of epochs to train (default: 10)')
    #
    parser.add_argument('--batch', type=int, default=64, metavar='N', help='batch size (default: 10000)')
    #
    parser.add_argument('--batch_test', type=int, default=200, metavar='N', help='batch size  for test set (default: 10000)')
    #
    parser.add_argument('--plotting', type=bool, default=True, metavar='N', help='number of epochs to train (default: 10)')
    #
    parser.add_argument('--folder', type=str, default='test',  help='specify directory to print results to')
    #
    parser.add_argument('--lamb', type=float, default='1',  help='balance between reconstruction and prediction loss')
    #
    parser.add_argument('--nu', type=float, default='1e-1',  help='tune backward loss')
    #
    parser.add_argument('--eta', type=float, default='1e-2',  help='tune consistent loss')
    #
    parser.add_argument('--steps', type=int, default='8',  help='steps for learning forward dynamics')
    #
    parser.add_argument('--steps_back', type=int, default='8',  help='steps for learning backwards dynamics')
    #
    parser.add_argument('--bottleneck', type=int, default='6',  help='size of bottleneck layer')
    #
    parser.add_argument('--lr_update', type=int, nargs='+', default=[30, 200, 400, 500], help='decrease learning rate at these epochs')
    #
    parser.add_argument('--lr_decay', type=float, default='0.2',  help='PCL penalty lambda hyperparameter')
    #
    parser.add_argument('--backward', type=int, default=0, help='train with backward dynamics')
    #
    parser.add_argument('--init_scale', type=float, default=0.99, help='init scaling')
    #
    parser.add_argument('--gradclip', type=float, default=0.05, help='gradient clipping')
    #
    parser.add_argument('--pred_steps', type=int, default='1000',  help='prediction steps')
    #
    parser.add_argument('--basis_function', type=str, default='chebyshev', help='alternatives to b-splines for KANs')
    #
    parser.add_argument('--degree', type=int, default=4, help='degree for polynomials')
    #
    parser.add_argument('--seed', type=int, default='1',  help='seed value')
    #
    parser.add_argument('--policy', type=str, default='KoopmanAE',  help='training policy which can be "KoopmanAE" which train the entire process,"AE" that only trains the Autoencoder on reconstruction, "Koopman" that trains the Koopman operator, "sequential" where we train the auto encoder first then the entire process, or Custom')
    #
    parser.add_argument('--num_combinations', type=int, default=50,  help='number of combinations')
    #
    parser.add_argument('--num_samples', type=int, default=100,  help='number of samples per combination or the number of initial conditions')
    #
    parser.add_argument('--time_steps', type=int, default=50,  help='number of time steps')
    #
    parser.add_argument('--max_time', type=int, default=10,  help='this will set the interval between [0-max_time] and n time_steps ')
    #
    parser.add_argument('--load_from_checkpoint', type=bool, default=False,  help='load from checkpoint')
    #
    parser.add_argument('--early_stopping', type=bool, default=False,  help='early stopping')
    #
    parser.add_argument('--spline_knots', type=int, default=5,  help='number of spline knots')
    #
    parser.add_argument('--opt_params', type=str, default='lr,lamb,alpha,batch,steps,steps_back,gradclip,degree,spline_knots', help='list of parameters to optimize')
    #
    parser.add_argument('--num_trials', type=int, default=10, help='number of trials')
    #
    parser.add_argument('--save', type=bool, default=True, help='save the model')

    return parser.parse_args()