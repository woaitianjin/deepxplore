'''This script automatically generate result of digits0-9'''
import os

# parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
# parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
# parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
# parser.add_argument('step', help="step size of gradient descent", type=float)
# parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
# args = parser.parse_args()

def run():
    for digit in range(10): # for 10 different digits
        for ratio in [1,2,5]: # tuning params for diff/gan ratios in iterations
            for weighted_diff in [0.5, 1, 2]:
                for rate in [0.0005,0.0002,0.0001, 0.002,0.001]: # diffNetwork learning rat
                    cmd = "python2.7 ganDiff.py %f 1 %f 0 %d %d"%((weighted_diff, rate, digit, ratio))
                    print(cmd)
                    os.system(cmd)



if __name__ == '__main__':
    run()