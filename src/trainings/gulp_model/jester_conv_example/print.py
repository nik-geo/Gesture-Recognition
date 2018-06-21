import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
def main():
	pytorch_network = model = torch.load('checkpoint.pth.tar')

	# The most useful, just print the network
	print(pytorch_network)

	# Also useful: will only print those layers with params
	state_dict = pytorch_network.state_dict()
	print(util.state_dict_layer_names(state_dict))

if __name__ == '__main__':
    main()
