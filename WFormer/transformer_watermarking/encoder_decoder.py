from transformer_watermarking.encoder import Restormer as Encoder
from transformer_watermarking.decoder import Message_decoder as Decoder
import torch
import torch.nn as nn
from transformer_watermarking.Noise import Noise
from thop import profile

class EncoderDecoder(nn.Module):
	'''
	A Sequential of Encoder_MP-Noise-Decoder
	'''

	def __init__(self, noise_layers):
		super(EncoderDecoder, self).__init__()
		self.encoder = Encoder()
		self.noise = Noise(noise_layers)
		self.decoder = Decoder()

	def forward(self, image, message):
		encoder_image = self.encoder(image, message)
		noise_image = self.noise([encoder_image,image])
		decoder_message = self.decoder(noise_image)

		return encoder_image, noise_image, decoder_message

if __name__ == '__main__':
	import numpy as np

	height = 128
	width = 128
	x = torch.randn((1, 3, height, width))
	model = EncoderDecoder(["Identity()"])
	message = torch.Tensor(np.random.choice([0, 1], (1, 64)))
	flops, params = profile(model, inputs=(x, message,))
	print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))