Model structure is based on nvidia's autopilot (https://arxiv.org/pdf/1604.07316.pdf)

Training data is solely from Udacity sample data. (https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)
Data augmentation is applied on this data set.
Also, Tried different drop out structure to prevent overfitting.

Model Structure:
	Normalization layer (Input)
	-> Conv. layer (24 filters 5x5 kernel w stride equals 2)
	-> RELU Activation layer
	-> Conv. layer (36 filters 5x5 kernel w stride equals 2)
	-> RELU Activation layer
	-> Conv. layer (48 filters 5x5 kernel w stride equals 2)
	-> RELU Activation layer
	-> Conv. layer (64 filters 3x3 kernel w stride equals 1)
	-> RELU Activation layer
	-> Conv. layer (64 filters 3x3 kernel w stride equals 1)
	-> RELU Activation layer
	-> Flatten layer
  -> FC layer (1164 filters)
	-> RELU Activation layer
	-> FC layer (100 filters)
	-> RELU Activation layer
	-> FC layer (50 filters)
	-> RELU Activation layer
	-> Dropout layer (50%)
	-> FC layer (10 filters)
	-> RELU Activation layer
	-> FC layer (Output)

Training is done using keras fit generator. The model was trained for 9 epochs, with 22000 samples per epoch.
