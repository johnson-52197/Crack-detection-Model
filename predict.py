from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
 
# load and prepare the image
def load_image(filename):
		# load the image
	img = load_img(filename, target_size=(128, 128))
	# convert to array
	img = img_to_array(img)

	img = img.reshape(1, 128, 128, 3)
	
	return img

# load an image and predict the class
def run_example():
	# load the image
	img = load_image('1.jpg') # enter the image file name to predict
	# load model
	model = tf.keras.models.load_model('model.h5') #loading the saved model
	# predict the class
	result = model.predict(img)

	if (result[0]<0):
		print("CRACK DETECTED")
	else:
		print("NO CRACK DETECTED")

	print(result)
 
# entry point, run the example
run_example()
