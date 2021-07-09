
import cv2 #CV Library
import numpy as np #numpy
import tensorflow.keras  # Loading the keras models
from PIL import Image, ImageOps # pre-processing
import time # time
import os # for accessing files

from gtts import gTTS # Google Text-to-Speech
import playsound # to play audio

######### to play audio
def playaudio(t): # t- msg object 
	language='en'
	m=gTTS(text=t,lang=language,slow=False)
	m.save('out.mp3')
	time.sleep(1)
	playsound.playsound('out.mp3')
	os.remove('out.mp3')


# load the DL Model
model=tensorflow.keras.models.load_model('model.h5')

# define the deep learning model function
def sign_language_detector(a):

	# empty data array variable
	data=np.ndarray(shape=(1,224,224,3),dtype=np.float32)

	#open the image
	image=Image.open(a) # test.jpg

	# resize the image
	size=(224,224)
	image=ImageOps.fit(image,size,Image.ANTIALIAS)

	# convert this into numpy array
	image_array=np.asarray(image)

	# Normalise the Image - (0 to 255)
	normalise_image_array=(image_array.astype(np.float32)/127.0)-1

	# loading the image into the array
	data[0]=normalise_image_array

	# pass this data to model
	prediction=model.predict(data)
	print(prediction) # [[0.5,0.5,0.7,0.3]]

	# Decision Logic
	prediction=list(prediction[0])
	max_prediction=max(prediction)
	index_max=prediction.index(max_prediction)

	if(index_max==0):
		print('Empty Gesture')
	elif(index_max==1):
		print('Like Gesture Found')
		playaudio('I like what you are talking about')
	elif(index_max==2):
		print('Dislike Gesture Found')
		playaudio('I dislike about your words')
	elif(index_max==3):
		print('Question Gesture Found')
		playaudio('I have a question to ask')


# Access Camera 
video=cv2.VideoCapture(1) # device-id=0 (single)
video.set(cv2.CAP_PROP_FRAME_WIDTH,240)
video.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

# Acquiring the Image
while True:
	res,frame=video.read()
	if res==1:
		cv2.imwrite('test.jpg',frame) # writing image

		# pass this image to DL Model
		sign_language_detector('test.jpg')
		cv2.imshow("Capturing",frame) # showing frame to user

		key=cv2.waitKey(1)
		if key==ord('q'):
			break

video.release()
cv2.destroyAllWindows()




