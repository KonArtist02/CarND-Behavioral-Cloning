import cv2

print('Loading config file ...')

resize_x = 80
resize_y = 40
cropped_y_top = int(resize_y*0.43)
cropped_y_bottom = int(resize_y*0.84)
img_size_x = resize_x
img_size_y = cropped_y_bottom-cropped_y_top

print ('resize_x: ', resize_x)
print ('resize_y: ', resize_y)
print ('New image size: width', img_size_x, 'height', img_size_y)


def process_image(img,resize_x,resize_y):
	img = cv2.resize(img,dsize=(resize_x,resize_y))
	img = img[int(resize_y*0.43):int(resize_y*0.84),:]
	return img