import cv2

print('Loading config file ...')

resize_x = 160
resize_y = 80
cropped_y_top = int(resize_y*0.43)
cropped_y_bottom = int(resize_y*0.84)
img_size_x = resize_x
img_size_y = cropped_y_bottom-cropped_y_top

augment_factor = 1 # of steering anges for off-center images
angle_thres = 0.05 # threshold for dividing data set into three bags

batch_size = 200

print ('resize_x: ', resize_x)
print ('resize_y: ', resize_y)
print ('New image size: width', img_size_x, 'height', img_size_y)


