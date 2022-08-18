import pyzed.sl as sl
import time
import cv2
from objdet.accelerated import YoloAccInference


params = {
            'conf_threshold': 0.3,
            'img_size': 640,
            'iou_threshold': 0.3,
            'weights_dir': '/home/trabotyx/trabotyx_dvc_plant_detection/models/carrot-stem-weed-early.pt',
            'debug': True,
            'verbose': 0,
            'sahi_enabled': True,
            'TTA_enabled' : False,
            'batch_size' : 2,
            'accelerated': False,
            'regen': False,
            }

detection_instance = YoloAccInference(params)

zed_list = []
left_list = []
depth_list = []
name_list = []
key = ''
number_of_cameras = 3



init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD720
init.camera_fps = 30 
cameras = sl.Camera.get_device_list()
runtime = sl.RuntimeParameters()


for i, cam in enumerate(cameras):
	init.set_from_serial_number(cam.serial_number)
	name_list.append("ZED {}".format(cam.serial_number))
	print("Opening {}".format(name_list[i]))
	zed_list.append(sl.Camera())
	left_list.append(sl.Mat())
	depth_list.append(sl.Mat())
	zed_list[i].open(init)



while key != 113:
	for index in range(number_of_cameras):
		err = zed_list[index].grab(runtime)
		if err == sl.ERROR_CODE.SUCCESS:
			zed_list[index].retrieve_image(left_list[index], sl.VIEW.LEFT)
			zed_list[index].retrieve_measure(depth_list[index], sl.MEASURE.DEPTH)

			time.sleep(0.001) #1ms
		inference_frame = cv2.cvtColor(left_list[index].get_data(), cv2.COLOR_BGRA2BGR)
		det_objs = detection_instance.detect(inference_frame)
		debug_frame = YoloAccInference.visualize(inference_frame.copy(), det_objs, 
				color_palette = [(0,255,0),(255,0,0),(0,0,255)], line_thickness = 1, 
				extra_label ='', category_names=['carrot', 'intersection', 'weed'], 
				verbose=3)
		cv2.imshow(name_list[index], debug_frame)
	key = cv2.waitKey(10)
cv2.destroyAllWindows()
for index in range(number_of_cameras):
	zed_list[index].close()


print("Finish")




