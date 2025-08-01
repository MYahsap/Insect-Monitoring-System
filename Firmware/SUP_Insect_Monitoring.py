# by MYahsap for SUP
##################################################################
import sensor, image, time, ml, math, uos, gc, machine
from machine import Pin
from machine import LED
##################################################################
#Variables
min_confidence = 0.5
sleep_time=3000 # 1 hour = 3600000 ms
tile_w = 160
tile_h = 160
window_w = 480
window_h = 640
offset_w = 600
offset_h = 400
sensor_framsize = sensor.UXGA
datetimetuple = (2025, 6, 1, 1, 0, 0, 0, 0)
csv_filename = "detections.csv"
##################################################################
#Status LED
statusLED = LED("LED_BLUE")
statusLED.on()

#Setup RTC
# if machine.reset_cause() != machine.DEEPSLEEP:
#     machine.RTC.datetime(datetimetuple)
timestamp = time.localtime()
timestamp_str = "%04d-%02d-%02d %02d:%02d:%02d" % timestamp[0:6]

#Prepare Sensor
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor_framsize)
sensor.set_windowing(offset_w, offset_h, window_w, window_h)
sensor.skip_frames(time=2000)

#IR Light
IR_LED = Pin("P1", Pin.OUT)
IR_LED.low()

#TFLite Import
net = ml.Model("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
labels = [line.rstrip('\n') for line in open("labels.txt")]

#Log File
try:
    is_empty = uos.stat(csv_filename)[6] == 0  # size == 0
except OSError:
    # File doesn't exist
    is_empty = True

csv_file = open(csv_filename, "a")
if is_empty:
    csv_file.write("timestamp,label,count\n")
    print("new csv doc")

#FOMO Post Process
threshold_list = [(math.ceil(min_confidence * 255), 255)]
def fomo_post_process(model, inputs, outputs):
    ob, oh, ow, oc = model.output_shape[0]

    x_scale = inputs[0].roi[2] / ow
    y_scale = inputs[0].roi[3] / oh
    scale = min(x_scale, y_scale)
    x_offset = ((inputs[0].roi[2] - (ow * scale)) / 2) + inputs[0].roi[0]
    y_offset = ((inputs[0].roi[3] - (ow * scale)) / 2) + inputs[0].roi[1]

    l = [[] for i in range(oc)]

    for i in range(oc):
        img = image.Image(outputs[0][0, :, :, i] * 255)
        blobs = img.find_blobs(
            threshold_list, x_stride=1, y_stride=1, area_threshold=1, pixels_threshold=1
        )
        for b in blobs:
            rect = b.rect()
            x, y, w, h = rect
            score = (
                img.get_statistics(thresholds=threshold_list, roi=rect).l_mean() / 255.0
            )
            x = int((x * scale) + x_offset)
            y = int((y * scale) + y_offset)
            w = int(w * scale)
            h = int(h * scale)
            l[i].append((x, y, w, h, score))
    return l

colors = [
    (255,   0,   0),   # Red
    (  0, 255,   0),   # Green
    (255, 255,   0),   # Yellow
    (  0,   0, 255),   # Blue
    (255,   0, 255),   # Magenta
    (  0, 255, 255),   # Cyan
    (255, 255, 255),   # White
    (128,   0,   0),   # Dark Red
    (  0, 128,   0),   # Dark Green
    (  0,   0, 128),   # Dark Blue
    (128, 128,   0),   # Olive
    (128,   0, 128),   # Purple
    (  0, 128, 128),   # Teal
    (192, 192, 192),   # Light Gray
    (128, 128, 128),   # Gray
]

##################################################################
# Image Capture
##################################################################

#turn on IR light and take image
IR_LED.high()
time.sleep(1)  # Wait for 2 seconds
img = sensor.snapshot()
IR_LED.low()

##################################################################
# Image Processing
##################################################################

#tile prep

tiles = []
for y in range(0, window_h, tile_h):
    for x in range(0, window_w, tile_w):
        tiles.append((x, y, tile_w, tile_h))

# Store all detections (indexed by label)
global_detections = [[] for _ in labels]

# Run inference per tile
for tile in tiles:
    x_off, y_off, w, h = tile
    sub_img = img.copy(roi=(x_off, y_off, w, h))

    detections = net.predict([sub_img], callback=fomo_post_process)

    for i, det_list in enumerate(detections):
        if i == 0: continue  # skip background
        for x, y, w, h, score in det_list:
            global_x = x + x_off
            global_y = y + y_off
            global_detections[i].append((global_x, global_y, w, h, score))

    #free memory
    sub_img = None
    gc.collect()

# Draw and log results
for i, detection_list in enumerate(global_detections):
    if i == 0: continue
    if len(detection_list) == 0: continue

    label = labels[i]
    count = len(detection_list)
    csv_file.write("%s,%s,%d\n" % (timestamp_str, label, count))

    print("********** %s **********" % label)
    print("Count:", count)

    for x, y, w, h, score in detection_list:
        center_x = math.floor(x + (w / 2))
        center_y = math.floor(y + (h / 2))
        print(f"x {center_x}\t y {center_y}\t w {w}\t h {h}\t score {score}")
        img.draw_rectangle(x, y, w, h, color=colors[i], thickness=2)
        label_text = "%s: %.2f" % (label, score)
        img.draw_string(x, y - 10, label_text, mono_space=False, color=colors[i])

#save image
filename = "%04d-%02d-%02d_%02d-%02d-%02d.jpg" % timestamp[0:6]
print("Saving image:", filename)
img.save(filename)

##################################################################
#Go Back to sleep
##################################################################

statusLED.off()
gc.collect()
csv_file.close()
print("Going to sleep for 1 hour...")
machine.deepsleep(sleep_time)
