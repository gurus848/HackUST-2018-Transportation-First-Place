import cv2
import keras
import os
import csv
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.utils import to_categorical

# x  - timeStamp
# y  - standID

def findsecondlast(text, pattern):
    return text.rfind(pattern, 0, text.rfind(pattern))

filepath = 'train.csv'
data = pd.read_csv(filepath)

b = data.loc[data['CALL_TYPE']=='B']
no_nan = b.loc[b['ORIGIN_STAND'].isnull() == False]
not_missing = no_nan.loc[no_nan['MISSING_DATA'] == False]
data = not_missing
tempData = data.drop(columns = ["TRIP_ID", "CALL_TYPE", "ORIGIN_CALL", "DAY_TYPE", "MISSING_DATA"])
data = tempData

npdata = data.as_matrix()
#print(npdata.shape)
tripTime = np.empty(shape=(806576,1))
tempdata = npdata[:, 3].tolist()
tempdata = [a.replace("[", "").replace("]","") for a in tempdata]
#print(len(tempdata))
for i, string in enumerate(tempdata):
    tripTime[i] = (string.count(',')+1)/2.0 * 15
    secondlastcomma = findsecondlast(string, ',') + 1
    lastcor = np.fromstring(string[secondlastcomma:], dtype = 'float32', sep = ',')
    tempdata[i] = lastcor

tempdata = np.array(tempdata)
npdata[:,3] = tempdata
print(npdata.shape)
print(tripTime.shape)
npdata = np.append(npdata, tripTime, axis = 1)
print(npdata.shape)
print(npdata)

#sorting based on driverID, timestamp
indices = np.lexsort((npdata[:, 2], npdata[:, 1]))
sortdata = npdata[indices]
print(sortdata)
npdata = sortdata

sortdata2 = npdata.copy()
for i in range(0,806575):
    sortdata2[i, 0] = sortdata2[i+1, 0]
for i in range(806575,0, -1):
    sortdata2[i, 3] = sortdata2[i-1, 3]
print(sortdata2)

cleandata = np.delete(sortdata2, 0, 0)
cleandata = np.delete(cleandata, 806574, 0)
firstDriver = cleandata[0, 1]
not_to_delete = np.ones(806574)
for i in range(0,806573):
    if cleandata[i, 1] != firstDriver:
        firstDriver = cleandata[i, 1]
        not_to_delete[i] = 0
        not_to_delete[i-1] = 0

print(806574 - np.sum(not_to_delete))
not_to_delete = not_to_delete.astype('bool')
cleandata = cleandata[not_to_delete]
print(cleandata.shape)
print(cleandata)

not_to_delete = np.ones(805698)
firstDriver = cleandata[0, 1]

for i in range(1, 805698):
    if cleandata[i, 1] != firstDriver:
        firstDriver = cleandata[i, 1]
        continue

    if cleandata[i, 2] - cleandata[i - 1, 2] - cleandata[
        i - 1, 4] > 8839:  # check if the timestamps differ by more than 30 minutes
        not_to_delete[i] = 0

print(np.sum(not_to_delete))
cleandata2 = cleandata.copy()
not_to_delete = not_to_delete.astype(np.bool)
cleandata2 = cleandata2[not_to_delete]
print(cleandata2.shape)
print(cleandata2)

ctr = 0
temparray = np.ones(shape = (cleandata2.shape[0], ))
for i in range(cleandata2.shape[0]):
    if cleandata2[i, 3].shape[0] == 0:
        temparray[i] = 0

print(cleandata2.shape)
temparray = temparray.astype('bool')
cleandata2 = cleandata2[temparray]
print(cleandata2.shape)
print(ctr)

import datetime

x_train = np.empty(shape = (504305,), dtype='object')
location = np.empty(shape = (504305,2))
for i in range(504305):
    loc = cleandata2[i, 3]
    lat = loc[0]
    lon = loc[1]
    location[i, 0] = lat
    location[i, 1] = lon
x_train = x_train[:, np.newaxis]
x_train = np.append(x_train, location, axis = 1)
# print(x_train.shape)
y_train = cleandata2[:, 0].astype('int32')
# print(y_train.shape)

timee = np.empty(shape = (504305,4), dtype='int32')

for i in range(cleandata2.shape[0]):
    ts = datetime.datetime.fromtimestamp(cleandata2[i][2])
    timee[i, 0] = ts.month
    timee[i, 1] = ts.day
    timee[i, 2] = ts.hour
    timee[i, 3] = ts.minute



x_train = np.delete(x_train, 0, 1)
x_train = np.append(x_train, timee, axis = 1)
# x_train = x_train.astype('float32')
print(x_train.shape)
print(x_train)


y_train = to_categorical(y_train)
y_train = np.delete(y_train, 0 ,1)
# print(y_train.shape)
# print(y_train)


over_x = cleandata2[:10000, :]
sorted_indices = np.lexsort((over_x[:, 2], over_x[:, 1]))
# print(sorted_indices.shape)
sorted_by_timestamp = over_x[sorted_indices]
x_train = x_train[:10000,:]
y_train = y_train[:10000,:]
x_train = x_train[sorted_indices]
y_train = y_train[sorted_indices]


gps_lat_min = 41.1405168841
gps_lat_max = 41.1835097223
gps_lon_min = -8.68917996027
gps_lon_max = -8.568195396819998
def get_image_coords_from_gps(x,y):
    return (int(((x - gps_lat_min)/(gps_lat_max - gps_lat_min))*255), int(((y - gps_lon_min)/(gps_lon_max - gps_lon_min)) * 255))


model = keras.models.load_model('final_weights.h5')

# background = np.empty(shape=(512,512,3))
# background.fill(255)  #white backgound image
background = cv2.imread("thing.png")
background = cv2.resize(background, (512,512))

def stand_to_img_coords(stand_id):

    mapping = np.array([[0.85197514, 0.65718871],
 [0.36454341, 0.81097582],
 [0.69797762, 0.19261985],
 [0.0755267,  0.55690361],
 [1.       ,  0.63193261],
 [0.93707358, 0.87942682],
 [0.45271031 ,0.39009965],
 [0.25760601, 1.        ],
 [0.08688987 ,0.68984292],
 [0.23710387, 0.68000566],
 [0.61677758, 0.62746103],
 [0.33965762, 0.48529905],
 [0.3854653 , 0.50181592],
 [0.21364738, 0.64561203],
 [0.18865056, 0.85385568],
 [0.81305055, 0.50740778],
 [0.55671359, 0.47025838],
 [0.18151913, 0.57508563],
 [0.3041447 , 0.69226595],
 [0.61431021, 0.51577845],
 [0.47378749, 0.49274791],
 [0.6466427 , 0.        ],
 [0.12790376, 0.63318195],
 [0.67484111, 0.82516364],
 [0.13451049, 0.59133058],
 [0.44295181, 0.90428076],
 [0.17480232, 0.6607358 ],
 [0.52812392, 0.86898018],
 [0.70947192, 0.41752747],
 [0.26206128, 0.50767246],
 [0.84750472, 0.20845334],
 [0.39879723, 0.5135846 ],
 [0.98319458, 0.73610644],
 [0.        , 0.60526539],
 [0.62603275, 0.32775374],
 [0.32889753, 0.3315    ],
 [0.47831632, 0.26146477],
 [0.46979927, 0.70202244],
 [0.24130949, 0.54866473],
 [0.31904983, 0.12541419],
 [0.52450061, 0.10907935],
 [0.74969892, 0.63972109],
 [0.18826737, 0.33304683],
 [0.21167507, 0.5239968 ],
 [0.18914119, 0.13542746],
 [0.88685184, 0.33594691],
 [0.77447056, 0.2866808 ],
 [0.71513048, 0.36475568],
 [0.90792348, 0.71613252],
 [0.68200712, 0.49822477],
 [0.4858243 , 0.56253448],
 [0.33606093, 0.62787326],
 [0.0160797 , 0.62129992],
 [0.40674208, 0.48605383],
 [0.59274726, 0.98217652],
 [0.5201288 , 0.80617457],
 [0.12099508, 0.64861949],
 [0.49019114, 0.69968951],
 [0.46952317, 0.39653994],
 [0.24671124, 0.65575769],
 [0.20452776, 0.73456838],
 [0.86890928, 0.35229664],
 [0.46190106, 0.65967291]])
    return (int(mapping[stand_id - 1][1] * 255), int(mapping[stand_id - 1][0] * 255))





# to_order = cleandata2.copy()
# dtype = [("stand", np.int), ("driver", np.int), ("timestamp", np.int), ("pos", np.ndarray), ("drive_time", np.int)]
# print(to_order.tolist())
# to_order = np.array(to_order.tolist(), dtype=dtype)
# to_order.astype(dtype)
# np.append([['stand', 'driver', 'timestamp', 'pos', 'drive_time']],to_order, axis=0)
# np.sort(to_order, order='timestamp')
# print(to_order)
# print(sorted_by_timestamp)
first_timestamp = sorted_by_timestamp[0][2]
start_index = 0
end_index = 0
for i in range(sorted_by_timestamp.shape[0]):
    #     print(sorted_by_timestamp[i][2])
    if sorted_by_timestamp[i][2] - first_timestamp < 1800:
        end_index = i
    else:
        #         print(start_index, end_index)
        #         print(first_timestamp, sorted_by_timestamp[i][2])
        if (start_index == end_index):
            start_index = end_index + 1
            first_timestamp = sorted_by_timestamp[i][2]
            continue
        #         print(start_index - end_index)
        stands = sorted_by_timestamp[start_index:end_index, 0]
        stands = stands.astype(np.int)
        my = np.bincount(stands)
        num_frames_recorded = np.nonzero(my)[0]
        counts = np.vstack((num_frames_recorded, my[num_frames_recorded])).T
        if counts.size != 0:
            # do the computation relating to putting the largest stands on the map, and show the taxis
            # print(counts)
            new_background = background.copy()
            maxe = (np.max(counts[:,1]))
            for j in range(counts.shape[0]):
                counts[j,1] /= maxe
                counts[j, 1] *= 20
            for j in range(counts.shape[0]):   #putting the largest things on the map
                cv2.circle(new_background, stand_to_img_coords(counts[j][0]), counts[j][1], (255,0,0), 5)

            for j in range(sorted_by_timestamp[start_index:end_index].shape[0]):
                cv2.circle(new_background, get_image_coords_from_gps(x_train[j][1], x_train[j][0]), 5, (0,255,0), 2)
                # print(x_train[i].reshape(1,6).shape)
                output = model.predict(x_train[j].reshape(1,6))
                argmaxe = np.argmax(output)
                # print(argmaxe)
                ideal = stand_to_img_coords(argmaxe)
                cv2.line(new_background, get_image_coords_from_gps(x_train[j][1], x_train[j][0]), ideal, (0,0,255), 3)

            cv2.imshow("demo", new_background)
            cv2.waitKey()
            print(start_index)

        start_index = end_index + 1
        first_timestamp = sorted_by_timestamp[i][2]
