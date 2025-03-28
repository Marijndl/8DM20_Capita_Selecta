import numpy as np


def row_averages_excluding_diagonal(matrix):
    matrix = np.array(matrix)  # Convert to NumPy array if not already
    n = matrix.shape[0]

    averages = []
    for i in range(n):
        row = np.delete(matrix[i], i)  # Exclude the diagonal element
        averages.append(np.mean(row))  # Compute the mean

    return averages


# Your matrix as a NumPy array
matrix = np.array([
    [0.00738361, 0.63281835, 0.5004453, 0.51176517, 0.44983116, 0.55277611,
     0.68174911, 0.5299787, 0.56913767, 0.63962239, 0.75028981, 0.53342145,
     0.42954957, 0.77779738, 0.47782524],
    [0.55622075, 0.96407213, 0.48722473, 0.62257422, 0.3828007, 0.5216521,
     0.69680153, 0.61088901, 0.7058952, 0.78999262, 0.70088814, 0.58681778,
     0.0015972, 0.6507406, 0.6233032],
    [0.47634113, 0.50662936, 0.8339262, 0.51234959, 0.04639095, 0.8233286,
     0.42185236, 0.61166504, 0.6179237, 0.44626463, 0.36447678, 0.75498533,
     0.81830224, 0.48876109, 0.39650742],
    [0.5076626, 0.64200911, 0.5802421, 0.93803569, 0.48187185, 0.58793073,
     0.59722361, 0.6261693, 0.69849153, 0.68329814, 0.57729805, 0.62194533,
     0.50980084, 0.56556641, 0.57134448],
    [0.31076988, 0.62400679, 0.02841354, 0.35671054, 0.84096401, 0.09989468,
     0.52103516, 0.61587701, 0.41520551, 0.12650878, 0.56507945, 0.02494171,
     0.12509331, 0.39183336, 0.63766914],
    [0.56156165, 0.5159201, 0.81019501, 0.54132669, 0.36472593, 0.90472599,
     0.47315003, 0.67643455, 0.73740728, 0.52007988, 0.53181093, 0.82012836,
     0.82268963, 0.50119821, 0.38168096],
    [0.68624462, 0.74027369, 0.49949976, 0.6001214, 0.45258446, 0.51598864,
     0.87395919, 0.47482849, 0.63644221, 0.66557676, 0.61193638, 0.56341381,
     0.43203208, 0.74282784, 0.53566718],
    [0.55932869, 0.53374801, 0.39228034, 0.6463473, 0.57361152, 0.70991433,
     0.43547501, 0.88131756, 0.84963309, 0.60371597, 0.5032156, 0.75275029,
     0.64061543, 0.3912192, 0.57281542],
    [0.57350803, 0.70209068, 0.7144531, 0.69840982, 0.4991398, 0.74919687,
     0.65075946, 0.83939709, 0.03380765, 0.72243901, 0.61530971, 0.78181109,
     0.64902394, 0.64372895, 0.52956189],
    [0.55013644, 0.79772972, 0.50931045, 0.66093966, 0.56713524, 0.57216179,
     0.73093624, 0.63758347, 0.70678551, 0.93221535, 0.69547118, 0.60022392,
     0.48488463, 0.65328879, 0.55175424],
    [0.65491442, 0.66018511, 0.21947355, 0.57569582, 0.45312975, 0.14086494,
     0.46834923, 0.55113944, 0.59535906, 0., 0.95316305, 0.51412833,
     0.44010873, 0.29516497, 0.56267679],
    [0.50901171, 0.55827363, 0.74272007, 0.60584843, 0.28462772, 0.78325596,
     0.46824585, 0.6910716, 0.79497708, 0.60688385, 0.4654242, 0.9596832,
     0.77097805, 0.48996993, 0.46307088],
    [0.46186727, 0.41011066, 0.82600213, 0.45491325, 0.394088, 0.82277739,
     0.35307411, 0.61020505, 0.66837975, 0.51363267, 0.47795201, 0.79191034,
     0., 0.46008696, 0.38403781],
    [0.79145235, 0.72896886, 0.50453379, 0.57759676, 0.33542522, 0.51739161,
     0.73333135, 0.43694754, 0.61386577, 0.67247005, 0.67912894, 0.57157908,
     0.4705751, 0.85242421, 0.49415127],
    [0.46494336, 0.59152586, 0.39056982, 0.57153528, 0.59582581, 0.36363951,
     0.54223645, 0.5835749, 0.55740906, 0.54239181, 0.54327869, 0.44505376,
     0.34604675, 0.4867001, 0.]
])

# Compute row averages excluding diagonal
averages = row_averages_excluding_diagonal(matrix)

# Print results
print(averages)

# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
# hd_distance = [[0.00872595, 0.62530656, 0.90739397, 0.64773485, 0.40036871,
#         0.95029409, 0.64673132, 0.67378819, 0.73130203, 0.7438177 ,
#         0.73863505, 0.84969226, 0.88972569, 0.8248049 , 0.43962944],
#        [0.58420801, 0.96407213, 0.94780924, 0.82787169, 0.35573422,
#         0.96078701, 0.69246964, 0.82407294, 0.95338785, 0.88477655,
#         0.7241716 , 0.96108704, 0.00451177, 0.7267702 , 0.57524257],
#        [0.33426041, 0.34277381, 0.8339262 , 0.41826237, 0.03306093,
#         0.79384724, 0.28298747, 0.48922097, 0.49458355, 0.36506189,
#         0.25482424, 0.69237552, 0.89872772, 0.34867741, 0.27586036],
#        [0.42217919, 0.51443771, 0.79547272, 0.93803569, 0.36558975,
#         0.76934297, 0.47631407, 0.63342046, 0.70726834, 0.596924  ,
#         0.47414144, 0.73739095, 0.78617854, 0.49300255, 0.43126806],
#        [0.35457485, 0.67539506, 0.06765594, 0.52308749, 0.84096401,
#         0.21501215, 0.56017082, 0.91691938, 0.61894975, 0.15460565,
#         0.63347864, 0.04552347, 0.32010292, 0.47742188, 0.63291597],
#        [0.39845363, 0.35709177, 0.84170708, 0.44970415, 0.24571988,
#         0.90472599, 0.32363375, 0.55374708, 0.60754133, 0.43067554,
#         0.3723323 , 0.78765268, 0.94151444, 0.36580005, 0.26696695],
#        [0.72552897, 0.75738068, 0.98077745, 0.80843873, 0.42302998,
#         0.95907291, 0.87395919, 0.65433326, 0.86639993, 0.78191731,
#         0.636373  , 0.93312146, 0.97258997, 0.83538535, 0.51098229],
#        [0.46094684, 0.42404857, 0.53005622, 0.6390319 , 0.43183254,
#         0.91597258, 0.34437885, 0.88131756, 0.8504512 , 0.5222962 ,
#         0.40963668, 0.88079343, 0.97279951, 0.33773297, 0.42907416],
#        [0.47227487, 0.55746097, 0.96421291, 0.68984916, 0.37552536,
#         0.9655192 , 0.51426462, 0.8385904 , 0.03381306, 0.62752811,
#         0.50051415, 0.91379053, 0.98429756, 0.55527091, 0.3964632 ],
#        [0.51922848, 0.72054005, 0.84831707, 0.77275654, 0.47991847,
#         0.90524175, 0.6565449 , 0.75533054, 0.83824452, 0.93221535,
#         0.64680865, 0.85088032, 0.9187306 , 0.65163166, 0.46405502],
#        [0.66763945, 0.65188358, 0.41170149, 0.74156652, 0.40897148,
#         0.25036307, 0.45102976, 0.71999498, 0.77254672, 0.        ,
#         0.95630157, 0.80559251, 0.93377013, 0.31830775, 0.51830751],
#        [0.37760262, 0.40180253, 0.84309278, 0.52383677, 0.1963218 ,
#         0.85347981, 0.33572822, 0.6033597 , 0.69464849, 0.49300061,
#         0.34202097, 0.9596832 , 0.97142449, 0.37779928, 0.32408963],
#        [0.30629893, 0.2775389 , 0.75815615, 0.35498541, 0.24875533,
#         0.73057448, 0.22707179, 0.45606589, 0.50525287, 0.38923204,
#         0.31843789, 0.6691382 , 0.        , 0.3120933 , 0.24483444],
#        [0.74877779, 0.69231086, 0.84335969, 0.69678827, 0.28443408,
#         0.82144241, 0.66018533, 0.55212159, 0.74570224, 0.72254447,
#         0.63486284, 0.8322686 , 0.89496443, 0.85242421, 0.4407068 ],
#        [0.53498999, 0.64545241, 0.86270961, 0.84646974, 0.6003343 ,
#         0.7587227 , 0.58768391, 0.87756059, 0.83929076, 0.66878108,
#         0.61414511, 0.8212351 , 0.88060294, 0.59830206, 0.        ]]
# DATA_PATH = r'D:\capita_selecta\DevelopmentData\DevelopmentData'
#
#
# patient_list = [patient for patient in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, patient))]
# # atlas_patients = patient_list[:5]
# atlas_patients = ["p109", "p120", "p125"]
# # register_patients = [patient for patient in patient_list if patient not in atlas_patients]
# register_patients = ["p137", "p141", "p143", "p144", "p147"]
#
# patient_list = [patient for patient in patient_list if patient not in register_patients]
#
# plt.figure(figsize=(10, 8))
# sns.heatmap(hd_distance, annot=True, xticklabels=patient_list, yticklabels=patient_list,
#             cmap="viridis")
# plt.xlabel("Registered Patients")
# plt.ylabel("Atlas Patients")
# plt.title("Precision Confusion Matrix")
# plt.show()
