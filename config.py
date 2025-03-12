
# Dataset
Dataset_TRAIN_mode = 'CelebA-HQ'    # CelebA / CelebA-HQ
Dataset_VALID_mode = 'CelebA-HQ'    # CelebA / CelebA-HQ
Dataset_TEST_mode = 'experiment1_10'     # CelebA-HQ jpg/ CelebA jpg/ LFW jpg/ FFHQ png/ AGE_ADULTS / experiment1_10 experiment2_2 png/ experiment2 / experiment3 png/ experiment4/experiment5
Dataset_PATH = '/home/WRL/WRL/Gen-RAN/Dataset/'
TRAIN_PATH_CelebAHQ = Dataset_PATH + 'CelebA-HQ/train/'
TRAIN_PATH_CelebA = Dataset_PATH + 'CelebA/train/'
VALID_PATH_CelebAHQ = Dataset_PATH + 'CelebA-HQ/valid/'
VALID_PATH_CelebA = Dataset_PATH + 'CelebA/valid/'
TEST_PATH_DIV2K = Dataset_PATH + 'DIV2K/test/'
TEST_PATH_CNM = Dataset_PATH + 'CNM/'
TEST_PATH_CelebAHQ = Dataset_PATH + 'CelebA-HQ/test/'
TEST_PATH_CelebA = Dataset_PATH + 'CelebA/test/'
TEST_PATH_FFHQ = Dataset_PATH + 'FFHQ/test/'
TEST_PATH_LFW = Dataset_PATH + 'LFW/test/'
TEST_PATH_AGE_ADULTS = Dataset_PATH + 'AGE_ADULTS/test/'
TEST_PATH_experiment1_10 = Dataset_PATH + 'experiments/experiment1/10/'
TEST_PATH_experiment1_1 = Dataset_PATH + 'experiments/experiment1/1/'
TEST_PATH_experiment2_2 = Dataset_PATH + 'experiments/experiment2/2/'
TEST_PATH_experiment2 = Dataset_PATH + 'experiments/experiment2/'
TEST_PATH_experiment3 = Dataset_PATH + 'experiments/experiment3/'
TEST_PATH_experiment4 = Dataset_PATH + 'experiments/experiment4/'
TEST_PATH_experiment5 = '/home/WRL/WRL/WRL/Privacy/PRO-Face/PRO-Face_Dataset/CelebA_align_crop_224/test/'
TEST_PATH_experiment6 = Dataset_PATH + 'experiments/experiment6/'

# if Dataset_TEST_mode == 'CelebA-HQ':
#     protected_face_paths = TEST_PATH_CelebAHQ
# if Dataset_TEST_mode == 'CelebA':
#     protected_face_paths = TEST_PATH_CelebA
# if Dataset_TEST_mode == 'LFW':
#     protected_face_paths = TEST_PATH_LFW
# if Dataset_TEST_mode == 'FFHQ':
#     protected_face_paths = TEST_PATH_FFHQ
# if Dataset_TEST_mode == 'AGE_ADULTS':
#     protected_face_paths = TEST_PATH_AGE_ADULTS
# if Dataset_TEST_mode == 'experiment1_8':
#     protected_face_paths = TEST_PATH_experiment1_8
# if Dataset_TEST_mode == 'experiment1_2':
#     protected_face_paths = TEST_PATH_experiment1_2
# if Dataset_TEST_mode == 'experiment2':
#     protected_face_paths = TEST_PATH_experiment2
# if Dataset_TEST_mode == 'experiment3':
#     protected_face_paths = TEST_PATH_experiment3
batch_size_train = 16         #16
cropsize_train = 128
batchsize_valid = 1
batchsize_test = 1   #1    2
resize_test = 512
# resize_test = 256
cropsize_test = 178 #512
cropsize_valid_CelebAHQ = 128
cropsize_valid_CelebA = 128
cropsize_valid_LFW = 128

IMAGE_TEST_PATH = '/home/WRL/WRL/Gen-RAN/image/test/'
IMAGE_TEST_PATH_annotations =  IMAGE_TEST_PATH + 'annotations_face/'
IMAGE_TEST_PATH_anonymization = IMAGE_TEST_PATH + 'anonymization_face/'
IMAGE_TEST_PATH_protected = IMAGE_TEST_PATH + 'protected_face/'
IMAGE_TEST_PATH_anonymized = IMAGE_TEST_PATH + 'anonymized_face/'
IMAGE_TEST_PATH_protected_rev = IMAGE_TEST_PATH + 'protected_face_rev/'
IMAGE_TEST_PATH_resi_anonymized_anonymization = IMAGE_TEST_PATH + 'resi_anonymized_anonymization/'
IMAGE_TEST_PATH_resi_protected_rev_protected = IMAGE_TEST_PATH + 'resi_protected_rev_protected/'
IMAGE_TEST_PATH_output_r = IMAGE_TEST_PATH + 'output_z_face/'
IMAGE_TEST_PATH_backward_z = IMAGE_TEST_PATH + 'backward_z_face/'
IMAGE_TEST_PATH_transit_place = IMAGE_TEST_PATH + 'transit_place/'


# IMAGE_TEST_PATH_annotations5 =  IMAGE_TEST_PATH + 'test5/annotations_face/'
IMAGE_TEST_PATH_anonymization5 = IMAGE_TEST_PATH + 'test5/anonymization_face/'
IMAGE_TEST_PATH_protected5 = IMAGE_TEST_PATH + 'test5/protected_face/'
IMAGE_TEST_PATH_anonymized5 = IMAGE_TEST_PATH + 'test5/anonymized_face/'
IMAGE_TEST_PATH_protected_rev5 = IMAGE_TEST_PATH + 'test5/protected_face_rev/'
IMAGE_TEST_PATH_resi_anonymized_anonymization5 = IMAGE_TEST_PATH + 'test5/resi_anonymized_anonymization/'
IMAGE_TEST_PATH_resi_protected_rev_protected5 = IMAGE_TEST_PATH + 'test5/resi_protected_rev_protected/'
IMAGE_TEST_PATH_output_r5 = IMAGE_TEST_PATH + 'test5/output_z_face/'
IMAGE_TEST_PATH_backward_z5 = IMAGE_TEST_PATH + 'test5/backward_z_face/'


if Dataset_TEST_mode == 'CelebA-HQ':
    format_test = 'jpg'
if Dataset_TEST_mode == 'CelebA':
    format_test = 'jpg'
if Dataset_TEST_mode == 'LFW':
    format_test = 'jpg'
if Dataset_TEST_mode == 'FFHQ':
    format_test = 'png'
if Dataset_TEST_mode == 'AGE_ADULTS':
    format_test = 'png'
# if Dataset_TEST_mode == 'experiment1_8':
#     format_test = 'jpg'
# if Dataset_TEST_mode == 'experiment1_2':
#     format_test = 'jpg'
# if Dataset_TEST_mode == 'experiment2':
#     format_test = 'jpg'
# if Dataset_TEST_mode == 'experiment3':
#     format_test = 'png'

