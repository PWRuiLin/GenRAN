import cv2
import torchvision
import torch.optim
import numpy as np
import torchvision.transforms as transforms
from Dataset import datasets
import config as C
from pathlib import PosixPath
from IFAG.IFAG_TEST import IFAG
from RFHN.RFHN_TEST import RFHN
from evaluation_metrics.calculate_PSNR_SSIM import calculate_PSNR_SSIM
from evaluation_metrics.calculate_APD_RMSE import calculate_APD_RMSE_MAE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor()  # 转换为张量
])
def test(img_format='png'):
    with torch.no_grad():
        T_recovery = 0
        # img_cat_all= torch.empty(1, 3, 4096, 512).to(device)
        for i_batch, protected_face in enumerate(datasets.testloader):
            # if i_batch == 100:
            protected_face = protected_face.to(device)
            # torchvision.utils.save_image(protected_face, C.IMAGE_TEST_PATH_transit_place + 'transit_place.png')
            anonymization_face, anonymization_face_enhanced = IFAG(protected_face)
            # anonymization_face_list, annotations_face_list = FAG(C.IMAGE_TEST_PATH_transit_place)
            # anonymization_face = np.copy(anonymization_face_list[0])
            # anonymization_face = transform(anonymization_face).unsqueeze(0)  # 添加批次维度

            anonymization_face_img, protected_face_img, anonymized_face_img, protected_face_rev_img, anonymization_face_rev_img, \
                lost_r_img, random_z_img, resi_anonymized_anonymization_img, resi_protected_rev_protected_img, T_recover \
                    = RFHN(anonymization_face_enhanced, protected_face, model='model_141_fmmrdb_best.pth') #model_141_fmmrdb_best

            torchvision.utils.save_image(anonymization_face,
                                         "/home/WRL/WRL/Gen-RAN/image/test/virtual/" + '%.d.' % i_batch + img_format)
            torchvision.utils.save_image(anonymization_face_img, C.IMAGE_TEST_PATH_anonymization + '%.d.' % i_batch + img_format)
            torchvision.utils.save_image(protected_face_img, C.IMAGE_TEST_PATH_protected + '%.d.' % i_batch + img_format)
            torchvision.utils.save_image(anonymized_face_img, C.IMAGE_TEST_PATH_anonymized + '%.d.' % i_batch + img_format)
            torchvision.utils.save_image(protected_face_rev_img, C.IMAGE_TEST_PATH_protected_rev + '%.d.' % i_batch + img_format)
            torchvision.utils.save_image(resi_anonymized_anonymization_img, C.IMAGE_TEST_PATH_resi_anonymized_anonymization + '%.d.' % i_batch + img_format)
            torchvision.utils.save_image(resi_protected_rev_protected_img, C.IMAGE_TEST_PATH_resi_protected_rev_protected + '%.d.' % i_batch + img_format)
            torchvision.utils.save_image(lost_r_img, C.IMAGE_TEST_PATH_output_r + '%.d.' % i_batch + img_format)
            torchvision.utils.save_image(random_z_img, C.IMAGE_TEST_PATH_backward_z + '%.d.' % i_batch + img_format)
            T_recovery = T_recovery + T_recover
            print(f'OK      i_batch = {i_batch} | T_recovery = {T_recovery}')
        # img_cat_all = img_cat_all[1:]
        # torchvision.utils.save_image(img_cat_all, '/home/WRL/WRL/RFA-Net/image/test/1experiment/1-10/' + 'all.' + img_format, nrow=10)
        PSNR_all = []; SSIM_all = []
        PSNR_all, SSIM_all = calculate_PSNR_SSIM(test_Y=True, secret=True, cover=False, img=img_format)
        print(f' secret | PSNR_AVE={sum(PSNR_all) / len(PSNR_all)} | SSIM_AVE={sum(SSIM_all) / len(SSIM_all)} ')
        PSNR_all = []; SSIM_all = []
        PSNR_all, SSIM_all = calculate_PSNR_SSIM(test_Y=True, secret=False, cover=True, img=img_format)
        print(f' cover | PSNR_AVE={sum(PSNR_all) / len(PSNR_all)} | SSIM_AVE={sum(SSIM_all) / len(SSIM_all)} ')

        MAE_all = []; RMSE_all = []; MAE = []
        MAE_all, RMSE_all, MAE = calculate_APD_RMSE_MAE(test_Y=True, secret=True, cover=False, img=img_format)
        print(f' secret | MAE_AVE={sum(MAE_all) / len(MAE_all)} | RMSE_AVE={sum(RMSE_all) / len(RMSE_all)} ')
        MAE_all = []; RMSE_all = []; MAE = []
        MAE_all, RMSE_all, MAE = calculate_APD_RMSE_MAE(test_Y=True, secret=False, cover=True, img=img_format)
        print(f' cover | MAE_AVE={sum(MAE_all) / len(MAE_all)} | RMSE_AVE={sum(RMSE_all) / len(RMSE_all)} ')
test(img_format='png')