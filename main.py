import cv2
import torchvision
import torch.optim
import numpy as np
import torchvision.transforms as transforms
from Dataset import datasets
import config as C
from pathlib import PosixPath
from FAG.FAG_TEST import FAG, FAG_old
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
                    = RFHN(anonymization_face_enhanced, protected_face, model='model_141_fmmrdb_best.pth') #model_161_rdb_best   model_141_best_2000
            # model_141_fmmrdb_Yeslowanonymization_Yeslowsecret_best model_141_DB_best
            #model_141_fmmrdb_Noresidual_Noconv2_best model_141_fmmrdb_Noresidual_best
            # cv2.imwrite(C.IMAGE_TEST_PATH_annotations + '%.d.' % i_batch + img_format, annotations_face_list[:, :, ::-1])
            # torchvision.utils.save_image(torch.cat((protected_face_img,protected_face_rev_img,anonymization_face_img,anonymized_face_img), 0), '/home/WRL/WRL/RFA-Net/image/test/1experiment/2-2/' + '%.d.' % i_batch + img_format)
            # img_cat = torch.cat((protected_face_img, protected_face_rev_img, resi_protected_rev_protected_img,
            #                            anonymization_face_img, anonymized_face_img, resi_anonymized_anonymization_img,
            #                            lost_r_img, random_z_img), 2)
            # img_cat_all = torch.cat((img_cat_all, img_cat), 0)
            # torchvision.utils.save_image(img_cat, '/home/WRL/WRL/RFA-Net/image/test/1experiment/1-10/' + '%.d.' % i_batch + img_format)
            torchvision.utils.save_image(anonymization_face_img, 'anonymization_face_img.' % i_batch + img_format)
            torchvision.utils.save_image(protected_face_img, 'protected_face_img.' % i_batch + img_format)
            torchvision.utils.save_image(anonymized_face_img, 'anonymized_face_img.' % i_batch + img_format)


test(img_format='png')
