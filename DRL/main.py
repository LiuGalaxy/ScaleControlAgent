import os
from osgeo import gdal
from DRL.myEnv import MinimalEnv
from DRL.myModel import CustomCNN_mask_feat
import numpy as np
from stable_baselines3 import A2C
import cv2

colorlist = {
    1: (68, 1, 84),
    2: (65, 67, 135),
    3: (42, 120, 142),
    4: (35, 168, 132),
    5: (122, 209, 81),
    6: (253, 231, 37),
}

mycolortable = gdal.ColorTable()
for i in colorlist:
    mycolortable.SetColorEntry(i, colorlist[i])


def pred_main(agent, img_path, save_path, PATCH_SIZE, THUMBSIZE):
    img_ds = gdal.Open(img_path)
    img = img_ds.ReadAsArray()

    result = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize), np.uint8)
    thumbnail = np.moveaxis(cv2.resize(np.moveaxis(img, 0, 2), (THUMBSIZE, THUMBSIZE)), 2, 0)
    for i in range(0, img.shape[1], int(PATCH_SIZE / 2)):
        for j in range(0, img.shape[2], int(PATCH_SIZE / 2)):
            iidx, yidx = i, j
            iidx = img.shape[1] - PATCH_SIZE if iidx + PATCH_SIZE >= img.shape[1] else iidx
            yidx = img.shape[2] - PATCH_SIZE if yidx + PATCH_SIZE >= img.shape[2] else yidx

            idx_mask = np.zeros((img.shape[1], img.shape[2]))
            idx_mask[iidx:iidx + PATCH_SIZE, yidx:yidx + PATCH_SIZE] = 1
            idx_mask = cv2.resize(idx_mask, (THUMBSIZE, THUMBSIZE),
                                  interpolation=cv2.INTER_NEAREST)

            obs = np.concatenate((thumbnail, idx_mask[None]))
            try:
                action, _states = agent.predict(obs, deterministic=True)
            except Exception as e:
                # Handle or log the error if needed
                print(f"Error predicting action")
                action = 0
            result[i:i + PATCH_SIZE, j:j + PATCH_SIZE] = action + 1
    driver = gdal.GetDriverByName("GTiff")
    saveimage = driver.Create(save_path, img.shape[2],
                              img.shape[1],
                              1, gdal.GDT_Byte, options=['COMPRESS=LZW'])
    band = saveimage.GetRasterBand(1)
    band.WriteArray(result[:, :])
    band.SetColorTable(mycolortable)
    band.SetNoDataValue(0)
    band.FlushCache()

    saveimage.SetGeoTransform(img_ds.GetGeoTransform())
    saveimage.SetProjection(img_ds.GetProjection())

    ds = band = None
    return


if __name__ == '__main__':
    n_actions = 6
    N_CHANNELS = 4
    THUMBSIZE = 1024
    PATCHSIZE = 512
    env = MinimalEnv(n_actions, N_CHANNELS, THUMBSIZE)
    policy_kwargs = dict(
        features_extractor_class=CustomCNN_mask_feat,
        features_extractor_kwargs=dict(features_dim=512, model_name='resnet18'),
    )
    agent = A2C('CnnPolicy', env, policy_kwargs=policy_kwargs)
    # agent.set_parameters(r"C:\Users\Galaxy\Downloads\model.zip",exact_match=True)
    agent = agent.load(r"C:\Users\Galaxy\Downloads\model.zip")
    pred_main(agent, "E:\OneDrive\DRL\ISPRS\SAN\examples\Example.tif",
              "E:\OneDrive\DRL\ISPRS\SAN\examples\JPEG\wa069_2021_1_1_1_scalemap.tif", PATCHSIZE, THUMBSIZE)
