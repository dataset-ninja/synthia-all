import glob
import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import file_exists, get_file_name, get_file_name_with_ext
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    train_data_path = "/home/alex/DATASETS/TODO/SYNTHIA-AL/train"
    test_data_path = "/home/alex/DATASETS/TODO/SYNTHIA-AL/test"

    group_tag_name = "im id"
    batch_size = 30
    rgb_folder = "/RGB/"
    depth_folder = "/Depth/"
    masks_folder = "/SemSeg/"
    bboxes_folder = "/labels_kitti/"
    images_ext = ".png"
    bboxes_ext = ".txt"

    # ds_name_to_split = {"train": train_data_path, "test": test_data_path}
    ds_name_to_split = {"test": test_data_path}

    def create_ann(image_path, unique_id):
        labels = []
        tags = []

        group_tag = sly.Tag(group_tag_meta, value=unique_id)
        tags.append(group_tag)

        subfolder_value = image_path.split("/")[-4]
        subfolder = sly.Tag(subfolder_meta, value=subfolder_value)
        tags.append(subfolder)

        img_height = 480
        img_wight = 640

        mask_path = image_path.replace(rgb_folder, masks_folder)
        if file_exists(mask_path):
            mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
            unique_pixels = np.unique(mask_np)

            for pixel in unique_pixels:
                obj_class = pixel_to_class.get(pixel)
                if obj_class is not None:
                    mask = mask_np == pixel
                    ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
                    for i in range(1, ret):
                        obj_mask = curr_mask == i
                        curr_bitmap = sly.Bitmap(obj_mask)
                        if curr_bitmap.area > 30:
                            curr_label = sly.Label(curr_bitmap, obj_class)
                            labels.append(curr_label)

        bbox_path = image_path.replace(rgb_folder, bboxes_folder).replace(images_ext, bboxes_ext)
        if file_exists(bbox_path):
            with open(bbox_path) as f:
                content = f.read().split("\n")
                for curr_data in content:
                    if len(curr_data) > 0:
                        curr_bboxes_data = curr_data.split(" ")
                        class_name = curr_bboxes_data[0].lower()
                        if class_name == "trafficlight":
                            class_name = "traffic light"
                        elif class_name == "trafficsign":
                            class_name = "traffic sign"

                        obj_class = meta.get_obj_class(class_name)

                        angle = sly.Tag(angle_meta, value=float(curr_bboxes_data[3]))

                        left = float(curr_bboxes_data[4])
                        right = float(curr_bboxes_data[6])
                        top = float(curr_bboxes_data[5])
                        bottom = float(curr_bboxes_data[7])
                        rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                        label = sly.Label(rectangle, obj_class, tags=[angle])
                        labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    subfolder_meta = sly.TagMeta("subfolder", sly.TagValueType.ANY_STRING)
    group_tag_meta = sly.TagMeta(group_tag_name, sly.TagValueType.ANY_STRING)
    angle_meta = sly.TagMeta("observation angle", sly.TagValueType.ANY_NUMBER)

    pixel_to_class = {
        1: sly.ObjClass("road", sly.AnyGeometry, color=(128, 64, 128)),
        2: sly.ObjClass("sidewalk", sly.AnyGeometry, color=(0, 0, 192)),
        3: sly.ObjClass("building", sly.AnyGeometry, color=(128, 0, 0)),
        4: sly.ObjClass("fence", sly.AnyGeometry, color=(64, 64, 128)),
        5: sly.ObjClass("void", sly.AnyGeometry, color=(0, 0, 0)),
        6: sly.ObjClass("pole", sly.AnyGeometry, color=(192, 192, 128)),
        7: sly.ObjClass("traffic light", sly.AnyGeometry, color=(0, 128, 128)),
        8: sly.ObjClass("traffic sign", sly.AnyGeometry, color=(192, 128, 128)),
        9: sly.ObjClass("vegetation", sly.AnyGeometry, color=(128, 128, 0)),
        11: sly.ObjClass("sky", sly.AnyGeometry, color=(128, 128, 128)),
        12: sly.ObjClass("pedestrian", sly.AnyGeometry, color=(64, 64, 0)),
        15: sly.ObjClass("van", sly.AnyGeometry, color=(230, 25, 75)),
        10: sly.ObjClass("terrain", sly.AnyGeometry, color=(60, 180, 75)),
    }

    cyclist = sly.ObjClass("cyclist", sly.AnyGeometry, color=(255, 225, 25))
    truck = sly.ObjClass("truck", sly.AnyGeometry, color=(0, 130, 200))
    wheelchair = sly.ObjClass("wheelchair", sly.AnyGeometry, color=(245, 130, 48))
    car = sly.ObjClass("car", sly.AnyGeometry, color=(64, 0, 128))
    bicycle = sly.ObjClass("bicycle", sly.AnyGeometry, color=(0, 128, 192))

    meta = sly.ProjectMeta(
        tag_metas=[subfolder_meta, group_tag_meta, angle_meta],
        obj_classes=list(pixel_to_class.values()),
    )

    meta = meta.add_obj_classes([cyclist, truck, wheelchair, car, bicycle])

    api.project.update_meta(project.id, meta.to_json())
    api.project.images_grouping(id=project.id, enable=True, tag_name=group_tag_name)

    for ds_name, data_path in ds_name_to_split.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        images_pathes = glob.glob(data_path + "/*/*/RGB/*.png")

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_pathes))

        for img_pathes_batch in sly.batched(images_pathes, batch_size=batch_size):
            images_pathes_batch = []
            images_names_batch = []
            unique_ids = []
            for im_path in img_pathes_batch:
                prefix = "_".join(tuple(im_path.split("/")[-4].split("_")[1:8]))

                unique_id = (
                    im_path.split("/")[-4]
                    + "_"
                    + im_path.split("/")[-3]
                    + "_"
                    + get_file_name(im_path)
                )
                unique_ids.extend([unique_id, unique_id])
                images_names_batch.append(
                    prefix + "_" + im_path.split("/")[-3] + "_" + get_file_name_with_ext(im_path)
                )
                images_pathes_batch.append(im_path)

                images_names_batch.append(
                    prefix
                    + "_"
                    + im_path.split("/")[-3]
                    + "_depth_"
                    + get_file_name_with_ext(im_path)
                )
                depth_im_path = im_path.replace(rgb_folder, depth_folder)
                images_pathes_batch.append(depth_im_path)

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = []
            for i in range(0, len(images_pathes_batch), 2):
                ann = create_ann(images_pathes_batch[i], unique_ids[i])
                anns.extend([ann, ann])

            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_pathes_batch))

    return project
