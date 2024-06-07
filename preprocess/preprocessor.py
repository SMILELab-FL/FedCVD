import wfdb
import h5py
import tqdm
import ast
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
import os
import cv2 as cv
import collections
import SimpleITK as sitk
import skimage
import scipy.io as sio
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import guarantee_path


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def loadvideo(filename: str) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv.VideoCapture(filename)

    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))

    video = np.zeros((frame_count, frame_height, frame_width), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        video[count, :, :] = frame

    video = video.transpose((0, 1, 2))
    return video.astype(np.uint8)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


class ECGPreprocessor:
    def __init__(self,
                 meta_input_file: str,
                 meta_output_file: str,
                 ecg_input_path: str,
                 ecg_output_file: str,
                 location: str,
                 input_columns_list: list[str],
                 output_columns_list: list[str],
                 rename_columns: dict,
                 labels_list: list,
                 labels_map: dict,
                 ecg_length: int,
                 ecg_leads: int
                 ):
        self.meta_input_file = meta_input_file
        self.meta_output_file = meta_output_file
        self.ecg_input_path = ecg_input_path
        self.ecg_output_file = ecg_output_file
        self.location = location
        self.input_columns_list = input_columns_list
        self.output_columns_list = output_columns_list
        self.rename_columns = rename_columns
        self.labels_list = labels_list
        self.labels_map = labels_map
        self.ecg_length = ecg_length
        self.ecg_leads = ecg_leads
        self.meta = None
        self.ecg = None

    def label_load(self):
        self.meta = pd.read_csv(self.meta_input_file)
        self.meta = self.meta[self.input_columns_list]
        self.meta["Method"] = pd.Series(np.zeros(self.meta.shape[0], dtype=int))
        self.meta["Location"] = pd.Series(np.full(self.meta.shape[0], self.location, dtype=object))

    def label_save(self):
        self.meta = self.meta.rename(columns=self.rename_columns)
        if self.output_columns_list is not None:
            self.meta[self.output_columns_list].to_csv(self.meta_output_file, index=False, encoding="utf-8")
        else:
            self.meta.to_csv(self.meta_output_file, index=False, encoding="utf-8")

    def run(self):
        self.label_load()
        self.label_preprocess()
        self.ecg_preprocess()
        self.label_save()

    def ecg_filter(self, ecg: np.array) -> (np.array, int):
        if ecg.shape[0] == self.ecg_leads:
            ecg = ecg.T
        nan_flag = np.isnan(ecg).any()
        zero_flag = np.all(ecg == 0)
        flag = 1
        if nan_flag or zero_flag:
            flag = 0
        else:
            if ecg.shape[0] < self.ecg_length:
                ecg = self.ecg_padding(ecg)
                flag = 2
            elif ecg.shape[0] > self.ecg_length:
                ecg = self.ecg_cutting(ecg)
                flag = 3
        return ecg.T, flag

    def ecg_padding(self, ecg: np.array) -> np.array:
        padding_col = (0, 0)
        diff = int(self.ecg_length - ecg.shape[0])
        if diff % 2:
            padding_row = (int(diff / 2) + 1, int(diff / 2))
        else:
            padding_row = (int(diff / 2), int(diff / 2))
        ecg = np.pad(ecg, (padding_row, padding_col), mode="edge")
        return ecg

    def ecg_cutting(self, ecg: np.array) -> np.array:
        return ecg[:self.ecg_length][:]

    @abstractmethod
    def label_preprocess(self):
        pass

    @abstractmethod
    def label_split(self, labels):
        pass

    @abstractmethod
    def label_aggregate(self, label_list):
        pass

    @abstractmethod
    def ecg_preprocess(self):
        pass


class SPHPreprocessor(ECGPreprocessor):
    def label_preprocess(self):
        del_rows = []
        for idx, row in tqdm.tqdm(self.meta.iterrows(), total=self.meta.shape[0]):
            label_code = self.label_aggregate(self.label_split(row["AHA_Code"]))
            age_flag = (row["Age"] > 0) and (row["Age"] < 120)
            if label_code and age_flag:
                self.meta.loc[idx, "AHA_Code"] = label_code
            else:
                del_rows.append(idx)
        self.meta.drop(del_rows, inplace=True)
        self.meta.reset_index(inplace=True)

    # 提取 AHA_Code 中主要标签并分割为列表形式
    def label_split(self, labels: str) -> list:
        label_list = []
        # 提取标签
        for labels in labels.split(';'):
            # 筛选主标签
            for label in labels.split('+'):
                x = int(label)
                label_list.append(label)
        label_list = list(set(label_list).intersection(set(self.labels_list)))
        return label_list

    def label_aggregate(self, label_list: list) -> str:
        map_list = [self.labels_map[label] for label in label_list]
        map_set = set(map_list)
        map_list = sorted(list(map_set))
        return ";".join(map_list)

    def ecg_preprocess(self):
        output_file = h5py.File(self.ecg_output_file, "w")
        ecg_id = 0
        del_rows = []
        for idx, row in tqdm.tqdm(self.meta.iterrows(), total=self.meta.shape[0]):
            dataset = row["ECG_ID"]
            file = h5py.File(self.ecg_input_path + dataset + ".h5", "r")
            ecg, flag = self.ecg_filter(file["ecg"][:][:])
            if flag != 0:
                output_file[f"{ecg_id:06d}"] = ecg
                self.meta.loc[idx, "ECG_ID"] = f"{ecg_id:06d}"
                self.meta.loc[idx, "Method"] = flag
                ecg_id += 1
            else:
                del_rows.append(idx)
            file.close()
        output_file.close()
        self.meta.drop(del_rows, inplace=True)
        self.meta.reset_index(inplace=True)


class PTBPreprocessor(ECGPreprocessor):
    def label_preprocess(self):
        del_rows = []
        self.meta["scp_codes"] = self.meta["scp_codes"].apply(lambda x: ast.literal_eval(x))
        self.meta["age"] = self.meta["age"].apply(lambda x: int(x))
        for idx, row in tqdm.tqdm(self.meta.iterrows(), total=self.meta.shape[0]):
            label_code = self.label_aggregate(self.label_split(row["scp_codes"]))
            age_flag = (int(row["age"]) > 0) and (int(row["age"]) < 120)
            if label_code and age_flag:
                self.meta.loc[idx, "scp_codes"] = label_code
                self.meta.loc[idx, "age"] = int(self.meta.loc[idx, "age"])
                self.meta.loc[idx, "sex"] = self._sex_map(self.meta.loc[idx, "sex"])
            else:
                del_rows.append(idx)
        self.meta.drop(del_rows, inplace=True)
        self.meta.reset_index(inplace=True)

    def label_split(self, labels: dict) -> list:
        label_list = [key for key in labels.keys()]
        label_list = list(set(label_list).intersection(set(self.labels_list)))
        return label_list

    def label_aggregate(self, label_list: list) -> str:
        map_list = [self.labels_map[label] for label in label_list]
        map_list = sorted(list(set(map_list)))
        return ";".join(map_list)

    def ecg_preprocess(self):
        output_file = h5py.File(self.ecg_output_file, "w")
        ecg_id = 0
        del_rows = []
        for idx, row in tqdm.tqdm(self.meta.iterrows(), total=self.meta.shape[0]):
            ecg = [wfdb.rdsamp(self.ecg_input_path + row['filename_hr'])]
            ecg = np.array([signal for signal, meta in ecg], dtype=np.float16)[0]
            ecg, flag = self.ecg_filter(ecg)
            if flag != 0:
                output_file[f"{ecg_id:06d}"] = ecg
                self.meta.loc[idx, "ecg_id"] = f"{ecg_id:06d}"
                self.meta.loc[idx, "Method"] = flag
                ecg_id += 1
            else:
                del_rows.append(idx)
        output_file.close()
        self.meta.drop(del_rows, inplace=True)
        self.meta.reset_index(inplace=True)

    @staticmethod
    def _sex_map(sex: int) -> str:
        sex_map = {0: "M", 1: "F"}
        return sex_map[sex]


class SXPHPreprocessor(ECGPreprocessor):
    def label_load(self):
        self.meta = pd.DataFrame({}, columns=self.output_columns_list)

    def label_preprocess(self):
        pass

    def label_split(self, labels: str) -> list[str, ...]:
        label_list = labels.split(",")
        for i in range(len(label_list)):
            label_list[i] = label_list[i].strip()
        label_list = list(set(label_list).intersection(set(self.labels_list)))
        return label_list

    def label_aggregate(self, label_list: list[str, ...]) -> str:
        map_list = [self.labels_map[label] for label in label_list]
        map_list = sorted(list(set(map_list)))
        return ";".join(map_list)

    def ecg_preprocess(self):
        output_file = h5py.File(self.ecg_output_file, "w")
        ecg_id = 0
        del_rows = []
        base_path = Path(self.ecg_input_path)
        first_level_directory_list = sorted(
            [directory for directory in base_path.iterdir() if directory.is_dir()]
        )
        for first_level_directory in tqdm.tqdm(first_level_directory_list):
            second_level_directory_list = sorted(
                [directory for directory in first_level_directory.iterdir() if directory.is_dir()]
            )
            for second_level_directory in second_level_directory_list:
                file_stem_list = sorted(
                    set([path.stem for path in second_level_directory.iterdir()
                         if path.is_file() and path.name != "RECORDS"])
                )
                for file_stem in file_stem_list:
                    ecg, meta = wfdb.rdsamp(str(Path(second_level_directory, file_stem)))
                    comments = {}
                    for item in meta["comments"]:
                        key, value = str(item).split(":")
                        comments[key] = value
                    label_code = self.label_aggregate(self.label_split(comments["Dx"]))
                    if comments["Age"].strip() != "NaN":
                        age = int(comments["Age"])
                    else:
                        age = -1
                    sex = self._sex_map(comments["Sex"])
                    age_flag = ((age > 0) and (age < 120))
                    if label_code and age_flag and sex:
                        ecg = np.array(ecg, dtype=np.float16)
                        ecg, flag = self.ecg_filter(ecg)
                        if flag != 0:
                            output_file[f"{ecg_id:06d}"] = ecg
                            self.meta.loc[ecg_id] = {
                                "ECG_ID": f"{ecg_id:06d}", "Code_Label": label_code, "Age": age,
                                "Sex": sex, "Method": flag, "Location": self.location
                            }
                            ecg_id += 1

    @staticmethod
    def _sex_map(sex: str) -> str | None:
        sex = sex.strip()
        sex_map = {"Male": "M", "Female": "F", "Unknown": None}
        return sex_map[sex]


class GEPreprocessor(ECGPreprocessor):
    def label_load(self):
        self.meta = pd.DataFrame({}, columns=self.output_columns_list)

    def label_preprocess(self):
        pass

    def label_split(self, labels: str) -> list[str, ...]:
        label_list = labels.split(",")
        for i in range(len(label_list)):
            label_list[i] = label_list[i].strip()
        label_list = list(set(label_list).intersection(set(self.labels_list)))
        return label_list

    def label_aggregate(self, label_list: list[str, ...]) -> str:
        map_list = [self.labels_map[label] for label in label_list]
        map_list = sorted(list(set(map_list)))
        return ";".join(map_list)

    def ecg_preprocess(self):
        output_file = h5py.File(self.ecg_output_file, "w")
        ecg_id = 0
        del_rows = []
        base_path = Path(self.ecg_input_path)
        directory_list = sorted(
            [directory for directory in base_path.iterdir() if directory.is_dir()]
        )
        for directory in tqdm.tqdm(directory_list):
            file_stem_list = sorted(
                set([path.stem for path in directory.iterdir()
                     if path.is_file() and path.name != "RECORDS"])
            )
            for file_stem in file_stem_list:
                ecg, meta = wfdb.rdsamp(str(Path(directory, file_stem)))
                comments = {}
                for item in meta["comments"]:
                    key, value = str(item).split(":")
                    comments[key] = value
                label_code = self.label_aggregate(self.label_split(comments["Dx"]))
                if comments["Age"].strip() != "NaN":
                    age = int(comments["Age"])
                else:
                    age = -1
                sex = self._sex_map(comments["Sex"])
                age_flag = ((age > 0) and (age < 120))
                if label_code and age_flag and sex:
                    ecg = np.array(ecg, dtype=np.float16)
                    ecg, flag = self.ecg_filter(ecg)
                    if flag != 0:
                        output_file[f"{ecg_id:06d}"] = ecg
                        self.meta.loc[ecg_id] = {
                            "ECG_ID": f"{ecg_id:06d}", "Code_Label": label_code, "Age": age,
                            "Sex": sex, "Method": flag, "Location": self.location
                        }
                        ecg_id += 1

    @staticmethod
    def _sex_map(sex: str) -> str | None:
        sex = sex.strip()
        sex_map = {"Male": "M", "Female": "F", "Unknown": None}
        return sex_map[sex]


class CamusPreprocessor:
    def __init__(self,
                 input_path: str,
                 meta_output_file: str,
                 echo_output_file: str,
                 output_columns_list: list[str],
                 view: str,
                 resize: tuple[int, int] = (112, 112)
                 ):
        self.input_path = input_path
        self.meta_output_file = meta_output_file
        self.echo_output_file = echo_output_file
        self.view = view
        self.meta = []
        self.idx = 0
        self.resize = resize
        self.output_columns_list = output_columns_list

    def convert(self, input_path):
        output_file = h5py.File(self.echo_output_file, "w")
        for directory in ["training", "testing"]:
            path = os.path.join(input_path, directory)
            for case in tqdm.tqdm(sorted(os.listdir(path))):
                case_path = os.path.join(path, case)
                if case == ".DS_Store" or os.path.isfile(case_path):
                    continue
                if os.listdir(case_path):
                    with open(os.path.join(case_path, "Info_" + self.view + ".cfg"), "r") as file:
                        data = {"ECHO_ID": f"{self.idx:06d}"}
                        for line in file:
                            key, value = line.split(":")
                            key = key.strip()
                            value = value.strip()
                            # value = value[:-1]
                            if value.isdigit():
                                value = int(value)
                            elif is_float(value):
                                value = float(value)
                            data[key] = value
                        end_frame = int(max((data["ED"], data["ES"])))
                        start_frame = int(min((data["ED"], data["ES"])))
                    data["ED"] = int(data["ED"] - start_frame)
                    data["ES"] = int(data["ES"] - start_frame)
                    data["NbFrame"] = int(end_frame - start_frame + 1)
                    case_identifier = f"{case}_{self.view}_sequence"
                    video = sitk.ReadImage(os.path.join(case_path, f"{case_identifier}.mhd"))
                    video = sitk.GetArrayFromImage(video)[start_frame - 1: end_frame]
                    video = np.array([cv.resize(v, self.resize) for v in video])
                    f, h, w = video.shape
                    c = 1
                    video = video.reshape(c, f, h, w)
                    label_dict = {}
                    for instant in ["ED", "ES"]:
                        case_identifier = f"{case}_{self.view}_{instant}"
                        label = sitk.ReadImage(
                            os.path.join(case_path, f"{case_identifier}_gt.mhd")
                        )
                        label = sitk.GetArrayFromImage(label)[0].astype(np.uint8)
                        r_label = np.zeros(self.resize, dtype=np.uint8)
                        for idx in range(4):
                            mask = cv.resize(np.where(label == idx, 1, 0).astype(np.uint8), self.resize)
                            r_label[np.where(mask == 1)] = idx
                        label_dict[data[instant]] = r_label
                    # sort label_dict with key
                    label_dict = dict(sorted(label_dict.items(), key=lambda x: x[0]))
                    group = output_file.create_group(f"{self.idx:06d}")
                    group["video"] = video
                    group["mask"] = np.array(list(label_dict.values()))
                    self.meta.append(data)
                    self.idx += 1
        output_file.close()

    def run(self):
        self.convert(self.input_path)
        self.meta = pd.DataFrame(self.meta)
        self.meta["Location"] = "client1"
        self.meta["LabelType"] = "0;1;2;3"
        self.meta = self.meta.astype({
            "ECHO_ID": str,
            "Age": int,
            "Sex": str,
            "ED": int,
            "ES": int,
            "NbFrame": int,
            "ImageQuality": str,
            "LVedv": float,
            "LVesv": float,
            "LVef": float,
            "LabelType": str,
            "Location": str
        })
        self.meta[self.output_columns_list].to_csv(self.meta_output_file, index=False, encoding="utf-8")


class EchoPreprocessor:
    def __init__(self,
                 input_path: str,
                 meta_output_file: str,
                 echo_output_file: str,
                 resize: tuple[int, int] = (112, 112),
                 ):
        self.input_path = input_path
        self.meta_output_file = meta_output_file
        self.echo_output_file = echo_output_file
        self.meta = []
        self.file_list = pd.read_csv(os.path.join(input_path, "FileList.csv"))
        frames = collections.defaultdict(list)
        trace = collections.defaultdict(_defaultdict_of_lists)
        with open(os.path.join(input_path, "VolumeTracings.csv"), "r") as file:
            header = file.readline().strip().split(",")
            assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

            for line in file:
                filename, x1, y1, x2, y2, frame = line.strip().split(',')
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                frame = int(frame)
                if frame not in trace[filename]:
                    frames[filename].append(frame)
                trace[filename][frame].append((x1, y1, x2, y2))
            for filename in frames:
                for frame in frames[filename]:
                    trace[filename][frame] = np.array(trace[filename][frame])
        self.volume_tracing = trace
        self.frames = frames
        self.resize = resize
        self.idx = 0
        self.output_columns_list = ["ECHO_ID", "NbFrame", "LVedv", "LVesv", "LVef", "LabelType", "Location"]

    def convert(self, file_df):
        output_file = h5py.File(self.echo_output_file, "w")

        for index, row in tqdm.tqdm(file_df.iterrows(), total=len(file_df)):
            video = loadvideo(
                str(os.path.join(self.input_path, "Videos", row["FileName"] + ".avi")),
            )
            masks = {}
            for frame in self.frames[f"{row['FileName']}.avi"]:
                trace = self.volume_tracing[f"{row['FileName']}.avi"][frame]
                x1, y1, x2, y2 = trace[:, 0], trace[:, 1], trace[:, 2], trace[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))
                r, c = skimage.draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int),
                                            (video.shape[1], video.shape[2]))
                mask = np.zeros((video.shape[1], video.shape[2]), np.uint8)
                mask[r, c] = 1
                masks[frame] = mask

            masks = dict(sorted(masks.items(), key=lambda e: e[0]))
            start_frame = int(min(masks.keys()))
            end_frame = int(max(masks.keys()))
            video = video[start_frame:end_frame + 1]
            video = np.array([cv.resize(v, self.resize) for v in video])
            # print(f"{row['FileName']}.avi")
            # print(video.shape)
            f, h, w = video.shape
            c = 1
            video = video.reshape(c, f, h, w)
            group = output_file.create_group(f"{self.idx:06d}")
            group["video"] = video
            group["mask"] = np.array(list(masks.values()))
            data = {
                "ECHO_ID": f"{self.idx:06d}",
                "NbFrame": end_frame - start_frame + 1,
                "LVedv": row["EDV"], "LVesv": row["ESV"], "LVef": row["EF"],
            }
            self.meta.append(data)
            self.idx += 1
        output_file.close()

    def run(self):
        self.convert(
            self.file_list
        )
        self.meta = pd.DataFrame(self.meta)
        self.meta["LabelType"] = "0;1"
        self.meta["Location"] = "client2"
        self.meta = self.meta.astype({
            "ECHO_ID": str,
            "NbFrame": int,
            "LVedv": float,
            "LVesv": float,
            "LVef": float,
            "LabelType": str,
            "Location": str
        })
        self.meta[self.output_columns_list].to_csv(self.meta_output_file, index=False, encoding="utf-8")


class HMCQUPreprocessor:
    def __init__(self,
                 input_path: str,
                 meta_output_file: str,
                 echo_output_file: str,
                 resize: tuple[int, int] = (112, 112)
                 ):
        self.input_path = input_path
        self.meta_output_file = meta_output_file
        self.echo_output_file = echo_output_file
        self.resize = resize
        self.file_list = pd.read_excel(os.path.join(self.input_path, "A4C.xlsx"))
        self.meta = []
        self.idx = 0
        self.file_list = self.file_list[self.file_list["LV Wall Ground-truth Segmentation Masks"] == "ü"]

    def convert(self, file_list, input_path):
        output_file = h5py.File(self.echo_output_file, "w")

        video_path = os.path.join(input_path, "HMC-QU", "A4C")
        mask_path = os.path.join(input_path, "LV Ground-truth Segmentation Masks")

        for index, row in tqdm.tqdm(file_list.iterrows(), total=len(self.file_list)):
            video = loadvideo(str(os.path.join(video_path, row["ECHO"] + ".avi")))
            mask_file = sio.loadmat(str(os.path.join(mask_path, "Mask_" + row["ECHO"] + ".mat")))
            mask = mask_file["predicted"].astype(np.uint8)
            start_frame = row["One cardiac-cycle frames"] - 1
            end_frame = start_frame + mask.shape[0]
            video = video[start_frame: end_frame]
            video = np.array([cv.resize(v, self.resize).astype(np.uint8) for v in video])
            f, h, w = video.shape
            c = 1
            video = video.reshape(c, f, h, w)
            mask = np.array([cv.resize(m, self.resize).astype(np.uint8) for m in mask])
            mask[np.where(mask == 1)] = 2
            group = output_file.create_group(f"{self.idx:06d}")
            group["video"] = video
            group["mask"] = mask
            data = {
                "ECHO_ID": f"{self.idx:06d}",
                "NbFrame": end_frame - start_frame
            }
            self.meta.append(data)
            self.idx += 1
        output_file.close()

    def run(self):
        self.convert(
            self.file_list,
            self.input_path
        )
        self.meta = pd.DataFrame(self.meta)
        self.meta["LabelType"] = "0;2"
        self.meta["Location"] = "client3"
        self.meta = self.meta.astype({
            "ECHO_ID": str,
            "NbFrame": int,
            "LabelType": str,
            "Location": str
        })
        self.meta[["ECHO_ID", "NbFrame", "LabelType", "Location"]].to_csv(self.meta_output_file, index=False,
                                                                          encoding="utf-8")


def preprocess_ecg_sph(
        input_path: str,
        output_path: str,
):
    location = "client1"
    if output_path[-1] == "/":
        output_path = output_path + "ECG/preprocessed/" + location + "/"
    else:
        output_path = output_path + "/ECG/preprocessed/" + location + "/"
    guarantee_path(output_path)
    meta_input_file = input_path + "metadata.csv" if input_path[-1] == "/" else input_path + "/metadata.csv"
    meta_output_file = output_path + "metadata.csv"
    ecg_input_path = input_path + "records/" if input_path[-1] == "/" else input_path + "/records/"
    ecg_output_file = output_path + "records.h5"
    input_columns_list = ["ECG_ID", "AHA_Code", "Age", "Sex"]
    output_columns_list = ["ECG_ID", "Code_Label", "Age", "Sex", "Method", "Location"]
    rename_columns = {"ECG_ID": "ECG_ID", "AHA_Code": "Code_Label", "Age": "Age", "Sex": "Sex"}
    labels_list = [
        "1",
        "21", "22", "23",
        "30",
        "50", "51", "55",
        "60",
        "83", "84", "85", "86", "87", "88",
        "101", "102", "104", "105", "106",
        "140", "142", "143",
        "160", "161", "165"
    ]
    labels_map = {
        "1": "0",
        "21": "1", "22": "2", "23": "3",
        "30": "4",
        "50": "5", "51": "6", "55": "7",
        "60": "8",
        "83": "10", "84": "10", "85": "10", "86": "10", "87": "10", "88": "11",
        "101": "12", "102": "12", "104": "12", "105": "13", "106": "13",
        "140": "14", "142": "15", "143": "16",
        "160": "17", "161": "18", "165": "19"
    }
    ecg_length = 5000
    ecg_leads = 12
    preprocessor = SPHPreprocessor(
        meta_input_file=meta_input_file,
        meta_output_file=meta_output_file,
        ecg_input_path=ecg_input_path,
        ecg_output_file=ecg_output_file,
        location=location,
        input_columns_list=input_columns_list,
        output_columns_list=output_columns_list,
        rename_columns=rename_columns,
        labels_list=labels_list,
        labels_map=labels_map,
        ecg_length=ecg_length,
        ecg_leads=ecg_leads
    )
    preprocessor.run()


def preprocess_ecg_ptb(
        input_path: str,
        output_path: str,
):
    location = "client2"
    if output_path[-1] == "/":
        output_path = output_path + "ECG/preprocessed/" + location + "/"
    else:
        output_path = output_path + "/ECG/preprocessed/" + location + "/"
    guarantee_path(output_path)
    meta_input_file = input_path + "ptbxl_database.csv" if input_path[-1] == "/" else input_path + "/ptbxl_database.csv"
    meta_output_file = output_path + "metadata.csv"
    ecg_input_path = input_path if input_path[-1] == "/" else input_path + "/"
    ecg_output_file = output_path + "records.h5"
    input_columns_list = ["age", "sex", "scp_codes", "ecg_id", "filename_hr"]
    output_columns_list = ["ECG_ID", "Code_Label", "Age", "Sex", "Method", "Location"]
    rename_columns = {"ecg_id": "ECG_ID", "scp_codes": "Code_Label", "age": "Age", "sex": "Sex"}
    labels_list = [
        "NORM",
        "STACH", "SBRAD", "SARRH",
        "PAC",
        "AFIB", "AFLT", "SVTAC",
        "PVC",
        "1AVB", "2AVB", "3AVB",
        "LAFB", "LPFB", "CLBBB", "IRBBB", "CRBBB",
        "LAO/LAE", "LVH", "RVH",
        "AMI", "IMI", "ASMI"
    ]
    labels_map = {
        "NORM": "0",
        "STACH": "1", "SBRAD": "2", "SARRH": "3",
        "PAC": "4",
        "AFIB": "5", "AFLT": "6", "SVTAC": "7",
        "PVC": "8",
        "1AVB": "9", "2AVB": "10", "3AVB": "11",
        "LAFB": "12", "LPFB": "12", "CLBBB": "12", "IRBBB": "13", "CRBBB": "13",
        "LAO/LAE": "14", "LVH": "15", "RVH": "16",
        "AMI": "17", "IMI": "18", "ASMI": "19"
    }
    ecg_length = 5000
    ecg_leads = 12

    preprocessor = PTBPreprocessor(
        meta_input_file=meta_input_file,
        meta_output_file=meta_output_file,
        ecg_input_path=ecg_input_path,
        ecg_output_file=ecg_output_file,
        location=location,
        input_columns_list=input_columns_list,
        output_columns_list=output_columns_list,
        rename_columns=rename_columns,
        labels_list=labels_list,
        labels_map=labels_map,
        ecg_length=ecg_length,
        ecg_leads=ecg_leads
    )
    preprocessor.run()


def preprocess_ecg_sxph(
        input_path: str,
        output_path: str,
):
    location = "client3"
    if output_path[-1] == "/":
        output_path = output_path + "ECG/preprocessed/" + location + "/"
    else:
        output_path = output_path + "/ECG/preprocessed/" + location + "/"
    guarantee_path(output_path)
    meta_input_file = ""
    meta_output_file = output_path + "metadata.csv"
    ecg_input_path = input_path + "WFDBRecords/" if input_path[-1] == "/" else input_path + "/WFDBRecords/"
    ecg_output_file = output_path + "records.h5"
    input_columns_list = []
    output_columns_list = ["ECG_ID", "Code_Label", "Age", "Sex", "Method", "Location"]
    rename_columns = {}
    labels_list = [
        "164854000",
        "427084000", "426177001", "427393009",
        "164885009",
        "164889003", "164890007", "426761007", "164884008",
        "270492004",
        "54016002", "28189009", "164903001", "195042002", "284941000119107",
        "27885002",
        "445118002", "445211001", "164909002",
        "713426002", "59118001", "164907000",
        "67741000119109", "164873001", "164877000",
        "429731003", "7326005", "164868007"
    ]
    labels_map = {
        "164854000": "0",
        "427084000": "1", "426177001": "2", "427393009": "3",
        "164885009": "4",
        "164889003": "5", "164890007": "6", "426761007": "7", "164884008": "8",
        "270492004": "9",
        "54016002": "10", "28189009": "10", "164903001": "10", "195042002": "10", "284941000119107": "10",
        "27885002": "11",
        "445118002": "12", "445211001": "12", "164909002": "12",
        "713426002": "13", "59118001": "13", "164907000": "13",
        "67741000119109": "14", "164873001": "15", "164877000": "16",
        "429731003": "17", "7326005": "18", "164868007": "19"
    }
    ecg_length = 5000
    ecg_leads = 12
    preprocessor = SXPHPreprocessor(
        meta_input_file=meta_input_file,
        meta_output_file=meta_output_file,
        ecg_input_path=ecg_input_path,
        ecg_output_file=ecg_output_file,
        location=location,
        input_columns_list=input_columns_list,
        output_columns_list=output_columns_list,
        rename_columns=rename_columns,
        labels_list=labels_list,
        labels_map=labels_map,
        ecg_length=ecg_length,
        ecg_leads=ecg_leads
    )
    preprocessor.run()


def preprocess_ecg_geo(
        input_path: str,
        output_path: str,
):
    location = "client4"
    if output_path[-1] == "/":
        output_path = output_path + "ECG/preprocessed/" + location + "/"
    else:
        output_path = output_path + "/ECG/preprocessed/" + location + "/"
    guarantee_path(output_path)
    meta_input_file = ""
    meta_output_file = output_path + "metadata.csv"
    ecg_input_path = input_path if input_path[-1] == "/" else input_path + "/"
    ecg_output_file = output_path + "records.h5"
    input_columns_list = []
    output_columns_list = ["ECG_ID", "Code_Label", "Age", "Sex", "Method", "Location"]
    rename_columns = {}
    labels_list = [
        "164854000",
        "427084000", "426177001", "427393009",
        "164885009",
        "164889003", "164890007", "426761007", "164884008",
        "270492004",
        "54016002", "28189009", "164903001", "195042002", "284941000119107",
        "27885002",
        "445118002", "445211001", "164909002",
        "713426002", "59118001", "164907000",
        "67741000119109", "164873001", "164877000",
        "429731003", "7326005", "164868007"
    ]
    labels_map = {
        "164854000": "0",
        "427084000": "1", "426177001": "2", "427393009": "3",
        "164885009": "4",
        "164889003": "5", "164890007": "6", "426761007": "7", "164884008": "8",
        "270492004": "9",
        "54016002": "10", "28189009": "10", "164903001": "10", "195042002": "10", "284941000119107": "10",
        "27885002": "11",
        "445118002": "12", "445211001": "12", "164909002": "12",
        "713426002": "13", "59118001": "13", "164907000": "13",
        "67741000119109": "14", "164873001": "15", "164877000": "16",
        "429731003": "17", "7326005": "18", "164868007": "19"
    }
    ecg_length = 5000
    ecg_leads = 12
    preprocessor = GEPreprocessor(
        meta_input_file=meta_input_file,
        meta_output_file=meta_output_file,
        ecg_input_path=ecg_input_path,
        ecg_output_file=ecg_output_file,
        location=location,
        input_columns_list=input_columns_list,
        output_columns_list=output_columns_list,
        rename_columns=rename_columns,
        labels_list=labels_list,
        labels_map=labels_map,
        ecg_length=ecg_length,
        ecg_leads=ecg_leads
    )
    preprocessor.run()


def preprocess_echo_camus(
        input_path: str,
        output_path: str,
):
    location = "client1"
    if output_path[-1] == "/":
        output_path = output_path + "ECHO/preprocessed/" + location + "/"
    else:
        output_path = output_path + "/ECHO/preprocessed/" + location + "/"
    guarantee_path(output_path)
    input_path = input_path if input_path[-1] == "/" else input_path + "/"
    meta_output_file = output_path + "metadata.csv"
    echo_output_file = output_path + "records.h5"
    output_columns_list = ["ECHO_ID", "Age", "Sex", "ED", "ES", "NbFrame", "ImageQuality", "LVedv",
                           "LVesv", "LVef", "LabelType", "Location"]
    view = "4CH"
    resize = (112, 112)
    preprocessor = CamusPreprocessor(
        input_path=input_path,
        meta_output_file=meta_output_file,
        echo_output_file=echo_output_file,
        output_columns_list=output_columns_list,
        view=view,
        resize=resize
    )
    preprocessor.run()


def preprocess_echo_dynamic(
        input_path: str,
        output_path: str,
):
    location = "client2"
    if output_path[-1] == "/":
        output_path = output_path + "ECHO/preprocessed/" + location + "/"
    else:
        output_path = output_path + "/ECHO/preprocessed/" + location + "/"
    guarantee_path(output_path)
    input_path = input_path if input_path[-1] == "/" else input_path + "/"
    meta_output_file = output_path + "metadata.csv"
    echo_output_file = output_path + "records.h5"
    resize = (112, 112)
    preprocessor = EchoPreprocessor(
        input_path=input_path,
        meta_output_file=meta_output_file,
        echo_output_file=echo_output_file,
        resize=resize
    )
    preprocessor.run()


def preprocess_echo_hmc(
        input_path: str,
        output_path: str,
):
    location = "client3"
    if output_path[-1] == "/":
        output_path = output_path + "ECHO/preprocessed/" + location + "/"
    else:
        output_path = output_path + "/ECHO/preprocessed/" + location + "/"
    guarantee_path(output_path)
    input_path = input_path if input_path[-1] == "/" else input_path + "/"
    meta_output_file = output_path + "metadata.csv"
    echo_output_file = output_path + "records.h5"
    resize = (112, 112)
    preprocessor = HMCQUPreprocessor(
        input_path=input_path,
        meta_output_file=meta_output_file,
        echo_output_file=echo_output_file,
        resize=resize
    )
    preprocessor.run()


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--dataset_name", type=str, default="")

if __name__ == "__main__":
    args = parser.parse_args()
    input_path = args.input_path
    input_path = input_path if input_path[-1] == "/" else input_path + "/"
    if args.dataset_name == "sph":
        preprocess_ecg_sph(input_path + "ECG/raw/SPH/", args.output_path)
    elif args.dataset_name == "ptb":
        preprocess_ecg_ptb(input_path + "ECG/raw/PTB/", args.output_path)
    elif args.dataset_name == "sxph":
        preprocess_ecg_sxph(input_path + "ECG/raw/SXPH/", args.output_path)
    elif args.dataset_name == "g12ec":
        preprocess_ecg_geo(input_path + "ECG/raw/G12EC/", args.output_path)
    elif args.dataset_name == "camus":
        preprocess_echo_camus(input_path + "ECHO/raw/CAMUS/", args.output_path)
    elif args.dataset_name == "echonet":
        preprocess_echo_dynamic(input_path + "ECHO/raw/ECHONET/", args.output_path)
    elif args.dataset_name == "hmcqu":
        preprocess_echo_hmc(input_path + "ECHO/raw/HMCQU/", args.output_path)
    else:
        raise ValueError("Invalid dataset name")
