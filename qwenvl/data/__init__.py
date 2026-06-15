import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}
MY_DATASET = { 
    "annotation_path": "/home/zhangxw/share_data/VLD/json/DU_XING_HUA.json", 
    "data_path": "", # Can be empty if paths are in annotations
}
MY_DATASET1 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/ALL.json",
    "data_path": "",
}
MY_DATASET2 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/GOU_GUANG_SHU_T1.json",
    "data_path": "",
}
MY_DATASET3 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/sample.json",
    "data_path": "",
}
MY_DATASET4 = {
    "annotation_path": "/home/zhangxw/share_data/VQA-RAD/train.json",
    "data_path": "",
}
MY_DATASET5 = {
    "annotation_path": "/home/zhangxw/share_data/MEDPIX/train.json",
    "data_path": "",
}
MY_DATASET6 = {
    "annotation_path": "/home/zhangxw/share_data/PMC-VQA/train.json",
    "data_path": "",
}
MY_DATASET7 = {
    "annotation_path": "/home/zhangxw/share_data/SLAKE/train.json",
    "data_path": "",
}
MY_DATASET8 = {
    "annotation_path": "/home/zhangxw/ROCOv2/webo_test.json",
    "data_path": "",
}
MY_DATASET9 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/liu fang qiong.json",
    "data_path": "",
}
MY_DATASET10 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/zhao_hong_FDG_prompt_final.json",
    "data_path": "",
}
MY_DATASET11 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/zhao_hong_T1.json",
    "data_path": "",
}
MY_DATASET12 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/zhao_hong_FDG_prompt_final_1.json",
    "data_path": "",
}
MY_DATASET13 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/MIX.json",
    "data_path": "",
}
MY_DATASET14 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/five_people.json",
    "data_path": "",
}
MY_DATASET15 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/five_people_val.json",
    "data_path": "",
}
MY_DATASET16 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/wang liao ju.json",
    "data_path": "",
}
MY_DATASET17 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/wang yuan qing.json",
    "data_path": "",
}
MY_DATASET18 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/ye ying chun.json",
    "data_path": "",
}
MY_DATASET19 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/YI_DE_FEN.json",
    "data_path": "",
}
MY_DATASET20 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/YOU_XIAN_FANG.json",
    "data_path": "",
}
MY_DATASET21 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/YU_DE_QIANG.json",
    "data_path": "",
}
MY_DATASET22 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/yu xing lian.json",
    "data_path": "",
}
MY_DATASET23 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/ZHANG_KUAN_MING.json",
    "data_path": "",
}
MY_DATASET24 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/ZHANG_MING_SHU.json",
    "data_path": "",
}
MY_DATASET25 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/ZHANG_TAO.json",
    "data_path": "",
}
MY_DATASET26 = {
    "annotation_path": "/home/zhangxw/share_data/VLD/json/zhao_hong.json",
    "data_path": "",
}
MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "my_dataset": MY_DATASET,
    "my_dataset1": MY_DATASET1,
    "my_dataset2": MY_DATASET2,
    "my_dataset3": MY_DATASET3,
    "my_dataset4": MY_DATASET4,
    "my_dataset5": MY_DATASET5,
    "my_dataset6": MY_DATASET6,
    "my_dataset7": MY_DATASET7,
    "my_dataset8": MY_DATASET8,
    "my_dataset9": MY_DATASET9,
    "my_dataset10": MY_DATASET10,
    "my_dataset11": MY_DATASET11,
    "my_dataset12": MY_DATASET12,
    "my_dataset13": MY_DATASET13,
    "my_dataset14": MY_DATASET14,
    "my_dataset15": MY_DATASET15,
    "my_dataset16": MY_DATASET16,
    "my_dataset17": MY_DATASET17,
    "my_dataset18": MY_DATASET18,
    "my_dataset19": MY_DATASET19,
    "my_dataset20": MY_DATASET20,
    "my_dataset21": MY_DATASET21,
    "my_dataset22": MY_DATASET22,
    "my_dataset23": MY_DATASET23,
    "my_dataset24": MY_DATASET24,
    "my_dataset25": MY_DATASET25,
    "my_dataset26": MY_DATASET26,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
