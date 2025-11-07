import os

from deepvoxnet2.components.mirc import Case, Dataset, Mirc, NiftiFileModality, Record
from deepvoxnet2.components.sampler import MircSampler


def get_dirs(data_path):
    dirs = []
    for dirpath, dirnames, filenames in os.walk(data_path):
        for f in filenames:
            if f.lower().endswith(".nii.gz"):
                dirs.append(dirpath)
                break
    return dirs


def get_cases_sampler(cases, dataset_id=""):
    """
    Walk data_path and return a list of directory paths that contain at least
    one .nii.gz file. Returned paths are absolute.
    """

    # create the mirc dataset
    dataset = Dataset("HI", dataset_id)
    for _path in cases:
        files = os.listdir(_path)
        print("Processing:", _path)
        op_file = [f for f in files if "OP-gesamt_SV" in f and not f.startswith(".")][0]
        ip_file = [f for f in files if "IP-gesamt_SV" in f and not f.startswith(".")][0]
        cid = os.path.basename(_path)
        mirc_case = Case(_path, cid)
        record = Record("0")
        record.add(NiftiFileModality("OP", os.path.join(_path, op_file)))
        record.add(NiftiFileModality("IP", os.path.join(_path, ip_file)))
        # record.add(
        #     NiftiFileModality(
        #         "Mask", os.path.join(data_path, "Mask_{}_r.nii.gz".format(cid))
        #     )
        # )
        mirc_case.add(record)
        dataset.add(mirc_case)
    mirc = Mirc(dataset)
    return MircSampler(mirc, shuffle=False)


if __name__ == "__main__":
    data_path = "data/MAIN PROJECT 11-25"
    cases = get_cases_sampler(data_path)
    for case in cases:
        print(os.path.basename(case))
