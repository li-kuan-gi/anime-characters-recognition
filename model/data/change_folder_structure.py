from pathlib import Path
import shutil
import argparse


def change_structure(folder_name):
    folder_name = folder_name.rstrip("/")
    data_dir = Path(folder_name)
    temp_dir = Path(f"{folder_name}_temp")

    for work_dir in data_dir.iterdir():
        for char_dir in work_dir.iterdir():
            for phase_dir in char_dir.iterdir():
                _, work, char, phase = phase_dir.parts
                dir_path = temp_dir / phase / f"{work}::::::{char}"
                dir_path.mkdir(parents=True)
                for file_path in phase_dir.iterdir():
                    *_, file_name = file_path.parts
                    shutil.move(file_path, dir_path / file_name)

    shutil.rmtree(data_dir)
    temp_dir.rename(Path(folder_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str)
    args = parser.parse_args()
    change_structure(args.folder)
