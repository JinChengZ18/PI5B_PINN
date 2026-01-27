"""生成 index.jsonl 文件，从 CSV 文件元数据中提取参数"""
import json
import re
from pathlib import Path


def parse_csv_header(csv_path: Path):
    """解析 CSV 文件头部的元数据"""
    with csv_path.open('r', encoding='utf-8') as f:
        lines = []
        for line in f:
            if line.startswith('%'):
                lines.append(line[1:].strip())
            else:
                break

    case_id = csv_path.stem

    metadata = {}
    for line in lines:
        if ',' in line:
            key, value = line.split(',', 1)
            metadata[key.strip()] = value.strip().strip('"')

    return case_id, metadata


def generate_index(dataset_dir: Path, output_file: Path):
    """生成 index.jsonl 文件"""
    csv_files = sorted(dataset_dir.glob("case_*.csv"))

    if not csv_files:
        print(f"在 {dataset_dir} 中未找到 CSV 文件")
        return

    print(f"找到 {len(csv_files)} 个 CSV 文件")

    entries = []
    for csv_file in csv_files:
        case_id, metadata = parse_csv_header(csv_file)

        match = re.search(r'(\d+)', case_id)
        if match:
            idx = int(match.group(1)) - 1

            soc_values = [2.0, 6.0, 10.0, 12.0]
            pmic_values = [0.2, 0.5, 1.0, 1.5]
            usb_values = [0.1, 0.3, 0.5]
            other_values = [0.2, 0.5, 1.0]

            i_other = idx % len(other_values)
            idx //= len(other_values)
            i_usb = idx % len(usb_values)
            idx //= len(usb_values)
            i_pmic = idx % len(pmic_values)
            i_soc = idx // len(pmic_values)

            parameters = {
                "soc_power": soc_values[i_soc],
                "pmic_power": pmic_values[i_pmic],
                "usb_power": usb_values[i_usb],
                "other_power": other_values[i_other],
            }
        else:
            parameters = {}

        entry = {
            "case_id": case_id,
            #"export_file": str(csv_file.relative_to(base_dir)),
            "export_file": str(csv_file.resolve()),
            "parameters": parameters,
            "metadata": metadata,
        }
        entries.append(entry)

    with output_file.open('w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"生成 index.jsonl: {output_file}")
    print(f"共 {len(entries)} 条记录")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    dataset_dir = base_dir / "data" / "thermal_heat_source"
    output_file = dataset_dir / "index.jsonl"

    generate_index(dataset_dir, output_file)
