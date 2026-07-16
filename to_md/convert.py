import json
import argparse
import os
import sys
from itertools import count
from pathlib import Path

REQUIRED_AI_FIELDS = (
    "tldr",
    "motivation",
    "method",
    "result",
    "conclusion",
)


def has_complete_ai(item):
    ai_data = item.get("AI")
    if not isinstance(ai_data, dict):
        return False
    return all(
        isinstance(ai_data.get(field), str)
        and ai_data[field].strip()
        and ai_data[field] != "Error"
        for field in REQUIRED_AI_FIELDS
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the jsonline file")
    args = parser.parse_args()
    data = []
    preference = os.environ.get('CATEGORIES', 'cs.CV, cs.CL').split(',')
    preference = list(map(lambda x: x.strip(), preference))
    def rank(cate):
        if cate in preference:
            return preference.index(cate)
        else:
            return len(preference)

    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    valid_data = []
    for item in data:
        if has_complete_ai(item):
            valid_data.append(item)
        else:
            print(
                f"Skipping item {item.get('id', 'unknown')}: incomplete AI data",
                file=sys.stderr,
            )

    if data and not valid_data:
        raise RuntimeError("No papers contain complete AI enhancement data")
    data = valid_data

    categories = set([item["categories"][0] for item in data])
    template = (Path(__file__).resolve().parent / "paper_template.md").read_text()
    categories = sorted(categories, key=rank)
    cnt = {cate: 0 for cate in categories}
    for item in data:
        if item["categories"][0] not in cnt.keys():
            continue
        cnt[item["categories"][0]] += 1

    markdown = f"<div id=toc></div>\n\n# Table of Contents\n\n"
    for idx, cate in enumerate(categories):
        markdown += f"- [{cate}](#{cate}) [Total: {cnt[cate]}]\n"

    idx = count(1)
    for cate in categories:
        markdown += f"\n\n<div id='{cate}'></div>\n\n"
        markdown += f"# {cate} [[Back]](#toc)\n\n"
        markdown += "\n\n".join(
            [
                template.format(
                    title=item["title"],
                    authors=",".join(item["authors"]),
                    summary=item["summary"],
                    url=item['abs'],
                    tldr=item['AI']['tldr'],
                    motivation=item['AI']['motivation'],
                    method=item['AI']['method'],
                    result=item['AI']['result'],
                    conclusion=item['AI']['conclusion'],
                    cate=item['categories'][0],
                    idx=next(idx)
                )
                for item in data if item["categories"][0] == cate
            ]
        )
    data_path = Path(args.data)
    output_path = data_path.with_name(data_path.name.split('_', 1)[0] + '.md')
    with output_path.open("w") as f:
        f.write(markdown)
