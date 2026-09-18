"""Small synthetic text pilot; tests learning plumbing, not general capability."""

import argparse
import itertools
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rng = random.Random(31)
    scenes = list(
        itertools.product(
            ["cat", "dog", "bird", "rabbit"],
            ["red", "blue", "green", "yellow"],
            ["table", "sofa", "window"],
            [1, 2],
        )
    )
    rng.shuffle(scenes)
    args.output.mkdir(parents=True, exist_ok=True)
    for split, subset in [
        ("train", scenes[:64]),
        ("validation", scenes[64:80]),
        ("test", scenes[80:]),
    ]:
        rows = []
        for animal, color, place, count in subset:
            animals = ["cat", "dog", "bird", "rabbit"]
            colors = ["red", "blue", "green", "yellow"]
            rng.shuffle(animals)
            rng.shuffle(colors)
            rows.append(
                {
                    "id": f"scene-{animal}-{color}-{place}-{count}",
                    "state": {
                        "text": f"Near the {place} there {'is one ' + animal if count == 1 else 'are two ' + animal + 's'}. A {color} box is beside them."
                    },
                    "questions": {
                        "animal": {
                            "type": "choice",
                            "instructions": "Which kind of animal is present?",
                            "criteria": {value: "A " + value for value in animals},
                        },
                        "color": {
                            "type": "choice",
                            "instructions": "What color is the box?",
                            "criteria": {value: value.capitalize() for value in colors},
                        },
                        "count": {
                            "type": "score",
                            "instructions": "How many animals are present?",
                            "criteria": ["One animal", "Two animals"],
                        },
                        "cat": {"type": "noul", "instructions": "Is a cat present?"},
                    },
                    "targets": {
                        "animal": animal,
                        "color": color,
                        "count": count - 1,
                        "cat": animal == "cat",
                    },
                }
            )
        (args.output / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    main()
