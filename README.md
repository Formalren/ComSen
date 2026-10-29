## Provided materials

| Location | Contents |
| --- | --- |
| `data_engine/` | Task-generation utilities and supporting agent code. |
| `data_engine/taskgenerate/` | Scene metadata and available initial-pose files for kitchens, living rooms, bedrooms, and bathrooms. |
| `evaluate/` | Selected evaluation utilities and agent wrappers. |
| `data/` | A placeholder directory; |

Scene folders contain `metadata.json` and, where available, `originPos.json`.
The metadata describe simulator objects and scene state; initial-pose files
describe position, rotation, and camera horizon. These files are not
source--follow-up test pairs or experimental outcome records.

## Run the available data checks

The following commands require **Python 3.10 or later** and use only its
standard library. No simulator, GPU, model API, or API key is needed.
Run them from the repository root, the directory containing `data_engine/`.
If your installation uses `python3`, substitute it for `python`.

Validate the syntax of all released scene JSON files:

```sh
python -c "import json; from pathlib import Path; files = sorted(Path('data_engine/taskgenerate').rglob('*.json')); assert files, 'Run from the repository root'; [json.loads(p.read_text(encoding='utf-8-sig')) for p in files]; print('Valid JSON files:', len(files))"
```

Inspect one included initial-pose record:

```sh
python -m json.tool data_engine/taskgenerate/bathrooms/FloorPlan401/originPos.json
```

These commands check or display the released data only. They do **not** run
an agent, validate task feasibility, detect MR violations, or reproduce
experimental metrics.

