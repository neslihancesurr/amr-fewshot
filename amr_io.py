"""Read/write AMR files in the '# ::id / # ::snt / graph' format."""
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AmrRecord:
    id: str
    snt: str
    graph: str


def read_amr_file(path):
    text = Path(path).read_text(encoding="utf-8")
    records = []
    for block in re.split(r"\n\s*\n", text.strip()):
        sid, snt, graph_lines = None, "", []
        for line in block.splitlines():
            if line.startswith("# ::id"):
                parts = line[len("# ::id"):].split()
                sid = parts[0] if parts else None
            elif line.startswith("# ::snt"):
                snt = line[len("# ::snt"):].strip()
            elif line.startswith("#"):
                continue
            else:
                graph_lines.append(line.rstrip())
        graph = "\n".join(graph_lines).strip()
        if sid and graph:
            records.append(AmrRecord(sid, snt, graph))
    return records


def write_amr_file(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"# ::id {r.id}\n# ::snt {r.snt}\n{r.graph}\n\n")
