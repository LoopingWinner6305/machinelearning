"""Write the fixed protocol heart-disease experiment to reports/metrics.json."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ml_fundamentals.heart import evaluate_heart
result,_,_,_=evaluate_heart(ROOT/'Decision_trees'/'processed.cleveland.data')
output=ROOT/'reports'/'metrics.json';output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
