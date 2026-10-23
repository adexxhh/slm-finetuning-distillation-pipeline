"""Build notebook source archives from clean sibling Git checkouts; standard library only."""
from pathlib import Path
import argparse,json,subprocess,zipfile

def git(root,*args):
    return subprocess.check_output(["git",*args],cwd=root,text=True).strip()

def build(base,output):
    output.mkdir(parents=True,exist_ok=True)
    configs={
      "sql-8b-source.zip":[("slm-finetuning-distillation-pipeline","sql-8b-source"),
                           ("autonomous-code-editor","sql-8b-source/repair-editor")],
      "remaining-projects-source.zip":[("mcp-database-auditor","remaining-projects/mcp-database-auditor"),
                                       ("autonomous-code-editor","remaining-projects/autonomous-code-editor")]}
    for filename,projects in configs.items():
        versions={}
        for name,_ in projects:
            root=base/name
            if not (root/".git").exists():raise RuntimeError("Missing sibling checkout: "+str(root))
            if git(root,"status","--porcelain"):raise RuntimeError("Commit/stash changes before packaging: "+name)
            versions[name]=git(root,"rev-parse","HEAD")
        with zipfile.ZipFile(output/filename,"w",zipfile.ZIP_DEFLATED) as archive:
            for name,prefix in projects:
                root=base/name
                for relative in git(root,"ls-files").splitlines():
                    p=root/relative;parts=Path(relative).parts
                    if not p.is_file():continue
                    if any(x in {".git","artifacts","checkpoints",".venv","__pycache__", ".ipynb_checkpoints"} for x in parts):continue
                    if p.name in {".env",".env.local","credentials.json","token.json"} or p.suffix in {".pyc",".pem",".key",".pt",".pth",".safetensors"}:continue
                    if p.stat().st_size>10*1024*1024:raise RuntimeError("Unexpected large source file: "+relative)
                    archive.write(p,prefix+"/"+relative)
            archive.writestr("source_versions.json",json.dumps(versions,indent=2))
        print(output/filename,versions)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",required=True,type=Path)
    args=parser.parse_args()
    build(Path(__file__).resolve().parents[2],args.output_dir.resolve())
