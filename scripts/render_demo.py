"""Render captured verification output as PNG/GIF/SVG; these are condensed terminal replays."""
from pathlib import Path
import subprocess,sys,json,re,textwrap,html,os
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
TITLE=ROOT.name
def render():
    if TITLE=="mcp-database-auditor":
        commands=[["-m","scripts.demo_scenario"],["-m","pytest","-q","--disable-warnings"]]
    elif TITLE=="autonomous-code-editor":
        commands=[["-m","scripts.demo_scenario"],["-m","pytest","-q","--disable-warnings"]]
    else:
        commands=[["-m","training.smoke_cpu"],["-m","pytest","-q","--disable-warnings"]]
    captured=[]
    for args in commands:
        result=subprocess.run([sys.executable,*args],cwd=ROOT,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=90)
        captured.append({"command":"python "+" ".join(args),"exit_code":result.returncode,"stdout":result.stdout,"stderr":result.stderr})
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    target=ROOT/"assets";target.mkdir(exist_ok=True)
    (target/"demo-transcript.json").write_text(json.dumps({"kind":"condensed terminal replay from real runs","commands":captured},indent=2),encoding="utf-8")
    lines=[]
    for item in captured:
        lines+=["$ "+item["command"]]
        output=re.sub(r"\x1b\[[0-9;]*[A-Za-z]","",item["stdout"])
        if TITLE=="slm-finetuning-distillation-pipeline":
            keep=[x for x in output.splitlines() if ("loss" in x or "CPU" in x or "passed" in x or "8b" in x or "quality" in x or "generated_tokens" in x or "scope" in x or "status" in x)]
        else:keep=[x for x in output.splitlines() if x.strip()]
        for line in keep:
            lines.extend(textwrap.wrap(line,width=100,replace_whitespace=False) or [""])
        lines+=[""]
    lines=lines[:32]
    fonts=[Path("C:/Windows/Fonts/consola.ttf"),Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")]
    font=next((ImageFont.truetype(str(p),18) for p in fonts if p.exists()),ImageFont.load_default())
    def frame(count):
        im=Image.new("RGB",(1280,860),"#0b1220");draw=ImageDraw.Draw(im)
        draw.rounded_rectangle((20,20,1260,840),radius=18,fill="#111c30",outline="#30425f",width=2)
        for x,c in [(48,"#fa6c6c"),(72,"#f4c768"),(96,"#5bd69a")]:draw.ellipse((x,44,x+13,57),fill=c)
        draw.text((125,40),TITLE,font=font,fill="#e5eefc")
        draw.text((45,80),"Actual CPU verification | condensed terminal replay",font=font,fill="#8aa5c7")
        y=125
        for line in lines[:count]:
            draw.text((45,y),line,font=font,fill="#5ee2b0" if line.startswith("$") else "#d5e3f5");y+=20
        draw.text((45,807),"Scope and limitations are documented in README.md. No simulated model scores.",font=font,fill="#8aa5c7")
        return im
    last=frame(len(lines));last.save(target/"demo.png")
    frames=[frame(i) for i in range(1,len(lines)+1)]
    frames[0].save(target/"demo.gif",save_all=True,append_images=frames[1:],
                   duration=[140]*(len(frames)-1)+[2500],loop=0,optimize=True)
    elements=[]
    for i,line in enumerate(lines):
        elements.append('<text x="35" y="'+str(90+20*i)+'">'+html.escape(line)+'</text>')
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="860"><rect width="100%" height="100%" fill="#0b1220"/><g fill="#d5e3f5" font-family="monospace" font-size="17"><text x="35" y="40">'+html.escape(TITLE)+' | actual verification replay</text>'+''.join(elements)+'</g></svg>'
    (target/"demo.svg").write_text(svg,encoding="utf-8")
    print("Created",target/"demo.png",target/"demo.gif")
if __name__=="__main__":render()
