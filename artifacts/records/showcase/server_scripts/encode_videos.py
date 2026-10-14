"""Encode native 400x400 PNG frames at 30 fps using local PyAV/libx264."""
import json,sys
from pathlib import Path
from fractions import Fraction
import av,numpy as np
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parents[1];V=D/'videos';records=[]
command='PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python '+str(Path(__file__))
fontpaths=list(Path('/usr/share/fonts').rglob('DejaVuSans.ttf'));font=ImageFont.truetype(str(fontpaths[0]),18) if fontpaths else ImageFont.load_default()
def encode(name,images,sources):
 p=V/(name+'.mp4');container=av.open(str(p),'w');s=container.add_stream('libx264',rate=30);s.width=images[0].width;s.height=images[0].height;s.pix_fmt='yuv420p';s.options={'crf':'18','preset':'medium'}
 for i,im in enumerate(images):
  frame=av.VideoFrame.from_image(im);frame.pts=i;frame.time_base=Fraction(1,30)
  for packet in s.encode(frame):container.mux(packet)
 for packet in s.encode():container.mux(packet)
 container.close()
 with av.open(str(p)) as check:
  stream=check.streams.video[0];n=sum(1 for f in check.decode(video=0));assert n==200 and stream.codec_context.name=='h264' and stream.pix_fmt=='yuv420p';info=dict(frames=n,width=stream.width,height=stream.height,fps=str(stream.average_rate),codec=stream.codec_context.name,pix_fmt=stream.pix_fmt)
 records.append(dict(file=str(p.relative_to(D)),kind='video',sources=sources,crf=18,fps=30,validation=info,generation_command=command))
 if name=='side_by_side_T5':return
 gif=V/(name+'.gif');small=[]
 for i in range(0,200,3):
  im=images[i].copy();im.thumbnail((400,400));small.append(im.convert('P',palette=Image.ADAPTIVE,colors=128))
 small[0].save(gif,save_all=True,append_images=small[1:],duration=[100]*66+[67],loop=0,optimize=True,disposal=2)
 assert gif.stat().st_size<8_000_000
 records.append(dict(file=str(gif.relative_to(D)),kind='gif',sources=sources,width=small[0].width,frames=len(small),bytes=gif.stat().st_size,sampled_indices=list(range(0,200,3)),generation_command=command))
 print(name,info,'GIF bytes',gif.stat().st_size,flush=True)
loaded={}
for name in ['SV100-GT_fixed_raw','SV100-GT_naive_warmup_aligned','SV100-T5_fixed_raw','SV100-T5_naive_warmup_raw']:
 paths=sorted((V/'frames'/name).glob('view_*.png'));assert len(paths)==200
 images=[Image.open(p).convert('RGB') for p in paths];assert all(im.size==(400,400) for im in images)
 encode(name,images,[str(p.relative_to(D)) for p in paths])
 if name.startswith('SV100-T5'):loaded[name]=images
sides=[]
for left,right in zip(loaded['SV100-T5_fixed_raw'],loaded['SV100-T5_naive_warmup_raw']):
 im=Image.new('RGB',(800,432),'white');im.paste(left,(0,32));im.paste(right,(400,32));draw=ImageDraw.Draw(im);draw.rectangle((0,0,799,31),fill='#20252b');draw.text((12,5),'T5 / fixed (raw)',font=font,fill='white');draw.text((412,5),'T5 / naive_warmup (raw)',font=font,fill='white');sides.append(im)
encode('side_by_side_T5',sides,['videos/SV100-T5_fixed_raw.mp4','videos/SV100-T5_naive_warmup_raw.mp4'])
(V/'encoding_summary.json').write_text(json.dumps(records,indent=2)+'\n')
