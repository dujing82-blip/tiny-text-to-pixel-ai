import os, csv
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path("tiny_pixel_dataset"); IMG=ROOT/"images"; IMG.mkdir(parents=True,exist_ok=True)
COLORS={"red":(230,55,55),"blue":(55,105,230),"green":(55,180,90),"yellow":(235,200,45)}
OBJECTS=["heart","robot","spaceship","tree","house"]; SIZES=["small","big"]
HELD={("blue","big","spaceship"),("yellow","small","robot"),("green","big","heart"),("red","small","tree"),("blue","small","house")}
def put(a,x,y,c):
    if 0<=x<16 and 0<=y<16:a[y,x]=c
def draw(obj,c,size,rng):
    a=np.zeros((16,16,3),np.uint8); dx=int(rng.integers(-1,2));dy=int(rng.integers(-1,2))
    if obj=="heart":
        b=np.array([[0,1,1,0,1,1,0],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[0,1,1,1,1,1,0],[0,0,1,1,1,0,0],[0,0,0,1,0,0,0],[0,0,0,0,0,0,0]],np.uint8)
        h=b if size=="small" else np.kron(b[:6,:6],np.ones((2,2),np.uint8)); H,W=h.shape;x0=(16-W)//2+dx;y0=(16-H)//2+dy
        for y in range(H):
            for x in range(W):
                if h[y,x]:put(a,x0+x,y0+y,c)
    elif obj=="robot":
        x0,y0,w,h=(5+dx,4+dy,6,7) if size=="small" else (3+dx,2+dy,10,11)
        for y in range(y0,y0+h):
            for x in range(x0,x0+w):
                if y in (y0,y0+h-1) or x in (x0,x0+w-1):put(a,x,y,c)
        put(a,x0+2,y0+2,(255,255,255));put(a,x0+w-3,y0+2,(255,255,255));put(a,x0+w//2,y0-1,c);put(a,x0+1,y0+h,c);put(a,x0+w-2,y0+h,c)
    elif obj=="spaceship":
        cx=8+dx;top=(3 if size=="small" else 1)+dy;height=9 if size=="small" else 13
        for k in range(height):
            half=max(0,min(4 if size=="big" else 3,k//2,(height-1-k)//2+1))
            for x in range(cx-half,cx+half+1):put(a,x,top+k,c)
        put(a,cx,top+3,(255,255,255));put(a,cx-2,top+height,(255,130,20));put(a,cx+2,top+height,(255,130,20))
    elif obj=="tree":
        cx=8+dx;top=(3 if size=="small" else 1)+dy;crown=6 if size=="small" else 9
        for k in range(crown):
            half=min(k//2+1,4 if size=="big" else 3)
            for x in range(cx-half,cx+half+1):put(a,x,top+k,c)
        for y in range(top+crown,min(16,top+crown+4)):put(a,cx,y,(135,85,45));put(a,cx+1,y,(135,85,45))
    else:
        x0,y0,w,h=(5+dx,7+dy,6,5) if size=="small" else (3+dx,6+dy,10,7);cx=x0+w//2;rh=3 if size=="small" else 5
        for k in range(rh):
            half=min(w//2,k+1 if size=="small" else 2*k+1)
            for x in range(cx-half,cx+half+1):put(a,x,y0-rh+k,c)
        for y in range(y0,min(16,y0+h)):
            for x in range(x0,min(16,x0+w)):put(a,x,y,c)
        for y in range(y0+h-3,min(16,y0+h)):put(a,cx,y,(120,75,40))
        put(a,x0+1,y0+1,(255,255,255));put(a,x0+w-2,y0+1,(255,255,255))
    return a
rows=[];idx=0
for cn,c in COLORS.items():
  for size in SIZES:
    for obj in OBJECTS:
      split="unseen_test" if (cn,size,obj) in HELD else "train"; n=50 if split=="unseen_test" else 80
      for _ in range(n):
        fn=f"{idx:05d}.png";Image.fromarray(draw(obj,c,size,np.random.default_rng(100000+idx))).save(IMG/fn)
        rows.append([f"images/{fn}",f"{cn} {size} {obj}",cn,size,obj,split]);idx+=1
with open(ROOT/"captions.csv","w",newline="") as f:
    w=csv.writer(f);w.writerow(["filename","caption","color","size","object","split"]);w.writerows(rows)
print(f"Created {len(rows)} images: {sum(r[-1]=='train' for r in rows)} train + {sum(r[-1]=='unseen_test' for r in rows)} held-out.")
