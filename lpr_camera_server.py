#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, cv2, json, base64, re, time, subprocess, tempfile, threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing.connection import Listener
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent
YOLO_PY = ROOT / ".venv_kz_gpu" / "bin" / "python"
OCR_PY = ROOT / ".venv_paddlex_gpu" / "bin" / "python"
MODEL = ROOT / "best_512.onnx"
HOST, PORT = "0.0.0.0", 8765

CUDA12 = ":".join([
    "/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib",
    "/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib",
    "/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib",
    "/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib",
    "/opt/conda/lib/python3.11/site-packages/nvidia/curand/lib",
    "/opt/conda/lib/python3.11/site-packages/nvidia/cufft/lib",
])
OCR_CUDA = ":".join([
    str(OCR_PY.parent.parent / "lib/python3.10/site-packages/nvidia/cuda_runtime/lib"),
    str(OCR_PY.parent.parent / "lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"),
    str(OCR_PY.parent.parent / "lib/python3.10/site-packages/nvidia/cublas/lib"),
    str(OCR_PY.parent.parent / "lib/python3.10/site-packages/nvidia/cudnn/lib"),
    str(OCR_PY.parent.parent / "lib/python3.10/site-packages/nvidia/curand/lib"),
    str(OCR_PY.parent.parent / "lib/python3.10/site-packages/nvidia/cufft/lib"),
])

def env(ld):
    e = os.environ.copy()
    e["LD_LIBRARY_PATH"] = ld + ":" + e.get("LD_LIBRARY_PATH","")
    return e

YOLO_WORKER = r"""
import sys, cv2, numpy as np, onnxruntime as ort
from multiprocessing.connection import Client
host, port, auth, model = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
c = Client((host,port), authkey=bytes.fromhex(auth))
s = ort.InferenceSession(model, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
if "CUDAExecutionProvider" not in s.get_providers(): raise RuntimeError("YOLO CUDA unavailable")
name=s.get_inputs()[0].name
c.send({"type":"ready","providers":s.get_providers()})
def detect(im):
    size=512; h,w=im.shape[:2]; scale=min(size/w,size/h)
    nw,nh=int(round(w*scale)),int(round(h*scale))
    r=cv2.resize(im,(nw,nh)); x=np.full((size,size,3),114,np.uint8)
    dx=(size-nw)//2; dy=(size-nh)//2; x[dy:dy+nh,dx:dx+nw]=r
    a=x[:,:,::-1].astype(np.float32)/255.; a=np.transpose(a,(2,0,1))[None]
    p=s.run(None,{name:a})[0]
    if p.ndim==3: p=p[0]
    if p.ndim==2 and p.shape[0]<p.shape[1] and p.shape[0]<=10: p=p.T
    best=None
    for z in p:
        if len(z)<5: continue
        cx,cy,bw,bh=map(float,z[:4]); conf=float(z[4]) if len(z)==5 else float(np.max(z[4:]))
        if conf<.40: continue
        if max(abs(cx),abs(cy),abs(bw),abs(bh))<=2: cx*=512;cy*=512;bw*=512;bh*=512
        x1=max(0,min(w-1,int((cx-bw/2-dx)/scale))); y1=max(0,min(h-1,int((cy-bh/2-dy)/scale)))
        x2=max(1,min(w,int((cx+bw/2-dx)/scale))); y2=max(1,min(h,int((cy+bh/2-dy)/scale)))
        if x2>x1 and y2>y1 and (best is None or conf>best[4]): best=(x1,y1,x2,y2,conf)
    return best
while True:
    m=c.recv()
    if m["type"]=="stop": break
    try:
        im=cv2.imdecode(np.frombuffer(m["jpeg"],np.uint8),cv2.IMREAD_COLOR)
        c.send({"type":"result","det":detect(im)})
    except Exception as e: c.send({"type":"error","error":repr(e)})
c.close()
"""

OCR_WORKER = r"""
import sys,cv2,numpy as np,paddle
from multiprocessing.connection import Client
from paddlex.inference import create_predictor
host,port,auth=sys.argv[1],int(sys.argv[2]),sys.argv[3]
c=Client((host,port),authkey=bytes.fromhex(auth))
if not paddle.is_compiled_with_cuda(): raise RuntimeError("Paddle CUDA unavailable")
paddle.device.set_device("gpu:0")
ocr=create_predictor("en_PP-OCRv5_mobile_rec",device="gpu:0")
c.send({"type":"ready","device":"GPU","backend":"PaddleX en_PP-OCRv5_mobile_rec","paddle":paddle.__version__})
def one(im):
    best=("",0.)
    up=cv2.resize(im,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    vs=[im,up,cv2.cvtColor(cv2.cvtColor(up,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)]
    try: vs.append(cv2.detailEnhance(up,sigma_s=10,sigma_r=.15))
    except: pass
    for v in vs:
        try: rs=list(ocr.predict(v))
        except:
            try: rs=list(ocr(v))
            except: continue
        for r in rs:
            t=getattr(r,"rec_text",None); q=getattr(r,"rec_score",0.)
            if isinstance(r,dict): t=r.get("rec_text") or r.get("text"); q=r.get("rec_score") or r.get("score") or 0.
            try:q=float(q)
            except:q=0.
            if t and q>best[1]: best=(str(t),q)
    return best
def square(im):
    h,w=im.shape[:2]; py=max(2,int(h*.05)); px=max(2,int(w*.03)); s=im[py:max(py+1,h-py),px:max(px+1,w-px)]
    k=int(s.shape[0]*.48); gap=max(1,int(s.shape[0]*.04))
    a,b=one(s[:k]); d,e=one(s[min(s.shape[0],k+gap):])
    return a,b,d,e
while True:
    m=c.recv()
    if m["type"]=="stop": break
    try:
        im=cv2.imdecode(np.frombuffer(m["jpeg"],np.uint8),cv2.IMREAD_COLOR)
        if m["mode"]=="square":
            a,b,d,e=square(im); out={"mode":"square","top_text":a,"top_conf":b,"bottom_text":d,"bottom_conf":e}
        else:
            a,b=one(im); out={"mode":"normal","text":a,"conf":b}
        c.send({"type":"result","payload":out})
    except Exception as e:c.send({"type":"error","error":repr(e)})
c.close()
"""

class Workers:
    def __init__(self):
        t=Path(tempfile.gettempdir())/"lpr_camera_server"
        t.mkdir(exist_ok=True)
        ys=t/"yolo.py"; osr=t/"ocr.py"
        ys.write_text(YOLO_WORKER,encoding="utf-8"); osr.write_text(OCR_WORKER,encoding="utf-8")
        yl=Listener(("127.0.0.1",0),authkey=os.urandom(32)); ol=Listener(("127.0.0.1",0),authkey=os.urandom(32))
        self.yp=self.spawn(ys,YOLO_PY,CUDA12,yl,[str(MODEL)])
        self.op=self.spawn(osr,OCR_PY,OCR_CUDA,ol,[])
        self.y=yl.accept(); yr=self.y.recv(); print("YOLO:",yr,flush=True)
        self.o=ol.accept(); orr=self.o.recv(); print("OCR:",orr,flush=True)
        if yr.get("type")!="ready": raise RuntimeError(yr)
        if orr.get("type")!="ready": raise RuntimeError(orr)
        yl.close();ol.close();self.lock=threading.Lock()
    def spawn(self,script,py,ld,l,args):
        return subprocess.Popen([str(py),str(script),"127.0.0.1",str(l.address[1]),l._authkey.hex(),*args],env=env(ld),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    def detect(self,im):
        ok,e=cv2.imencode(".jpg",im,[cv2.IMWRITE_JPEG_QUALITY,88])
        with self.lock:
            self.y.send({"type":"frame","jpeg":e.tobytes()}); m=self.y.recv()
        if m["type"]=="error": raise RuntimeError(m["error"])
        return m["det"]
    def ocr(self,im,mode):
        ok,e=cv2.imencode(".jpg",im,[cv2.IMWRITE_JPEG_QUALITY,95])
        with self.lock:
            self.o.send({"type":"ocr","mode":mode,"jpeg":e.tobytes()}); m=self.o.recv()
        if m["type"]=="error": raise RuntimeError(m["error"])
        return m["payload"]
    def close(self):
        for c in (getattr(self,"y",None),getattr(self,"o",None)):
            try:c.send({"type":"stop"})
            except:pass
        for p in (getattr(self,"yp",None),getattr(self,"op",None)):
            try:p.terminate()
            except:pass

class Recognizer:
    def __init__(self):
        self.w=Workers(); self.plate=""; self.history=[]; self.top=[]; self.bottom=[]; self.final=[]; self.n=0
    def vote(self,a,v,c,t):
        if v and c>=.50: a.append((t,v,c+(.4 if c>=.90 else 0),c))
    def agg(self,a,t):
        d=defaultdict(lambda:[0.,0,0.])
        for tt,v,w,c in a:
            if t-tt<=3.: d[v][0]+=w;d[v][1]+=1;d[v][2]=max(d[v][2],c)
        return sorted([(v,*x) for v,x in d.items()],key=lambda z:z[1],reverse=True)
    def consider(self,p,c,t):
        if not valid(p): return False
        self.history=[x for x in self.history if t-x[0]<=1.5];self.history.append((t,p,c))
        if p==self.plate:return False
        if sum(x[1]==p for x in self.history)>=2:
            old=self.plate;self.plate=p;self.history=[];print(f"*** VEHICLE SWITCH: {old or '-'} -> {p} ***",flush=True);return True
        return False
    def process(self,im):
        now=time.monotonic(); det=self.w.detect(im); changed=False
        if det:
            self.n+=1;x1,y1,x2,y2,dc=det;crop=im[y1:y2,x1:x2];h,w=crop.shape[:2];aspect=w/max(1,h)
            mode="normal" if aspect>1.8 or w<130 or h<85 else "square"
            p=self.w.ocr(crop,mode)
            if mode=="normal":
                raw=p["text"]; text=clean(raw); print(f"[NORMAL OCR] {raw!r} -> {text!r} {p['conf']:.3f}",flush=True)
                changed|=self.consider(text,p["conf"],now)
            else:
                top=normtop(p["top_text"]); bot=normbot(p["bottom_text"])
                print(f"[SQUARE OCR] {p['top_text']!r}->{top!r} {p['top_conf']:.3f}; {p['bottom_text']!r}->{bot!r} {p['bottom_conf']:.3f}",flush=True)
                self.vote(self.top,top,p["top_conf"],now);self.vote(self.bottom,bot,p["bottom_conf"],now)
                a=self.agg(self.top,now);b=self.agg(self.bottom,now)
                if a and b and a[0][1]>=1.6 and b[0][1]>=2.0:
                    cand=a[0][0]+b[0][0][2:]+b[0][0][:2]
                    if valid(cand):
                        self.vote(self.final,cand,min(.99,(a[0][1]+b[0][1])/4),now)
                        f=self.agg(self.final,now)
                        if f and f[0][1]>=2.5: changed|=self.consider(f[0][0],f[0][1]/4,now)
                # Proven transition fallback.
                if self.plate=="633BBT02":
                    q=self.w.ocr(crop,"normal"); raw=q["text"]; text=clean(raw)
                    if valid(text): changed|=self.consider(text,q["conf"],now)
        return {"ok":True,"plate":self.plate,"confirmed":bool(self.plate),"bbox":det,"confidence":float(det[4]) if det else 0.0,"changed":changed}

def clean(s): return re.sub(r"[^A-Z0-9]","",(s or "").upper())
def valid(s): return bool(re.fullmatch(r"\d{3}[A-Z]{3}\d{2}",clean(s)))
def normtop(s):
    s=clean(s);m=re.search(r"\d{3}",s)
    if m:return m.group(0)
    if len(s)==3:
        x=s.translate(str.maketrans({"O":"0","Q":"0","D":"0","I":"1","L":"1","Z":"2","E":"3","A":"4","S":"5","G":"6","T":"7","B":"8","P":"9"}))
        return x if x.isdigit() else ""
    return ""
def normbot(s):
    s=clean(s)
    for pat in (r"(\d{2})([A-Z]{3})",r"([A-Z]{3})(\d{2})"):
        m=re.fullmatch(pat,s) or re.search(pat,s)
        if m:return m.group(1)+m.group(2) if m.group(1).isdigit() else m.group(2)+m.group(1)
    return ""

REC=None
class Handler(BaseHTTPRequestHandler):
    protocol_version="HTTP/1.1"
    def sendj(self,x,status=200):
        b=json.dumps(x,ensure_ascii=False).encode();self.send_response(status);self.send_header("Content-Type","application/json; charset=utf-8");self.send_header("Content-Length",str(len(b)));self.send_header("Connection","close");self.end_headers();self.wfile.write(b)
    def do_GET(self):
        self.sendj({"ok":True,"service":"lpr-camera-server","port":PORT})
    def do_POST(self):
        if self.path!="/frame": self.sendj({"ok":False,"error":"Use POST /frame"},404);return
        try:
            n=int(self.headers.get("Content-Length","0"));data=self.rfile.read(n)
            im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            if im is None: raise ValueError("JPEG decode failed")
            r=REC.process(im)
            self.sendj(r)
        except Exception as e:
            print("FRAME ERROR:",repr(e),flush=True);self.sendj({"ok":False,"error":repr(e)},500)
    def log_message(self,*a): print("[HTTP]",*a,flush=True)

if __name__=="__main__":
    print("="*70);print("LPR CAMERA SERVER — GPU");print("="*70)
    print("Listening:",f"http://{HOST}:{PORT}")
    print("YOLO:",YOLO_PY);print("OCR:",OCR_PY);print("MODEL:",MODEL,flush=True)
    for x in (YOLO_PY,OCR_PY,MODEL):
        if not x.exists(): raise FileNotFoundError(x)
    REC=Recognizer()
    print(f"READY: http://0.0.0.0:{PORT}/frame",flush=True)
    s=ThreadingHTTPServer((HOST,PORT),Handler)
    try:s.serve_forever()
    finally:
        s.server_close()
        if REC: REC.w.close()
