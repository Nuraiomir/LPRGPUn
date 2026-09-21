"""
YOLO GPU detection worker (subprocess), extracted verbatim from the recognition
logic validated in lpr_v19_universal.py (three-video smoke test: 545BDR05,
979CBB02, 049BXS02 and their full sequences, all confirmed correctly).

Not a rewrite: this is the exact byte-for-byte YOLO_WORKER source that v19
writes to a temp file and spawns as a subprocess, now saved as a real,
importable/inspectable project file instead of an inline string literal.
Both the offline evaluator and the API server spawn THIS file, so there is
one single YOLO worker implementation instead of two slowly-diverging copies.

Protocol (multiprocessing.connection, authkey-secured):
  argv: host, port, authkey_hex, onnx_model_path
  -> sends {"type": "ready", "providers": [...]} once the ONNX session is up
  recv {"type": "frame", "fid": <any>, "jpeg": <bytes>}
    -> send {"type": "result", "fid": <same>, "det": (x1,y1,x2,y2,conf) or None,
              "ms": <float, worker-side detect() time>}
    -> or {"type": "error", "fid": <same>, "error": repr(exc)}
  recv {"type": "stop"} -> exits cleanly
"""
import sys
import time
from multiprocessing.connection import Client
import cv2
import numpy as np
import onnxruntime as ort

host, port, auth_hex, model = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
conn = Client((host, port), authkey=bytes.fromhex(auth_hex))

session = ort.InferenceSession(
    model,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
conn.send({"type": "ready", "providers": session.get_providers()})
inp = session.get_inputs()[0].name

def letterbox(im, size=512):
    h,w=im.shape[:2]
    scale=min(size/w,size/h)
    nw,nh=int(round(w*scale)),int(round(h*scale))
    r=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR)
    canvas=np.full((size,size,3),114,np.uint8)
    dx=(size-nw)//2; dy=(size-nh)//2
    canvas[dy:dy+nh,dx:dx+nw]=r
    return canvas,scale,dx,dy

def detect(im):
    x,scale,dx,dy=letterbox(im)
    a=x[:,:,::-1].astype(np.float32)/255.0
    a=np.transpose(a,(2,0,1))[None]
    pred=session.run(None,{inp:a})[0]
    if pred.ndim==3: pred=pred[0]
    if pred.ndim==2 and pred.shape[0]<pred.shape[1] and pred.shape[0]<=10:
        pred=pred.T
    best=None
    fh,fw=im.shape[:2]
    for row in pred:
        if len(row)<5: continue
        cx,cy,bw,bh=map(float,row[:4])
        conf=float(row[4]) if len(row)==5 else float(np.max(row[4:]))
        if conf<0.40: continue
        if max(abs(cx),abs(cy),abs(bw),abs(bh))<=2:
            cx*=512; cy*=512; bw*=512; bh*=512
        x1=max(0,min(fw-1,int((cx-bw/2-dx)/scale)))
        y1=max(0,min(fh-1,int((cy-bh/2-dy)/scale)))
        x2=max(1,min(fw,int((cx+bw/2-dx)/scale)))
        y2=max(1,min(fh,int((cy+bh/2-dy)/scale)))
        if x2<=x1 or y2<=y1: continue
        if best is None or conf>best[4]:
            best=(x1,y1,x2,y2,conf)
    return best

while True:
    msg=conn.recv()
    if msg["type"]=="frame":
        fid=msg["fid"]
        arr=np.frombuffer(msg["jpeg"],np.uint8)
        im=cv2.imdecode(arr,cv2.IMREAD_COLOR)
        t=time.perf_counter()
        try:
            det=detect(im)
            conn.send({"type":"result","fid":fid,"det":det,
                       "ms":(time.perf_counter()-t)*1000.0})
        except Exception as e:
            conn.send({"type":"error","fid":fid,"error":repr(e)})
    elif msg["type"]=="stop":
        break
conn.close()
