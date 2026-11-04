#!/usr/bin/env python3
"""Exact checkpoint MLP inference without starting TensorFlow per SITL trial.

Only the known 12->64->64->4 tanh architecture is accepted. Weights remain
float32 and unchanged. Validate against PPO2 before using in a new batch.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import socket
import zipfile
import numpy as np


class NumpyPolicy:
    def __init__(self,path):
        self.path=Path(path);self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        with zipfile.ZipFile(str(self.path)) as archive:
            with np.load(io.BytesIO(archive.read('parameters')),allow_pickle=False) as arrays:
                keys=('pi_fc0','pi_fc1','pi');shapes=((12,64),(64,64),(64,4))
                self.layers=[]
                for key,shape in zip(keys,shapes):
                    w=arrays['model/'+key+'/w:0'].copy();b=arrays['model/'+key+'/b:0'].copy()
                    if w.shape!=shape or b.shape!=(shape[1],) or w.dtype!=np.float32 or b.dtype!=np.float32:
                        raise ValueError('Unexpected checkpoint architecture or dtype')
                    self.layers.append((w,b))

    def predict(self,batch,deterministic=True):
        if not deterministic:raise ValueError('Only deterministic inference is supported')
        x=np.asarray(batch,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=12 or not np.isfinite(x).all():raise ValueError('Expected finite [N,12] observation')
        for w,b in self.layers:x=np.tanh(np.matmul(x,w)+b)
        return x,None


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=5801)
    args=parser.parse_args();model=NumpyPolicy(args.model)
    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sock.bind((args.host,args.port))
    print('numpy_policy_ready '+json.dumps(dict(model=str(args.model),sha256=model.sha256,port=args.port)),flush=True)
    try:
        while True:
            payload,client=sock.recvfrom(65535)
            seq=None
            try:
                request=json.loads(payload.decode());seq=request.get('seq')
                if request.get('type','infer')=='ping':response=dict(ok=True,type='pong',seq=seq)
                elif request.get('type','infer')=='infer':
                    action,_=model.predict(np.asarray(request['obs'],dtype=np.float32).reshape(1,12))
                    response=dict(ok=True,type='action',seq=seq,action=action[0].tolist())
                else:raise ValueError('Unknown request type')
            except (ValueError,KeyError,TypeError,AttributeError) as exc:response=dict(ok=False,seq=seq,error=str(exc))
            sock.sendto(json.dumps(response,separators=(',',':')).encode(),client)
    except KeyboardInterrupt:pass
    finally:sock.close()


if __name__=='__main__':main()
