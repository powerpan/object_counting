"""Local-only frontend cache for an authorized SSH Gradio demonstration.

No model calls are added. Dynamic requests pass to the existing loopback tunnel;
Gradio image outputs are read byte-for-byte over a fresh bounded SSH connection.
"""
import argparse,http.client,http.server,subprocess,shlex,zipfile,mimetypes,hashlib,json,re
from pathlib import Path
from urllib.parse import urlsplit,unquote


def main(a):
    z=zipfile.ZipFile(a.wheel);names=set(z.namelist());cache=Path(a.cache);cache.mkdir(parents=True,exist_ok=True)
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def bytes(self,data,mime):
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):self.route()
        def do_POST(self):self.route()
        def route(self):
            try:
                path=unquote(urlsplit(self.path).path)
                member='gradio/templates/frontend'+path
                if self.command=='GET' and path.startswith('/assets/') and member in names:
                    return self.bytes(z.read(member),mimetypes.guess_type(path)[0] or 'application/octet-stream')
                match=re.fullmatch(r'/gradio_api/file=(/tmp/gradio/[0-9a-f]{64}/[A-Za-z0-9_.-]+)',path)
                if self.command=='GET' and match:
                    remote=match.group(1);key=hashlib.sha256(remote.encode()).hexdigest();f=cache/key
                    if not f.exists():
                        cmd='cat -- '+shlex.quote(remote)
                        p=subprocess.run(['ssh','-C','-F',a.ssh_config,a.ssh_alias,cmd],capture_output=True,timeout=60)
                        if p.returncode!=0:raise RuntimeError('Image transfer failed')
                        f.write_bytes(p.stdout)
                        (cache/(key+'.json')).write_text(json.dumps({'bytes':len(p.stdout),'sha256':hashlib.sha256(p.stdout).hexdigest()}))
                    return self.bytes(f.read_bytes(),mimetypes.guess_type(remote)[0] or 'application/octet-stream')
                c=http.client.HTTPConnection('127.0.0.1',a.upstream,timeout=90)
                body=self.rfile.read(int(self.headers.get('Content-Length','0'))) if self.command=='POST' else None
                headers={k:v for k,v in self.headers.items() if k.lower() not in ['host','connection','accept-encoding','transfer-encoding']}
                headers['Host']='127.0.0.1:'+str(a.port)
                c.request(self.command,self.path,body,headers);r=c.getresponse();self.send_response(r.status)
                for k,v in r.getheaders():
                    if k.lower() not in ['connection','transfer-encoding','content-length']:self.send_header(k,v)
                self.send_header('Connection','close');self.end_headers();self.close_connection=True
                while True:
                    block=r.read1(65536)
                    if not block:break
                    self.wfile.write(block);self.wfile.flush()
                c.close()
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception:
                self.close_connection=True
                try:self.send_error(502,'Demonstration transport unavailable')
                except OSError:pass
    http.server.ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=7863);p.add_argument('--upstream',type=int,required=True)
    p.add_argument('--wheel',required=True);p.add_argument('--cache',required=True);p.add_argument('--ssh-config',required=True);p.add_argument('--ssh-alias',required=True)
    main(p.parse_args())
