import http.server
import json
import subprocess
import sys
import threading
import time
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get('APXINF_MODEL_DIR', os.path.join(ROOT_DIR, 'models', 'Qwen3.8-27B-AWQ-INT4'))
BINARY = os.environ.get('APXINF_BINARY', os.path.join(ROOT_DIR, 'target', 'release', 'apxinf'))
MAX_CONTEXT = 16640

class ApxInfHandler(http.server.BaseHTTPRequestHandler):
    model_proc = None
    model_lock = threading.Lock()
    
    @classmethod
    def ensure_model(cls):
        if cls.model_proc is not None and cls.model_proc.poll() is not None:
            cls.model_proc = None
        if cls.model_proc is None:
            with cls.model_lock:
                if cls.model_proc is None:
                    sys.stderr.write('Starting model process...\n')
                    cls.model_proc = subprocess.Popen(
                        [BINARY, 'serve', '--model', MODEL_DIR, '--device', 'cuda', '--dtype', 'bf16'],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, bufsize=1,
                    )
                    for _ in range(300):
                        line = cls.model_proc.stderr.readline()
                        if 'Ready' in line:
                            break
                        sys.stderr.write('  model: ' + line.strip() + '\n')
                    sys.stderr.write('Model ready!\n')
    
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok', 'evaluation_contract': 'apxinf.qwen38_27b.inference_interface.v1', 'model_revision': '63768c10df38c0395e12ef49edac1bd539eaeeea', 'max_model_len': MAX_CONTEXT, 'parallel_requests': 1, 'fallback_active': False, 'capabilities': {'pretokenized_input_ids': True, 'token_id_output': True, 'multimodal': False}}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def send_json_error(self, status, message, error_type='invalid_request'):
        payload = json.dumps({'error': {'type': error_type, 'message': message}}).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.path == '/v1/evaluations/generate':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                req = json.loads(body)
            except (json.JSONDecodeError, TypeError):
                self.send_json_error(400, 'request body must be valid JSON')
                return
            if not isinstance(req, dict):
                self.send_json_error(400, 'request body must be a JSON object')
                return
            if 'images' in req:
                self.send_json_error(400, 'multimodal input is unsupported', 'unsupported_capability')
                return
            input_ids = req.get('input_ids')
            if (not isinstance(input_ids, list) or not input_ids or
                    any(isinstance(x, bool) or not isinstance(x, int) or x < 0 or x >= 248320 for x in input_ids)):
                self.send_json_error(400, 'input_ids must contain valid vocabulary token IDs')
                return
            if req.get('temperature', 0.0) != 0.0:
                self.send_json_error(400, 'only temperature=0 is supported')
                return
            max_new_tokens = req.get('max_new_tokens', 128)
            if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or
                    max_new_tokens <= 0 or max_new_tokens >= MAX_CONTEXT):
                self.send_json_error(400, 'max_new_tokens is outside the supported range')
                return
            if len(input_ids) + max_new_tokens > MAX_CONTEXT:
                self.send_json_error(400, "request exceeds the supported context capacity", "capacity_exceeded")
                return

            rust_req = {
                'input_ids': input_ids,
                'max_new_tokens': max_new_tokens,
                'ignore_eos': req.get('ignore_eos', True),
            }
            request_id = req.get('request_id', req.get('id', 'req-a'))
            if not isinstance(request_id, str) or not request_id:
                request_id = 'req-a'
            stream = req.get('stream', True)

            self.ensure_model()

            try:
                self.model_proc.stdin.write(json.dumps(rust_req) + '\n')
                self.model_proc.stdin.flush()
                output_ids = []
                usage = {}
                if stream:
                    # Start the SSE response before waiting for generation so
                    # the evaluator observes true first-token latency.
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    self.wfile.flush()
                while True:
                    line = self.model_proc.stdout.readline()
                    if not line:
                        raise RuntimeError('model process closed stdout')
                    line = line.strip()
                    if line == 'data: [DONE]':
                        break
                    if not line.startswith('data: '):
                        continue
                    data = json.loads(line[6:])
                    if data.get('type') == 'token':
                        token_id = data['token_id']
                        index = len(output_ids)
                        output_ids.append(token_id)
                        if stream:
                            event = {'type': 'token', 'request_id': request_id, 'index': index, 'token_id': token_id}
                            self.wfile.write(('data: ' + json.dumps(event, separators=(',', ':')) + '\n\n').encode())
                            self.wfile.flush()
                    elif data.get('type') == 'done':
                        usage = data.get('usage', {})
                    elif data.get('type') == 'error':
                        raise RuntimeError(data.get('error', 'model generation failed'))
                if stream:
                    done = {'type': 'done', 'request_id': request_id, 'usage': usage}
                    self.wfile.write(('data: ' + json.dumps(done, separators=(',', ':')) + '\n\n').encode())
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                else:
                    result = {'type': 'result', 'request_id': request_id, 'output_ids': output_ids, 'usage': usage}
                    payload = json.dumps(result, separators=(',', ':')).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    self.wfile.flush()
            except Exception as e:
                sys.stderr.write('Error: ' + str(e) + '\n')
                self.send_json_error(500, str(e), 'backend_error')
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        ts = time.strftime('%H:%M:%S')
        sys.stderr.write('[' + ts + '] ' + str(args[0]) + '\n')

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    server = http.server.HTTPServer(('0.0.0.0', port), ApxInfHandler)
    sys.stderr.write('Server listening on port ' + str(port) + '\n')
    server.serve_forever()
