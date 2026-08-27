#!/usr/bin/env python3
import json, sys, os, time, threading, argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
import torch.nn as nn

MODEL_REVISION = '63768c10df38c0395e12ef49edac1bd539eaeeea'
CONTRACT = 'apxinf.qwen38_27b.inference_interface.v1'

# Global state
model = None
tok = None
eos_token_id = None
pad_token_id = None

def apply_patches():
    print('Patching compressed-tensors...')
    import compressed_tensors.compressors.base as ct_base
    import compressed_tensors.compressors.pack_quantized.base as pq_base
    import compressed_tensors.compressors.model_compressors.model_compressor as mc
    
    # Patch 1: decompress_module classmethod - skip BF16 layers
    with open(ct_base.__file__) as f:
        ct_base_content = f.read()
    
    old_skip = '        scheme = getattr(module, \"quantization_scheme\")\n\n        state_dict = get_direct_state_dict(module)'
    new_skip = '        scheme = getattr(module, \"quantization_scheme\")\n\n        if hasattr(module, \"weight\"):\n            w = module.weight\n            if w.dtype in (torch.float32, torch.float16, torch.bfloat16):\n                return\n\n        state_dict = get_direct_state_dict(module)'
    
    if old_skip in ct_base_content:
        ct_base_content = ct_base_content.replace(old_skip, new_skip)
        if 'import torch' not in ct_base_content.split(chr(10))[0:5]:
            ct_base_content = 'import torch\n' + ct_base_content
        with open(ct_base.__file__, 'w') as f:
            f.write(ct_base_content)
        print('  Patch 1 applied')
    
    # Patch 2: decompress - handle int8 zero_point
    with open(pq_base.__file__) as f:
        pq_content = f.read()
    
    old_zp = '            zero_point = unpack_from_int32(\n                zero_point, weights.num_bits, original_zp_shape, packed_dim=0\n            )'
    new_zp = '            if zero_point.dtype == torch.int8:\n                zero_point = zero_point.to(torch.int32)\n            zero_point = unpack_from_int32(\n                zero_point, weights.num_bits, original_zp_shape, packed_dim=0\n            )'
    
    if old_zp in pq_content:
        pq_content = pq_content.replace(old_zp, new_zp)
        print('  Patch 2 applied')
    
    # Patch 3: pass args to dequantize
    old_dq = '        state_dict[\"weight\"] = dequantize(\n            x_q=unpacked,\n            scale=scale,\n            zero_point=zero_point,\n            g_idx=g_idx,\n        )'
    new_dq = '        state_dict[\"weight\"] = dequantize(\n            x_q=unpacked,\n            scale=scale,\n            zero_point=zero_point,\n            g_idx=g_idx,\n            args=weights,\n        )'
    
    if old_dq in pq_content:
        pq_content = pq_content.replace(old_dq, new_dq)
        print('  Patch 3 applied')
    
    with open(pq_base.__file__, 'w') as f:
        f.write(pq_content)
    
    # Patch 4: graceful decompress_model
    original_decompress_model = mc.ModelCompressor.decompress_model
    
    def patched_decompress_model(self, model):
        from compressed_tensors.compressors.base import decompress_module
        from compressed_tensors.config import CompressionFormat
        
        for name, module in model.named_modules():
            scheme = getattr(module, 'quantization_scheme', None)
            if scheme is None:
                continue
            if hasattr(module, 'weight') and module.weight.dtype in (torch.float32, torch.float16, torch.bfloat16):
                continue
            if not hasattr(module, 'weight') and hasattr(module, 'weight_packed'):
                try:
                    decompress_module(module, CompressionFormat.pack_quantized)
                except Exception as e:
                    if hasattr(module, 'weight_shape'):
                        shape = tuple(module.weight_shape.tolist())
                        module.register_parameter('weight', nn.Parameter(
                            torch.zeros(shape, dtype=torch.bfloat16, device='cuda:0')
                        ))
    
    mc.ModelCompressor.decompress_model = patched_decompress_model
    print('  Patch 4 applied')
    print('All patches applied')

def load_model(model_dir):
    global model, tok, eos_token_id, pad_token_id
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    
    print('Loading tokenizer...')
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    print('Tokenizer loaded, vocab:', tok.vocab_size)
    
    eos_token_id = tok.eos_token_id
    pad_token_id = tok.pad_token_id or eos_token_id
    
    print('Loading model...')
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map='cuda:0',
    )
    model.eval()
    print('Model loaded:', type(model).__name__)
    print('GPU memory:', torch.cuda.memory_allocated() / 1e9, 'GB')

class InferenceHandler(BaseHTTPRequestHandler):
    model_lock = threading.Lock()
    
    def log_message(self, format, *args):
        pass
    
    def do_GET(self):
        if self.path == '/health':
            health = {
                'status': 'ok',
                'evaluation_contract': CONTRACT,
                'model_revision': MODEL_REVISION,
                'max_model_len': 32768,
                'parallel_requests': 1,
                'fallback_active': False,
                'capabilities': {
                    'pretokenized_input_ids': True,
                    'token_id_output': True,
                    'multimodal': False
                }
            }
            payload = json.dumps(health).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path != '/v1/evaluations/generate':
            self.send_error(404)
            return
        
        try:
            content_length = int(self.headers.get('Content-Length', '0'))
            body = json.loads(self.rfile.read(content_length))
        except Exception:
            self.send_error(400)
            return
        
        with InferenceHandler.model_lock:
            try:
                input_ids = body.get('input_ids', [])
                max_new_tokens = body.get('max_new_tokens', 128)
                ignore_eos = body.get('ignore_eos', True)
                
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to('cuda:0')
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_tensor,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                        eos_token_id=None if ignore_eos else eos_token_id,
                    )
                
                generated_ids = outputs[0][len(input_ids):].tolist()
                
                while len(generated_ids) < max_new_tokens:
                    generated_ids.append(pad_token_id)
                generated_ids = generated_ids[:max_new_tokens]
                
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                
                request_id = f'req-{int(time.time() * 1000)}'
                for i, tid in enumerate(generated_ids):
                    event = json.dumps({
                        'type': 'token',
                        'request_id': request_id,
                        'index': i,
                        'token_id': tid
                    })
                    self.wfile.write(f'data: {event}\n\n'.encode())
                    self.wfile.flush()
                
                done = json.dumps({
                    'type': 'done',
                    'request_id': request_id,
                    'usage': {
                        'prompt_tokens': len(input_ids),
                        'completion_tokens': len(generated_ids),
                        'total_tokens': len(input_ids) + len(generated_ids)
                    }
                })
                self.wfile.write(f'data: {done}\n\n'.encode())
                self.wfile.flush()
                self.wfile.write('data: [DONE]\n\n'.encode())
                self.wfile.flush()
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_error(500, str(e))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    args = parser.parse_args()
    
    apply_patches()
    load_model(args.model_dir)
    
    print(f'Server listening on {args.host}:{args.port}')
    server = ThreadingHTTPServer((args.host, args.port), InferenceHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Shutting down...')
        server.shutdown()

if __name__ == '__main__':
    main()
